"""Product bases over single-mode spaces, with optional sector restriction."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from operator_algebra._types import as_complex
from operator_algebra.algebra import AlgebraRules, _mode_sort_key
from operator_algebra.operators import Monomial, OpSum
from operator_algebra.solve.spaces import ComplexArray, IntArray, LocalSpace

_CHUNK = 2**20

type Levels = NDArray[np.unsignedinteger[Any]]


@dataclass
class Basis:
    """An ordered product basis over the modes of one or more spaces.

    Attributes
    ----------
    modes : tuple[tuple[AlgebraRules, Hashable], ...]
        ``(algebra, label)`` of each mode, grouped by space.
    spaces : tuple[LocalSpace, ...]
        Space of each mode.
    starts : tuple[int, ...]
        Position of the first mode in each mode's space group.
    mode_index : dict[tuple[AlgebraRules, Hashable], int]
        Map ``(algebra, label) -> position`` in ``modes``.
    levels : ndarray, shape (dim, len(modes))
        One row per basis state: the level of each mode.
    strides : ndarray
        Weights turning a row of levels into its position in the full
        product space, ``code = levels @ strides``.
    codes : ndarray
        Code of each basis state, in increasing order.
    dim : int
        Number of basis states.

    """

    modes: tuple[tuple[AlgebraRules, Hashable], ...]
    spaces: tuple[LocalSpace, ...]
    starts: tuple[int, ...]
    mode_index: dict[tuple[AlgebraRules, Hashable], int]
    levels: Levels
    strides: IntArray
    codes: IntArray
    dim: int


def enumerate_basis(
    spaces: Mapping[LocalSpace, Sequence[Hashable]],
    *,
    sectors: Mapping[OpSum, complex] | None = None,
    predicate: Callable[[Levels], NDArray[np.bool_]] | None = None,
) -> Basis:
    """Construct the product basis of the given modes.

    Parameters
    ----------
    spaces : Mapping[LocalSpace, Sequence[Hashable]]
        Each space mapped to its mode labels, e.g.
        ``{FermionSpace(): ["up", "dn"], BosonSpace(n_max=5): [0]}``.
        Labels are put in canonical order within each space.
    sectors : Mapping[OpSum, complex], optional
        Quantum-number operators mapped to their target eigenvalues. Each
        operator must be diagonal in the level basis.
    predicate : Callable, optional
        Extra filter applied after sector restriction. Receives an array of
        states (one row of levels each) and returns which to keep.

    """
    modes: list[tuple[AlgebraRules, Hashable]] = []
    mode_spaces: list[LocalSpace] = []
    starts: list[int] = []
    for space, labels in spaces.items():
        start = len(modes)
        for label in sorted(labels, key=_mode_sort_key):
            modes.append((space.rules, label))
            mode_spaces.append(space)
            starts.append(start)
    mode_index = {m: i for i, m in enumerate(modes)}
    if len(mode_index) < len(modes):
        msg = "A mode is listed more than once."
        raise ValueError(msg)
    sectors = sectors or {}
    for qn_op in sectors:
        check_modes(qn_op, mode_index)

    dims = [s.dim for s in mode_spaces]
    total = math.prod(dims)
    if total > np.iinfo(np.int64).max:
        msg = "The full product space has too many states to index."
        raise ValueError(msg)
    strides = np.array([math.prod(dims[k + 1 :]) for k in range(len(dims))])
    dtype = np.min_scalar_type(max(dims) - 1)

    # walk the full product space in chunks so memory stays bounded
    kept = []
    for first in range(0, total, _CHUNK):
        codes = np.arange(first, min(first + _CHUNK, total))
        levels = np.stack(np.unravel_index(codes, dims), axis=1).astype(dtype)
        keep = np.ones(len(levels), dtype=bool)
        for qn_op, target in sectors.items():
            value = _diagonal_values(qn_op, levels, mode_index, mode_spaces, starts)
            keep &= np.isclose(value, target, rtol=1e-9, atol=1e-10)
        if predicate is not None:
            keep &= predicate(levels)
        kept.append(levels[keep])

    levels = np.concatenate(kept)
    return Basis(
        modes=tuple(modes),
        spaces=tuple(mode_spaces),
        starts=tuple(starts),
        mode_index=mode_index,
        levels=levels,
        strides=strides,
        codes=levels @ strides,
        dim=len(levels),
    )


def apply_monomial(
    mono: Monomial,
    levels: Levels,
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> tuple[IntArray, Levels, ComplexArray]:
    """Apply ``mono`` to every row of ``levels``.

    Returns the rows ``mono`` does not annihilate, their new levels and
    their amplitudes. Annihilated rows are dropped as soon as they appear so
    later operators skip them.
    """
    rows = np.arange(len(levels))
    levels = levels.copy()
    amp = np.ones(len(levels), dtype=complex)
    for op in reversed(mono.ops):
        k = mode_index[(op.rules, op.mode)]
        space, start = spaces[k], starts[k]
        new_level, factor = space.action(op)
        step = factor[levels[:, k]]
        alive = step != 0
        if not alive.all():
            rows, levels, amp, step = (
                rows[alive],
                levels[alive],
                amp[alive],
                step[alive],
            )
        amp *= step
        string = space.string_factor(op)
        if start < k and np.any(string != 1):
            amp *= np.prod(string[levels[:, start:k]], axis=1)
        levels[:, k] = new_level[levels[:, k]]
    return rows, levels, amp


def state_index(basis: Basis, levels: ArrayLike) -> int | IntArray:
    """Position in ``basis`` of a state, or of each row of states.

    Example: ``state_index(basis, (1, 0, 0, 0))``.
    """
    rows = np.asarray(levels)
    codes = rows @ basis.strides
    pos = np.searchsorted(basis.codes, codes)
    if np.any(pos == basis.dim) or np.any(basis.codes[pos % basis.dim] != codes):
        msg = "State is not in the basis."
        raise ValueError(msg)
    return int(pos) if rows.ndim == 1 else pos


def check_modes(
    op: OpSum, mode_index: Mapping[tuple[AlgebraRules, Hashable], int]
) -> None:
    """Raise ValueError if ``op`` acts on a mode that is not in the basis."""
    for mono in op.terms:
        for single_op in mono.ops:
            if (single_op.rules, single_op.mode) not in mode_index:
                msg = (
                    f"Operator references mode {single_op.mode!r} of "
                    f"{single_op.rules!r}, which is not in the basis."
                )
                raise ValueError(msg)


def _diagonal_values(
    op: OpSum,
    levels: Levels,
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> ComplexArray:
    total = np.zeros(len(levels), dtype=complex)
    for mono, coeff in op.terms.items():
        rows, new, amp = apply_monomial(mono, levels, mode_index, spaces, starts)
        moved = np.any(new != levels[rows], axis=1)
        if np.any(moved):
            msg = (
                "Sector operator is not diagonal in the level basis: it "
                f"changes state {levels[rows[moved][0]].tolist()}."
            )
            raise ValueError(msg)
        total[rows] += as_complex(coeff) * amp
    return total
