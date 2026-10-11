"""Product bases over single-mode spaces, with optional sector restriction."""

from __future__ import annotations

import math
import warnings
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
_LARGE_PRODUCT = 2**22
_TOL = 1e-9

type Levels = NDArray[np.unsignedinteger[Any]]
type FloatArray = NDArray[np.float64]
type ChargeTables = tuple[list[FloatArray], float]


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
        operator must be diagonal in the level basis. Sums of single-mode
        terms (particle number, Sz, ...) are used to build only the states
        in the sector. Other operators, and ``predicate``, filter those
        states afterwards, so without a single-mode sector every product
        state is built first, which can take a lot of memory.
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
    additive: list[tuple[ChargeTables, float]] = []
    others: dict[OpSum, complex] = {}
    for qn_op, target in (sectors or {}).items():
        check_modes(qn_op, mode_index)
        tables = _charge_tables(qn_op, mode_index, mode_spaces, starts)
        if tables is None:
            others[qn_op] = target
        else:
            additive.append((tables, complex(target).real))

    dims = [s.dim for s in mode_spaces]
    total = math.prod(dims)
    if total > np.iinfo(np.int64).max:
        msg = "The full product space has too many states to index."
        raise ValueError(msg)
    strides = np.array([math.prod(dims[k + 1 :]) for k in range(len(dims))])
    if not additive and (others or predicate) and total > _LARGE_PRODUCT:
        msg = (
            f"Building all {total:,} product states before filtering. Adding a "
            "sector such as particle number builds only the states in it."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    levels = _generate(dims, additive)
    if others or predicate is not None:
        kept = [levels[:0]]
        for first in range(0, len(levels), _CHUNK):
            chunk = levels[first : first + _CHUNK]
            keep = np.ones(len(chunk), dtype=bool)
            for qn_op, target in others.items():
                value = _diagonal_values(qn_op, chunk, mode_index, mode_spaces, starts)
                keep &= np.isclose(value, target, rtol=_TOL, atol=_TOL)
            if predicate is not None:
                keep &= predicate(chunk)
            kept.append(chunk[keep])
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


def _charge_tables(
    op: OpSum,
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> ChargeTables | None:
    """Split ``op`` into a value per level of each mode, plus a constant.

    Returns None if a term acts on more than one mode or a value is not real.
    """
    tables = [np.zeros(s.dim) for s in spaces]
    constant = 0.0
    for mono, coeff in op.terms.items():
        value = as_complex(coeff)
        touched = {mode_index[(o.rules, o.mode)] for o in mono.ops}
        if not touched:
            if value.imag:
                return None
            constant += value.real
            continue
        if len(touched) > 1:
            return None
        (k,) = touched
        # one row per level of mode k; earlier modes empty, so no string phase
        levels = np.zeros((spaces[k].dim, len(spaces)), dtype=np.uint8)
        levels[:, k] = np.arange(spaces[k].dim)
        rows, new, amp = apply_monomial(mono, levels, mode_index, spaces, starts)
        if np.any(new != levels[rows]):
            msg = (
                "Sector operator is not diagonal in the level basis: it "
                f"changes state {levels[rows][0].tolist()}."
            )
            raise ValueError(msg)
        contribution = value * amp
        if np.any(contribution.imag):
            return None
        tables[k][rows] += contribution.real
    return tables, constant


def _generate(
    dims: Sequence[int], additive: Sequence[tuple[ChargeTables, float]]
) -> Levels:
    """Build the product states whose additive charges hit their targets.

    Modes are added one at a time, dropping prefixes whose charge can no
    longer reach a target given what the remaining modes can add. Rows come
    out in product order.
    """
    dtype = np.min_scalar_type(max(dims) - 1)
    tables = [t for (t, _), _ in additive]
    targets = [target for _, target in additive]
    charges = [np.full(1, constant) for (_, constant), _ in additive]
    # least and most the modes from k on can still add, per sector
    lows = [_suffix_sums([t.min() for t in ts]) for ts in tables]
    highs = [_suffix_sums([t.max() for t in ts]) for ts in tables]

    levels = np.zeros((1, 0), dtype=dtype)
    for k, dim in enumerate(dims):
        levels = np.hstack(
            [
                np.repeat(levels, dim, axis=0),
                np.tile(np.arange(dim, dtype=dtype), len(levels))[:, None],
            ]
        )
        keep = np.ones(len(levels), dtype=bool)
        for i in range(len(additive)):
            charges[i] = np.repeat(charges[i], dim) + np.tile(
                tables[i][k], len(charges[i])
            )
            keep &= charges[i] + lows[i][k + 1] <= targets[i] + _TOL
            keep &= charges[i] + highs[i][k + 1] >= targets[i] - _TOL
        levels = levels[keep]
        charges = [q[keep] for q in charges]
    return levels


def _suffix_sums(values: Sequence[float]) -> FloatArray:
    """``out[k] = sum(values[k:])``, with ``out[len(values)] = 0``."""
    return np.append(np.cumsum(values[::-1])[::-1], 0.0)


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
