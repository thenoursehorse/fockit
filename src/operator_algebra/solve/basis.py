"""Product bases over single-mode spaces, optionally restricted by symmetries."""

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
from operator_algebra.normal_order import normal_order
from operator_algebra.operators import Monomial, OpSum, zero
from operator_algebra.simplify import drop_zeros
from operator_algebra.solve.spaces import ComplexArray, IntArray, LocalSpace
from operator_algebra.transforms import substitute_modes

_CHUNK = 2**20
_LARGE_PRODUCT = 2**22
_TOL = 1e-9

type Levels = NDArray[np.unsignedinteger[Any]]
type FloatArray = NDArray[np.float64]
type ChargeTables = tuple[list[FloatArray], float]
type PermutationSymmetry = tuple[Mapping[Hashable, Hashable], complex]
type Swaps = tuple[IntArray, IntArray]


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
    permutation_symmetries : tuple[PermutationSymmetry, ...]
        The ``(label_map, eigenvalue)`` pairs the basis was built with.
    group : tuple[ndarray, ...]
        Every symmetry operation as a column permutation, identity first.
        ``group[g][k]`` is where mode ``k`` is sent.
    characters : ndarray
        Eigenvalue of each operation in ``group``.
    swaps : tuple[tuple[ndarray, ndarray], ...]
        For each operation, the pairs of modes whose order it reverses and
        whose space has a nontrivial exchange phase.
    norms : ndarray
        Norm of the symmetric state built on each basis state.

    Notes
    -----
    With permutation symmetries, each basis state stands for the combination
    of its orbit, ``sum_g conj(character_g) g|state>``, and ``levels`` holds
    the orbit representative (the member with the smallest code).

    """

    modes: tuple[tuple[AlgebraRules, Hashable], ...]
    spaces: tuple[LocalSpace, ...]
    starts: tuple[int, ...]
    mode_index: dict[tuple[AlgebraRules, Hashable], int]
    levels: Levels
    strides: IntArray
    codes: IntArray
    dim: int
    permutation_symmetries: tuple[PermutationSymmetry, ...]
    group: tuple[IntArray, ...]
    characters: ComplexArray
    swaps: tuple[Swaps, ...]
    norms: FloatArray


def enumerate_basis(
    spaces: Mapping[LocalSpace, Sequence[Hashable]],
    *,
    diagonal_symmetries: Mapping[OpSum, complex] | None = None,
    permutation_symmetries: Sequence[PermutationSymmetry] = (),
    predicate_function: Callable[[Levels], NDArray[np.bool_]] | None = None,
) -> Basis:
    """Construct the product basis of the given modes.

    Parameters
    ----------
    spaces : Mapping[LocalSpace, Sequence[Hashable]]
        Each space mapped to its mode labels, e.g.
        ``{FermionSpace(): ["up", "dn"], BosonSpace(n_max=5): [0]}``.
        Labels are put in canonical order within each space.
    diagonal_symmetries : Mapping[OpSum, complex], optional
        Symmetry operators that are diagonal in the level basis, mapped to
        the eigenvalue to keep, e.g. ``{N: 6, Sz: 0}``. Sums of single-mode
        terms (particle number, Sz, ...) are used to build only the states
        with those eigenvalues. Other operators, and ``predicate_function``,
        filter those states afterwards, so without a single-mode one every
        product state is built first, which can take a lot of memory.
    permutation_symmetries : Sequence[tuple[Mapping, complex]], optional
        Mode relabellings and the eigenvalue to keep for each, e.g.
        translation ``({(i, s): ((i + 1) % L, s), ...}, np.exp(2j * np.pi *
        k / L))``. A map applies to the labels of every space; unmapped labels
        stay. The maps must commute with each other and map the states
        selected by ``diagonal_symmetries`` and ``predicate_function`` onto
        themselves.
    predicate_function : Callable, optional
        Extra filter applied last. Receives an array of states (one row of
        levels each) and returns which to keep.

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
    group, characters = _group(permutation_symmetries, modes, mode_index, mode_spaces)
    swaps = tuple(_swaps(perm, mode_spaces, starts) for perm in group)
    additive: list[tuple[ChargeTables, float]] = []
    others: dict[OpSum, complex] = {}
    for qn_op, target in (diagonal_symmetries or {}).items():
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
    if not additive and (others or predicate_function) and total > _LARGE_PRODUCT:
        msg = (
            f"Building all {total:,} product states before filtering. Adding a "
            "diagonal symmetry such as particle number builds only the states in it."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    levels = _generate(dims, additive)
    if others or predicate_function is not None:
        kept = [levels[:0]]
        for first in range(0, len(levels), _CHUNK):
            chunk = levels[first : first + _CHUNK]
            keep = np.ones(len(chunk), dtype=bool)
            for qn_op, target in others.items():
                value = _diagonal_values(qn_op, chunk, mode_index, mode_spaces, starts)
                keep &= np.isclose(value, target, rtol=_TOL, atol=_TOL)
            if predicate_function is not None:
                keep &= predicate_function(chunk)
            kept.append(chunk[keep])
        levels = np.concatenate(kept)

    norms = np.ones(len(levels))
    if len(group) > 1:
        levels, norms = _orbit_representatives(
            levels, group, characters, swaps, mode_spaces, strides
        )

    return Basis(
        modes=tuple(modes),
        spaces=tuple(mode_spaces),
        starts=tuple(starts),
        mode_index=mode_index,
        levels=levels,
        strides=strides,
        codes=levels @ strides,
        dim=len(levels),
        permutation_symmetries=tuple(permutation_symmetries),
        group=group,
        characters=characters,
        swaps=swaps,
        norms=norms,
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


def representatives(
    basis: Basis, levels: Levels
) -> tuple[IntArray, ComplexArray, IntArray]:
    """Map each row of ``levels`` to its orbit representative.

    Returns the representative's code, the operation that reaches it, and
    the phase that operation picks up: ``g|row> = phase |representative>``.
    """
    codes = np.stack([levels @ basis.strides[perm] for perm in basis.group])
    element = codes.argmin(axis=0)
    phase = np.ones(len(levels), dtype=complex)
    for g in np.unique(element):
        rows = element == g
        phase[rows] = _exchange_phase(levels[rows], basis.swaps[g], basis.spaces)
    return codes[element, np.arange(len(levels))], phase, element


def check_symmetric(
    op: OpSum, permutation_symmetries: Sequence[PermutationSymmetry]
) -> None:
    """Raise ValueError if ``op`` changes under any of the mode relabellings."""
    for label_map, _ in permutation_symmetries:
        change = normal_order(substitute_modes(op, label_map) - op)
        if drop_zeros(change, tol=1e-10) != zero():
            msg = f"Operator is not symmetric under the relabelling {label_map!r}."
            raise ValueError(msg)


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
    # least and most the modes from k on can still add, per symmetry
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


def _group(
    permutation_symmetries: Sequence[PermutationSymmetry],
    modes: Sequence[tuple[AlgebraRules, Hashable]],
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
) -> tuple[tuple[IntArray, ...], ComplexArray]:
    """Every product of the generating relabellings, with its eigenvalue."""
    identity = np.arange(len(modes))
    generators = []
    for label_map, eigenvalue in permutation_symmetries:
        perm = np.array(
            [
                mode_index.get((rules, label_map.get(label, label)), -1)
                for rules, label in modes
            ]
        )
        if np.any(perm < 0) or len(set(perm.tolist())) < len(perm):
            msg = f"Relabelling {label_map!r} is not a one-to-one map of the modes."
            raise ValueError(msg)
        if any(spaces[k] != spaces[j] for k, j in enumerate(perm)):
            msg = f"Relabelling {label_map!r} moves a mode to a different space."
            raise ValueError(msg)
        order, power = 1, perm
        while not np.array_equal(power, identity):
            power, order = perm[power], order + 1
        if not np.isclose(complex(eigenvalue) ** order, 1):
            msg = f"Eigenvalue {eigenvalue} is not a root of unity of order {order}."
            raise ValueError(msg)
        generators.append((perm, complex(eigenvalue), order))
    for a, _, _ in generators:
        for b, _, _ in generators:
            if not np.array_equal(a[b], b[a]):
                msg = "Symmetries must commute with each other."
                raise ValueError(msg)

    elements: dict[tuple[int, ...], complex] = {tuple(identity): 1}
    for perm, eigenvalue, order in generators:
        grown = dict(elements)
        for element, character in elements.items():
            power = np.asarray(element)
            for n in range(1, order):
                power = perm[power]
                key = tuple(power.tolist())
                value = character * eigenvalue**n
                if key in grown and not np.isclose(grown[key], value):
                    msg = "Eigenvalues are inconsistent: the sector is empty."
                    raise ValueError(msg)
                grown[key] = value
        elements = grown
    group = tuple(np.array(key) for key in elements)
    return group, np.array(list(elements.values()), dtype=complex)


def _swaps(
    perm: IntArray, spaces: Sequence[LocalSpace], starts: Sequence[int]
) -> Swaps:
    """Pairs of modes in the same space whose order ``perm`` reverses."""
    a, b = np.triu_indices(len(perm), 1)
    keep = (perm[a] > perm[b]) & (np.asarray(starts)[a] == np.asarray(starts)[b])
    nontrivial = [bool(np.any(spaces[k].exchange_factor() != 1)) for k in a.tolist()]
    keep &= np.array(nontrivial, dtype=bool)
    return a[keep], b[keep]


def _exchange_phase(
    levels: Levels, swaps: Swaps, spaces: Sequence[LocalSpace]
) -> ComplexArray:
    """Phase from reordering the modes of each row, one swapped pair at a time."""
    phase = np.ones(len(levels), dtype=complex)
    for a, b in zip(*swaps, strict=True):
        phase *= spaces[a].exchange_factor()[levels[:, a], levels[:, b]]
    return phase


def _orbit_representatives(
    levels: Levels,
    group: Sequence[IntArray],
    characters: ComplexArray,
    swaps: Sequence[Swaps],
    spaces: Sequence[LocalSpace],
    strides: IntArray,
) -> tuple[Levels, FloatArray]:
    """Keep orbit representatives whose symmetric combination is nonzero."""
    selected = levels @ strides
    kept, kept_norms = [levels[:0]], [np.zeros(0)]
    for first in range(0, len(levels), _CHUNK):
        chunk = levels[first : first + _CHUNK]
        codes = np.stack([chunk @ strides[perm] for perm in group])
        pos = np.minimum(np.searchsorted(selected, codes), len(selected) - 1)
        if np.any(selected[pos] != codes):
            msg = (
                "A permutation symmetry maps some selected states outside those "
                "kept by diagonal_symmetries or predicate_function."
            )
            raise ValueError(msg)
        own = codes[0]
        reps = codes.min(axis=0) == own
        chunk, codes, own = chunk[reps], codes[:, reps], own[reps]
        # norm = |G| sum over operations that fix the state
        norm = np.zeros(len(chunk), dtype=complex)
        for g in range(len(group)):
            fixed = codes[g] == own
            phase = _exchange_phase(chunk[fixed], swaps[g], spaces)
            norm[fixed] += np.conj(characters[g]) * phase
        norm = norm.real * len(group)
        nonzero = norm > _TOL
        kept.append(chunk[nonzero])
        kept_norms.append(norm[nonzero])
    return np.concatenate(kept), np.concatenate(kept_norms)


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
