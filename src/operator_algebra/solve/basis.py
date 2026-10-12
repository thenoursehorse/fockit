"""Product bases over single-mode spaces, optionally restricted by symmetries."""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from operator_algebra._types import as_complex
from operator_algebra.algebra import AlgebraRules, mode_sort_key
from operator_algebra.normal_order import normal_order
from operator_algebra.operators import Monomial, Op, OpSum, identity, zero
from operator_algebra.simplify import drop_zeros
from operator_algebra.solve.spaces import ComplexArray, IntArray, LocalSpace
from operator_algebra.transforms import hc, substitute_modes

_CHUNK = 2**20
_LARGE_PRODUCT = 2**22
_TOL = 1e-9
_MAX_ORDER = 64

type Levels = NDArray[np.integer[Any]]
type FloatArray = NDArray[np.float64]
type ChargeTables = tuple[list[FloatArray], float]
type PermutationSymmetry = tuple[Mapping[Hashable, Hashable], complex]
type OnsiteSymmetry = tuple[Sequence[OpSum], complex]
type OnsiteTable = tuple[int, IntArray, ComplexArray, ComplexArray]
type Swaps = tuple[IntArray, IntArray]
type Action = Callable[[Levels], tuple[Levels, ComplexArray]]


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
        Mode permutation of every symmetry operation, identity first.
        Operation ``g`` applies its on-site symmetries, then sends mode ``k``
        to ``group[g][k]``.
    characters : ndarray
        Eigenvalue of each operation in ``group``.
    swaps : tuple[tuple[ndarray, ndarray], ...]
        For each operation, the pairs of modes whose order it reverses and
        whose space has a nontrivial exchange phase.
    onsite_symmetries : tuple[OnsiteSymmetry, ...]
        The ``(factors, eigenvalue)`` pairs the basis was built with.
    onsite_tables : tuple[tuple[OnsiteTable, ...], ...]
        For each on-site symmetry and each mode it acts on: the mode, and per
        level the new level, the amplitude and the string factors.
    onsite_powers : tuple[tuple[int, ...], ...]
        Power of each on-site symmetry in each operation of ``group``.
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
    onsite_symmetries: tuple[OnsiteSymmetry, ...]
    onsite_tables: tuple[tuple[OnsiteTable, ...], ...]
    onsite_powers: tuple[tuple[int, ...], ...]
    norms: FloatArray


def enumerate_basis(
    spaces: Mapping[LocalSpace, Sequence[Hashable]],
    *,
    diagonal_symmetries: Mapping[OpSum, complex] | None = None,
    permutation_symmetries: Sequence[PermutationSymmetry] = (),
    onsite_symmetries: Sequence[OnsiteSymmetry] = (),
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
        stay. The maps must map the states selected by ``diagonal_symmetries``
        and ``predicate_function`` onto themselves. They need not commute,
        but the eigenvalues must fit together, e.g. reflection with
        translation only at momentum 0 or pi. States that only appear in
        degenerate multiplets of non-commuting symmetries are in no sector.
    onsite_symmetries : Sequence[tuple[Sequence[OpSum], complex]], optional
        Products of single-mode operators and the eigenvalue to keep, e.g.
        spin inversion ``([sigma_x(i) for i in range(L)], 1)`` or
        particle-hole ``([c(i) + (-1) ** i * cdag(i) for i in range(L)], 1)``.
        Each factor acts on one mode and sends every level to one level with
        a phase; the product is taken in the basis's mode order. They must
        commute with each other and with the permutation symmetries on the
        selected states, and map those states onto themselves.
    predicate_function : Callable, optional
        Extra filter applied last. Receives an array of states (one row of
        levels each) and returns which to keep.

    """
    modes: list[tuple[AlgebraRules, Hashable]] = []
    mode_spaces: list[LocalSpace] = []
    starts: list[int] = []
    for space, labels in spaces.items():
        start = len(modes)
        for label in sorted(labels, key=mode_sort_key):
            modes.append((space.rules, label))
            mode_spaces.append(space)
            starts.append(start)
    mode_index = {m: i for i, m in enumerate(modes)}
    if len(mode_index) < len(modes):
        msg = "A mode is listed more than once."
        raise ValueError(msg)
    group, characters, generators = _group(
        permutation_symmetries, modes, mode_index, mode_spaces
    )
    onsite_tables = tuple(
        _onsite_tables(factors, mode_index, mode_spaces)
        for factors, _ in onsite_symmetries
    )
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

    onsite_powers = tuple((0,) * len(onsite_tables) for _ in group)
    if onsite_symmetries and len(levels):
        group, characters, onsite_powers = _with_onsite(
            levels,
            group,
            characters,
            generators,
            onsite_symmetries,
            onsite_tables,
            mode_spaces,
            starts,
        )
    swaps = tuple(_swaps(perm, mode_spaces, starts) for perm in group)

    norms = np.ones(len(levels))
    if len(group) > 1:
        levels, norms = _orbit_representatives(
            levels,
            group,
            characters,
            swaps,
            onsite_powers,
            onsite_tables,
            mode_spaces,
            starts,
            strides,
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
        onsite_symmetries=tuple(onsite_symmetries),
        onsite_tables=onsite_tables,
        onsite_powers=onsite_powers,
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


def expand_state(state: ArrayLike, basis: Basis, full_basis: Basis) -> ComplexArray:
    """Write states of ``basis`` in the occupation states of ``full_basis``.

    ``state`` has shape ``(basis.dim,)`` or ``(basis.dim, k)``. ``full_basis``
    must have the same modes and contain every state of each orbit, e.g. the
    same call to ``enumerate_basis`` without ``permutation_symmetries``.
    """
    if full_basis.modes != basis.modes:
        msg = "full_basis must have the same modes as basis."
        raise ValueError(msg)
    psi = np.asarray(state)
    weights = psi / np.sqrt(basis.norms).reshape(-1, *[1] * (psi.ndim - 1))
    out = np.zeros((full_basis.dim, *psi.shape[1:]), dtype=complex)
    images = _onsite_images(
        basis.levels, basis.onsite_powers, basis.onsite_tables, basis.starts
    )
    for g, perm in enumerate(basis.group):
        moved, onsite_phase = images[basis.onsite_powers[g]]
        codes = moved @ basis.strides[perm]
        pos = np.searchsorted(full_basis.codes, codes)
        if np.any(pos == full_basis.dim) or np.any(
            full_basis.codes[pos % full_basis.dim] != codes
        ):
            msg = "full_basis is missing states that basis is built from."
            raise ValueError(msg)
        phase = (
            np.conj(basis.characters[g])
            * onsite_phase
            * _exchange_phase(moved, basis.swaps[g], basis.spaces)
        )
        np.add.at(out, pos, weights * phase.reshape(-1, *[1] * (psi.ndim - 1)))
    return out


def representatives(
    basis: Basis, levels: Levels
) -> tuple[IntArray, ComplexArray, IntArray]:
    """Map each row of ``levels`` to its orbit representative.

    Returns the representative's code, the operation that reaches it, and
    the phase that operation picks up: ``g|row> = phase |representative>``.
    """
    images = _onsite_images(
        levels, basis.onsite_powers, basis.onsite_tables, basis.starts
    )
    codes = np.stack(
        [
            images[powers][0] @ basis.strides[perm]
            for perm, powers in zip(basis.group, basis.onsite_powers, strict=True)
        ]
    )
    element = codes.argmin(axis=0)
    phase = np.ones(len(levels), dtype=complex)
    for g in np.unique(element):
        rows = element == g
        moved, onsite_phase = images[basis.onsite_powers[g]]
        phase[rows] = onsite_phase[rows] * _exchange_phase(
            moved[rows], basis.swaps[g], basis.spaces
        )
    return codes[element, np.arange(len(levels))], phase, element


def check_symmetric(
    op: OpSum,
    permutation_symmetries: Sequence[PermutationSymmetry],
    onsite_symmetries: Sequence[OnsiteSymmetry] = (),
) -> None:
    """Raise ValueError if ``op`` changes under any of the symmetries."""
    for label_map, _ in permutation_symmetries:
        change = normal_order(substitute_modes(op, label_map) - op)
        if drop_zeros(change, tol=1e-10) != zero():
            msg = f"Operator is not symmetric under the relabelling {label_map!r}."
            raise ValueError(msg)
    for factors, _ in onsite_symmetries:
        change = normal_order(_conjugate(op, factors) - op)
        if drop_zeros(change, tol=1e-10) != zero():
            msg = "Operator is not symmetric under an on-site symmetry."
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
) -> tuple[tuple[IntArray, ...], ComplexArray, list[IntArray]]:
    """Every product of the generating relabellings, with its eigenvalue.

    Also returns the generating relabellings themselves.
    """
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
        generators.append((perm, complex(eigenvalue)))

    # multiply the known operations by each generator until nothing new appears
    elements: dict[tuple[int, ...], complex] = {tuple(identity): 1}
    frontier = [identity]
    while frontier:
        grown = []
        for element in frontier:
            character = elements[tuple(element.tolist())]
            for perm, eigenvalue in generators:
                product = perm[element]
                key = tuple(product.tolist())
                if key not in elements:
                    elements[key] = character * eigenvalue
                    grown.append(product)
                elif not np.isclose(elements[key], character * eigenvalue):
                    msg = (
                        "Eigenvalues don't fit the symmetries, so the sector is "
                        "empty. For example, each eigenvalue must be a root of "
                        "unity of its symmetry's order, and reflection with "
                        "translation needs momentum 0 or pi."
                    )
                    raise ValueError(msg)
        frontier = grown
    group = tuple(np.array(key) for key in elements)
    characters = np.array(list(elements.values()), dtype=complex)
    return group, characters, [perm for perm, _ in generators]


def _onsite_tables(
    factors: Sequence[OpSum],
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
) -> tuple[OnsiteTable, ...]:
    """Per mode: the new level, amplitude and string factors of each level."""
    tables = []
    for factor in factors:
        check_modes(factor, mode_index)
        touched = {
            mode_index[(o.rules, o.mode)] for mono in factor.terms for o in mono.ops
        }
        if len(touched) != 1:
            msg = "Each on-site factor must act on exactly one mode."
            raise ValueError(msg)
        (k,) = touched
        space = spaces[k]
        hits = np.zeros(space.dim, dtype=int)
        new_level = np.zeros(space.dim, dtype=int)
        amplitude = np.zeros(space.dim, dtype=complex)
        strings = np.ones((space.dim, space.dim), dtype=complex)
        for mono, coeff in factor.terms.items():
            level = np.arange(space.dim)
            amp = np.full(space.dim, as_complex(coeff))
            string = np.ones(space.dim, dtype=complex)
            for op in reversed(mono.ops):
                new, step = space.action(op)
                amp = amp * step[level]
                string = string * space.string_factor(op)
                level = new[level]
            hit = amp != 0
            hits += hit
            new_level[hit], amplitude[hit], strings[hit] = level[hit], amp[hit], string
        one_to_one = sorted(new_level.tolist()) == list(range(space.dim))
        if np.any(hits != 1) or not one_to_one or not np.allclose(abs(amplitude), 1):
            msg = "An on-site factor must send each level to one level with a phase."
            raise ValueError(msg)
        tables.append((k, new_level, amplitude, strings))
    if len({k for k, _, _, _ in tables}) < len(tables):
        msg = "An on-site symmetry has more than one factor on a mode."
        raise ValueError(msg)
    return tuple(sorted(tables, key=lambda table: table[0]))


def _with_onsite(
    levels: Levels,
    group: Sequence[IntArray],
    characters: ComplexArray,
    generators: Sequence[IntArray],
    onsite_symmetries: Sequence[OnsiteSymmetry],
    onsite_tables: Sequence[Sequence[OnsiteTable]],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> tuple[tuple[IntArray, ...], ComplexArray, tuple[tuple[int, ...], ...]]:
    """Add the on-site symmetries to the group, checked on ``levels``."""
    actions = [
        functools.partial(_apply_onsite, tables=tables, starts=starts)
        for tables in onsite_tables
    ]
    others: list[Action] = [
        functools.partial(
            _permute, perm=perm, swaps=_swaps(perm, spaces, starts), spaces=spaces
        )
        for perm in generators
    ]
    for action in actions:
        if not all(_commute(action, other, levels) for other in others):
            msg = "Symmetries must commute with each other."
            raise ValueError(msg)
        others.append(action)

    elements: list[tuple[IntArray, complex, tuple[int, ...]]] = [
        (perm, complex(character), ())
        for perm, character in zip(group, characters, strict=True)
    ]
    for action, (_, eigenvalue) in zip(actions, onsite_symmetries, strict=True):
        order, constant = _order_on(action, levels)
        if not np.isclose(complex(eigenvalue) ** order, constant):
            msg = (
                f"Eigenvalue {eigenvalue} is not allowed: the on-site symmetry "
                f"to the power {order} is {constant:.3g}."
            )
            raise ValueError(msg)
        elements = [
            (perm, character * complex(eigenvalue) ** n, (*powers, n))
            for perm, character, powers in elements
            for n in range(order)
        ]
    return (
        tuple(perm for perm, _, _ in elements),
        np.array([character for _, character, _ in elements]),
        tuple(powers for _, _, powers in elements),
    )


def _order_on(action: Action, levels: Levels) -> tuple[int, complex]:
    """Smallest ``n`` with ``action**n`` a constant on ``levels``, and that constant."""
    moved, phase = levels, np.ones(len(levels), dtype=complex)
    for n in range(1, _MAX_ORDER + 1):
        moved, step = action(moved)
        phase = phase * step
        if np.array_equal(moved, levels) and np.allclose(phase, phase[0]):
            return n, complex(phase[0])
    msg = "An on-site symmetry has no finite order on the selected states."
    raise ValueError(msg)


def _commute(a: Action, b: Action, levels: Levels) -> bool:
    """Whether applying ``a`` then ``b`` agrees with ``b`` then ``a``."""
    ab, ab_first = a(levels)
    ab, ab_second = b(ab)
    ba, ba_first = b(levels)
    ba, ba_second = a(ba)
    same_levels = np.array_equal(ab, ba)
    return same_levels and np.allclose(ab_first * ab_second, ba_first * ba_second)


def _apply_onsite(
    levels: Levels, tables: Sequence[OnsiteTable], starts: Sequence[int]
) -> tuple[Levels, ComplexArray]:
    """Apply a product of on-site factors to every row, last mode first."""
    levels = levels.copy()
    phase = np.ones(len(levels), dtype=complex)
    for k, new_level, amplitude, strings in reversed(tables):
        level = levels[:, k].astype(np.int64)
        phase *= amplitude[level]
        if starts[k] < k and np.any(strings != 1):
            earlier = levels[:, starts[k] : k]
            phase *= np.prod(strings[level[:, None], earlier], axis=1)
        levels[:, k] = new_level[level]
    return levels, phase


def _onsite_images(
    levels: Levels,
    onsite_powers: Sequence[tuple[int, ...]],
    onsite_tables: Sequence[Sequence[OnsiteTable]],
    starts: Sequence[int],
) -> dict[tuple[int, ...], tuple[Levels, ComplexArray]]:
    """Apply each combination of on-site powers once."""
    images: dict[tuple[int, ...], tuple[Levels, ComplexArray]] = {}
    for powers in onsite_powers:
        if powers in images:
            continue
        moved, phase = levels, np.ones(len(levels), dtype=complex)
        for tables, power in zip(onsite_tables, powers, strict=True):
            for _ in range(power):
                moved, step = _apply_onsite(moved, tables, starts)
                phase = phase * step
        images[powers] = (moved, phase)
    return images


def _permute(
    levels: Levels, perm: IntArray, swaps: Swaps, spaces: Sequence[LocalSpace]
) -> tuple[Levels, ComplexArray]:
    """Send mode ``k`` of every row to ``perm[k]``."""
    moved = np.empty_like(levels)
    moved[:, perm] = levels
    return moved, _exchange_phase(levels, swaps, spaces)


def _conjugate(op: OpSum, factors: Sequence[OpSum]) -> OpSum:
    """``U op U†`` for ``U`` the product of the unitary ``factors``."""
    images: dict[Op, OpSum] = {}
    result = zero()
    for mono, coeff in op.terms.items():
        term = identity() * coeff
        for single in mono.ops:
            if single not in images:
                image = OpSum({Monomial((single,)): 1})
                for factor in reversed(factors):
                    image = normal_order(factor * image * hc(factor))
                images[single] = image
            term = term * images[single]
        result = result + term
    return result


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
    onsite_powers: Sequence[tuple[int, ...]],
    onsite_tables: Sequence[Sequence[OnsiteTable]],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
    strides: IntArray,
) -> tuple[Levels, FloatArray]:
    """Keep orbit representatives whose symmetric combination is nonzero.

    Raises ValueError unless the states are a union of whole orbits.
    """
    if not len(levels):
        return levels, np.zeros(0)
    selected = levels @ strides

    # a state represents its orbit if no image has a smaller code; drop
    # states as soon as one image beats them
    found = []
    for first in range(0, len(levels), _CHUNK):
        chunk = levels[first : first + _CHUNK].astype(np.int64)
        images = _onsite_images(chunk, onsite_powers, onsite_tables, starts)
        own = selected[first : first + _CHUNK]
        rows = np.arange(len(chunk))
        for perm, powers in zip(group[1:], onsite_powers[1:], strict=True):
            moved = images[powers][0]
            rows = rows[moved[rows] @ strides[perm] >= own[rows]]
        found.append(first + rows)
    reps = levels[np.concatenate(found)]

    own = reps @ strides
    images = _onsite_images(reps, onsite_powers, onsite_tables, starts)
    codes = np.stack(
        [
            images[powers][0] @ strides[perm]
            for perm, powers in zip(group, onsite_powers, strict=True)
        ]
    )
    fixed = codes == own
    # every representative's orbit is selected, and the orbits add up to all
    # selected states, so nothing else is selected
    pos = np.minimum(np.searchsorted(selected, codes), len(selected) - 1)
    orbit_sizes = len(group) / fixed.sum(axis=0)
    if np.any(selected[pos] != codes) or not np.isclose(orbit_sizes.sum(), len(levels)):
        msg = (
            "A symmetry maps some selected states outside those kept by "
            "diagonal_symmetries or predicate_function."
        )
        raise ValueError(msg)

    # norm = |G| sum over operations that fix the state
    norm = np.zeros(len(reps), dtype=complex)
    for g in range(len(group)):
        moved, onsite_phase = images[onsite_powers[g]]
        rows = fixed[g]
        phase = onsite_phase[rows] * _exchange_phase(moved[rows], swaps[g], spaces)
        norm[rows] += np.conj(characters[g]) * phase
    norm = norm.real * len(group)
    nonzero = norm > _TOL
    return reps[nonzero], norm[nonzero]


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
