"""Symmetry operations on basis states: groups, orbits and phases."""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from fockit._types import as_complex
from fockit.algebra.operators import Monomial, Op, OpSum, as_opsum, identity, zero
from fockit.algebra.ordering import normal_order
from fockit.algebra.rules import AlgebraRules
from fockit.algebra.simplify import drop_zeros
from fockit.algebra.transforms import hc, substitute_modes
from fockit.hilbert.spaces import (
    ComplexArray,
    FloatArray,
    IntArray,
    Levels,
    LocalSpace,
)

if TYPE_CHECKING:
    from fockit.hilbert.basis import Basis

_CHUNK = 2**20
_TOL = 1e-9
_MAX_ORDER = 64

type PermutationSymmetry = tuple[Mapping[Any, Hashable], complex]
type OnsiteSymmetry = tuple[Sequence[OpSum], complex]
type OnsiteTable = tuple[int, IntArray, ComplexArray, ComplexArray]
type Swaps = tuple[IntArray, IntArray]
type Action = Callable[[Levels], tuple[Levels, ComplexArray]]


def representatives(
    basis: Basis, levels: Levels
) -> tuple[IntArray, ComplexArray, IntArray]:
    """Map each row of ``levels`` to its orbit representative.

    Returns the representative's code, the operation that reaches it, and
    the phase that operation picks up: ``g|row> = phase |representative>``.
    """
    images = onsite_images(
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
        phase[rows] = onsite_phase[rows] * exchange_phase(
            moved[rows], basis.swaps[g], basis.spaces
        )
    return codes[element, np.arange(len(levels))], phase, element


def symmetry_phases(
    op: OpSum,
    permutation_symmetries: Sequence[PermutationSymmetry],
    onsite_symmetries: Sequence[OnsiteSymmetry] = (),
) -> list[complex]:
    """Return the phase each symmetry multiplies ``op`` by: ``g op g† = phase op``.

    Permutation symmetries come first, then on-site ones. Raises ValueError
    if a symmetry turns ``op`` into something other than a multiple of
    itself, e.g. translation and the single-site ``c(0)``.
    """
    images = [
        (substitute_modes(op, label_map), f"the relabelling {label_map!r}")
        for label_map, _ in permutation_symmetries
    ]
    images += [
        (_conjugate(op, factors), "an on-site symmetry")
        for factors, _ in onsite_symmetries
    ]
    op = normal_order(op)
    phases = []
    for raw_image, name in images:
        image = normal_order(raw_image)
        phase = 1 + 0j
        if op.terms:
            mono, coeff = next(iter(op.terms.items()))
            phase = as_complex(image.terms.get(mono, 0)) / as_complex(coeff)
        if drop_zeros(image - phase * op, tol=1e-10) != zero():
            msg = (
                f"Operator doesn't just pick up a phase under {name}. Split it "
                "into parts that do, e.g. c_k = Σ_j e^(-ikj) c(j) for translation."
            )
            raise ValueError(msg)
        phases.append(phase)
    return phases


def symmetry_components(
    op: OpSum | complex, basis: Basis
) -> dict[tuple[complex, ...], OpSum]:
    """Split ``op`` into parts that each take ``basis`` to one symmetry sector.

    Each key holds the eigenvalues of the sector that part leads to, in the
    order ``basis`` lists its permutation and then on-site symmetries. Build
    that sector with ``build_basis`` and use ``to_matrix(part, target,
    source=basis)``. The parts add up to ``op``, so under translation
    ``c(0)`` splits into ``c_k / √L``.
    """
    actions: list[Callable[[OpSum], OpSum]] = [
        functools.partial(substitute_modes, mapping=label_map)
        for label_map, _ in basis.permutation_symmetries
    ]
    actions += [
        functools.partial(_conjugate, factors=factors)
        for factors, _ in basis.onsite_symmetries
    ]
    # no single symmetry can need more steps than the whole group has elements
    limit = max(len(basis.group), _MAX_ORDER)
    parts = [normal_order(as_opsum(op))]
    for action in actions:
        parts = [piece for part in parts for piece in _split(part, action, limit)]

    eigenvalues = [
        complex(e) for _, e in (*basis.permutation_symmetries, *basis.onsite_symmetries)
    ]
    out: dict[tuple[complex, ...], OpSum] = {}
    for part in parts:
        try:
            phases = symmetry_phases(
                part, basis.permutation_symmetries, basis.onsite_symmetries
            )
        except ValueError:
            msg = (
                "The symmetries don't commute on this operator, so it has no "
                "parts with one eigenvalue each. Split it with fewer symmetries, "
                "e.g. translation alone."
            )
            raise ValueError(msg) from None
        key = tuple(
            _round(e * phase) for e, phase in zip(eigenvalues, phases, strict=True)
        )
        out[key] = out.get(key, zero()) + part
    return out


def permutation_group(
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


def tabulate_onsite(
    factors: Sequence[OpSum],
    mode_index: Mapping[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
) -> tuple[OnsiteTable, ...]:
    """Per mode: the new level, amplitude and string factors of each level."""
    tables = []
    for factor in factors:
        keys = {(o.rules, o.mode) for mono in factor.terms for o in mono.ops}
        if not keys <= mode_index.keys():
            msg = "An on-site factor acts on a mode that is not in the basis."
            raise ValueError(msg)
        touched = {mode_index[key] for key in keys}
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
        if not np.allclose(strings, strings[0]):
            msg = (
                "An on-site factor's terms must pick up the same string factor "
                "from earlier modes, e.g. not a + a† for anyons."
            )
            raise ValueError(msg)
        tables.append((k, new_level, amplitude, strings))
    if len({k for k, _, _, _ in tables}) < len(tables):
        msg = "An on-site symmetry has more than one factor on a mode."
        raise ValueError(msg)
    return tuple(sorted(tables, key=lambda table: table[0]))


def add_onsite(
    levels: Levels,
    group: Sequence[IntArray],
    characters: ComplexArray,
    generators: Sequence[IntArray],
    onsite_symmetries: Sequence[OnsiteSymmetry],
    onsite_tables: Sequence[Sequence[OnsiteTable]],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> tuple[tuple[IntArray, ...], ComplexArray, tuple[tuple[int, ...], ...]]:
    """Add the on-site symmetries to the group, checking they commute on ``levels``."""
    # a power of U is a constant on every state once it is on these rows, one
    # per level of each mode: it then multiplies a state by one phase per mode
    probes = np.zeros((sum(s.dim for s in spaces), len(spaces)), dtype=np.int64)
    row = 0
    for k, space in enumerate(spaces):
        probes[row : row + space.dim, k] = np.arange(space.dim)
        row += space.dim
    actions = [
        functools.partial(_apply_onsite, tables=tables, starts=starts)
        for tables in onsite_tables
    ]
    others: list[Action] = [
        functools.partial(
            _permute,
            perm=perm,
            swaps=reversed_pairs(perm, spaces, starts),
            spaces=spaces,
        )
        for perm in generators
    ]
    for action in actions:
        for other in others:
            problem = _commutation_problem(action, other, levels)
            if problem:
                raise ValueError(problem)
        others.append(action)

    elements: list[tuple[IntArray, complex, tuple[int, ...]]] = [
        (perm, complex(character), ())
        for perm, character in zip(group, characters, strict=True)
    ]
    for action, (_, eigenvalue) in zip(actions, onsite_symmetries, strict=True):
        order, constant = _order_on(action, probes)
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


def reversed_pairs(
    perm: IntArray, spaces: Sequence[LocalSpace], starts: Sequence[int]
) -> Swaps:
    """Pairs of modes in the same space whose order ``perm`` reverses."""
    a, b = np.triu_indices(len(perm), 1)
    keep = (perm[a] > perm[b]) & (np.asarray(starts)[a] == np.asarray(starts)[b])
    nontrivial = [bool(np.any(spaces[k].exchange_factor() != 1)) for k in a.tolist()]
    keep &= np.array(nontrivial, dtype=bool)
    return a[keep], b[keep]


def exchange_phase(
    levels: Levels, swaps: Swaps, spaces: Sequence[LocalSpace]
) -> ComplexArray:
    """Phase from reordering the modes of each row, one swapped pair at a time."""
    phase = np.ones(len(levels), dtype=complex)
    for a, b in zip(*swaps, strict=True):
        phase *= spaces[a].exchange_factor()[levels[:, a], levels[:, b]]
    return phase


def onsite_images(
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


def orbit_representatives(
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
        images = onsite_images(chunk, onsite_powers, onsite_tables, starts)
        own = selected[first : first + _CHUNK]
        rows = np.arange(len(chunk))
        for perm, powers in zip(group[1:], onsite_powers[1:], strict=True):
            moved = images[powers][0]
            rows = rows[moved[rows] @ strides[perm] >= own[rows]]
        found.append(first + rows)
    reps = levels[np.concatenate(found)]

    own = reps @ strides
    images = onsite_images(reps, onsite_powers, onsite_tables, starts)
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
        phase = onsite_phase[rows] * exchange_phase(moved[rows], swaps[g], spaces)
        norm[rows] += np.conj(characters[g]) * phase
    norm = norm.real * len(group)
    nonzero = norm > _TOL
    return reps[nonzero], norm[nonzero]


def _order_on(action: Action, levels: Levels) -> tuple[int, complex]:
    """Smallest ``n`` with ``action**n`` a constant on ``levels``, and that constant."""
    moved, phase = levels, np.ones(len(levels), dtype=complex)
    for n in range(1, _MAX_ORDER + 1):
        moved, step = action(moved)
        phase = phase * step
        if np.array_equal(moved, levels) and np.allclose(phase, phase[0]):
            return n, complex(phase[0])
    msg = (
        "An on-site symmetry has no finite order. If it is diagonal, use "
        "diagonal_symmetries instead."
    )
    raise ValueError(msg)


def _commutation_problem(a: Action, b: Action, levels: Levels) -> str | None:
    """Explain how applying ``a`` then ``b`` differs from ``b`` then ``a``.

    Returns None if they agree on every row of ``levels``.
    """
    ab, ab_first = a(levels)
    ab, ab_second = b(ab)
    ba, ba_first = b(levels)
    ba, ba_second = a(ba)
    if not np.array_equal(ab, ba):
        return "Symmetries must commute with each other."
    ratio = (ab_first * ab_second) / (ba_first * ba_second)
    if np.allclose(ratio, 1):
        return None
    if np.allclose(ratio, -1):
        return (
            "These symmetries anticommute, so no state has an eigenvalue of both. "
            "Use one of them: the other maps its sectors onto each other, so "
            "their spectra are equal."
        )
    if np.allclose(ratio, ratio[0]):
        return (
            f"Swapping these symmetries multiplies every state by {ratio[0]:.3g}, "
            "so no state has an eigenvalue of both. Use one of them."
        )
    return (
        "These symmetries commute on some states but not others, e.g. only at "
        "some particle numbers. Restrict the states with diagonal_symmetries, "
        "e.g. fix the particle number."
    )


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


def _permute(
    levels: Levels, perm: IntArray, swaps: Swaps, spaces: Sequence[LocalSpace]
) -> tuple[Levels, ComplexArray]:
    """Send mode ``k`` of every row to ``perm[k]``."""
    moved = np.empty_like(levels)
    moved[:, perm] = levels
    return moved, exchange_phase(levels, swaps, spaces)


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
                    # a factor on another mode should cancel, but leaves round-off
                    scale = max(abs(as_complex(v)) for v in image.terms.values())
                    image = drop_zeros(image, tol=1e-12 * scale)
                images[single] = image
            term = term * images[single]
        result = result + term
    return result


def _split(op: OpSum, action: Callable[[OpSum], OpSum], limit: int) -> list[OpSum]:
    """Split ``op`` into parts that ``action`` multiplies by one phase each.

    Applies ``action`` until ``op`` comes back, after ``p`` steps; the part
    with phase ``λ = e^(2πin/p)`` is the average of ``λ^-m action^m(op)``.
    """
    images = [op]
    while True:
        image = normal_order(action(images[-1]))
        if drop_zeros(image - op, tol=1e-10) == zero():
            break
        images.append(image)
        if len(images) > limit:
            msg = (
                f"A symmetry applied {limit} times never gave the operator back. "
                "Its algebra's rules may not reduce products to one normal form."
            )
            raise ValueError(msg)
    period = len(images)
    parts = []
    for n in range(period):
        phase = np.exp(2j * np.pi * n / period)
        part = zero()
        for m, image in enumerate(images):
            part += complex(phase**-m / period) * image
        part = drop_zeros(part, tol=1e-12)
        if part != zero():
            parts.append(part)
    return parts


def _round(value: complex) -> complex:
    """Round to 10 decimals, without negative zeros, so equal keys match."""
    return complex(round(value.real, 10) + 0.0, round(value.imag, 10) + 0.0)
