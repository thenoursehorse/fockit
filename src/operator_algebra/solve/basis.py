"""Fock-space basis construction with optional sector restriction."""

from __future__ import annotations

import cmath
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from operator_algebra._types import as_complex
from operator_algebra.algebra import AlgebraRules, _mode_sort_key
from operator_algebra.operators import OpSum
from operator_algebra.solve.spaces import LocalSpace, State, apply_op


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
    states : tuple[State, ...]
        Ordered basis states. Each is a tuple of levels, one per mode, in
        ``modes`` order.
    lookup : dict[State, int]
        Inverse map ``state -> index``.
    dim : int
        Number of basis states.
    mode_index : dict[tuple[AlgebraRules, Hashable], int]
        Map ``(algebra, label) -> position`` in ``modes``.

    """

    modes: tuple[tuple[AlgebraRules, Hashable], ...]
    spaces: tuple[LocalSpace, ...]
    starts: tuple[int, ...]
    states: tuple[State, ...]
    lookup: dict[State, int]
    dim: int
    mode_index: dict[tuple[AlgebraRules, Hashable], int]


def enumerate_basis(
    spaces: Mapping[LocalSpace, Sequence[Hashable]],
    *,
    sectors: Mapping[OpSum, complex] | None = None,
    predicate: Callable[[State], bool] | None = None,
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
        operator must be diagonal in the occupation basis.
    predicate : Callable[[State], bool], optional
        Extra filter applied after sector restriction. Receives the
        basis state.

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

    states: list[State] = list(product(*(range(s.dim) for s in mode_spaces)))

    if sectors:
        states = [
            s
            for s in states
            if _matches_sectors(s, sectors, mode_index, mode_spaces, starts)
        ]
    if predicate is not None:
        states = [s for s in states if predicate(s)]

    states_t = tuple(states)
    return Basis(
        modes=tuple(modes),
        spaces=tuple(mode_spaces),
        starts=tuple(starts),
        states=states_t,
        lookup={s: i for i, s in enumerate(states_t)},
        dim=len(states_t),
        mode_index=mode_index,
    )


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


def _matches_sectors(
    state: State,
    sectors: Mapping[OpSum, complex],
    mode_index: dict[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> bool:
    for qn_op, target in sectors.items():
        check_modes(qn_op, mode_index)
        value = _diagonal_value(qn_op, state, mode_index, spaces, starts)
        if not cmath.isclose(value, complex(target), abs_tol=1e-10):
            return False
    return True


def _diagonal_value(
    op: OpSum,
    state: State,
    mode_index: dict[tuple[AlgebraRules, Hashable], int],
    spaces: Sequence[LocalSpace],
    starts: Sequence[int],
) -> complex:
    total: complex = 0j
    for mono, coeff in op.terms.items():
        amp: complex = as_complex(coeff)
        cur: State = state
        for op_single in reversed(mono.ops):
            k = mode_index[(op_single.rules, op_single.mode)]
            res = apply_op(spaces[k], op_single, k, cur, starts[k])
            if res is None:
                amp = 0j
                break
            a, cur = res
            amp *= a
        if amp == 0:
            continue
        if cur != state:
            msg = (
                "Sector operator is not diagonal in the occupation basis: "
                f"applying it to state {state} produced a different state."
            )
            raise ValueError(msg)
        total += amp
    return total
