"""Fock-space basis construction with optional sector restriction."""

from __future__ import annotations

import cmath
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from operator_algebra._types import as_complex
from operator_algebra.algebra import _mode_sort_key
from operator_algebra.inspection import modes as op_modes
from operator_algebra.operators import OpSum
from operator_algebra.solve.spaces import LocalSpace, State, apply_op


@dataclass
class Basis:
    """An ordered Fock-space basis.

    Attributes
    ----------
    modes : tuple[Hashable, ...]
        Mode labels in canonical order.
    space : LocalSpace
        Hilbert space of each mode.
    states : tuple[State, ...]
        Ordered basis states. Each is a tuple of levels, one per mode, in
        ``modes`` order.
    lookup : dict[State, int]
        Inverse map ``state -> index``.
    dim : int
        Number of basis states.
    mode_index : dict[Hashable, int]
        Map ``mode label -> position`` in canonical order.

    """

    modes: tuple[Hashable, ...]
    space: LocalSpace
    states: tuple[State, ...]
    lookup: dict[State, int]
    dim: int
    mode_index: dict[Hashable, int]


def enumerate_basis(
    modes: Sequence[Hashable],
    space: LocalSpace,
    *,
    sectors: Mapping[OpSum, complex] | None = None,
    predicate: Callable[[State], bool] | None = None,
) -> Basis:
    """Construct an ordered basis for the Fock space of the given modes.

    Parameters
    ----------
    modes : Sequence[Hashable]
        Mode labels. Reordered into canonical order internally.
    space : LocalSpace
        Hilbert space of each mode, e.g. ``BosonSpace(n_max=5)``.
    sectors : Mapping[OpSum, complex], optional
        Quantum-number operators mapped to their target eigenvalues. Each
        operator must be diagonal in the occupation basis.
    predicate : Callable[[State], bool], optional
        Extra filter applied after sector restriction. Receives the
        basis state.

    """
    sorted_modes = tuple(sorted(modes, key=_mode_sort_key))
    mode_index = {m: i for i, m in enumerate(sorted_modes)}

    states: list[State] = list(product(range(space.dim), repeat=len(sorted_modes)))

    if sectors:
        states = [s for s in states if _matches_sectors(s, sectors, space, mode_index)]
    if predicate is not None:
        states = [s for s in states if predicate(s)]

    states_t = tuple(states)
    return Basis(
        modes=sorted_modes,
        space=space,
        states=states_t,
        lookup={s: i for i, s in enumerate(states_t)},
        dim=len(states_t),
        mode_index=mode_index,
    )


def _matches_sectors(
    state: State,
    sectors: Mapping[OpSum, complex],
    space: LocalSpace,
    mode_index: dict[Hashable, int],
) -> bool:
    for qn_op, target in sectors.items():
        if type(qn_op.rules) is not type(space.rules):
            msg = (
                f"Sector operator rules {qn_op.rules!r} do not match "
                f"basis rules {space.rules!r}."
            )
            raise ValueError(msg)
        for mode in op_modes(qn_op):
            if mode not in mode_index:
                msg = (
                    "Sector operator references mode "
                    f"{mode!r} which is not in the basis."
                )
                raise ValueError(msg)
        value = _diagonal_value(qn_op, state, space, mode_index)
        if not cmath.isclose(value, complex(target), abs_tol=1e-10):
            return False
    return True


def _diagonal_value(
    op: OpSum,
    state: State,
    space: LocalSpace,
    mode_index: dict[Hashable, int],
) -> complex:
    total: complex = 0j
    for mono, coeff in op.terms.items():
        amp: complex = as_complex(coeff)
        cur: State = state
        for op_single in reversed(mono.ops):
            res = apply_op(space, op_single, mode_index[op_single.mode], cur)
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
