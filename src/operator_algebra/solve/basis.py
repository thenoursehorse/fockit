"""Fock-space basis construction with optional symmetry sector restriction."""

from __future__ import annotations

import cmath
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from operator_algebra._types import as_complex
from operator_algebra.algebra import (
    AlgebraRules,
    _mode_sort_key,
)
from operator_algebra.inspection import modes as op_modes
from operator_algebra.operators import OpSum
from operator_algebra.solve.states import State, apply_op, enumerate_states


@dataclass
class Basis:
    """An ordered Fock-space basis.

    Attributes
    ----------
    modes : tuple[Hashable, ...]
        Mode labels in canonical order.
    rules : AlgebraRules
        The algebra (instance) governing this basis.
    states : tuple[State, ...]
        Ordered basis states (ints for fermion/Pauli, tuples for boson).
    lookup : dict[State, int]
        Inverse map ``state -> index``.
    dim : int
        Number of basis states.
    n_max : int | None
        Bosonic occupation cutoff (``None`` for fermion/Pauli).
    mode_index : dict[Hashable, int]
        Map ``mode label -> position`` in canonical order.

    """

    modes: tuple[Hashable, ...]
    rules: AlgebraRules
    states: tuple[State, ...]
    lookup: dict[State, int]
    dim: int
    n_max: int | None
    mode_index: dict[Hashable, int]


def enumerate_basis(
    modes: Sequence[Hashable],
    rules: AlgebraRules | type[AlgebraRules],
    *,
    sectors: Mapping[OpSum, complex] | None = None,
    predicate: Callable[[State], bool] | None = None,
    n_max: int | None = None,
) -> Basis:
    """Construct an ordered basis for the Fock space of the given modes.

    Parameters
    ----------
    modes : Sequence[Hashable]
        Mode labels. Reordered into canonical order internally.
    rules : AlgebraRules or type[AlgebraRules]
        Algebra rules instance (or class — instantiated automatically).
    sectors : Mapping[OpSum, complex], optional
        Quantum-number operators mapped to their target eigenvalues. Each
        operator must be diagonal in the occupation basis.
    predicate : Callable[[State], bool], optional
        Extra filter applied after sector restriction. Receives the raw
        basis state (int for fermion/Pauli, tuple for boson).
    n_max : int | None
        Bosonic occupation cutoff (``None`` for fermion/Pauli).

    """
    rules_inst = rules() if isinstance(rules, type) else rules
    sorted_modes = tuple(sorted(modes, key=_mode_sort_key))
    mode_index = {m: i for i, m in enumerate(sorted_modes)}

    states: list[State] = enumerate_states(rules_inst, len(sorted_modes), n_max)

    if sectors:
        states = [
            s
            for s in states
            if _matches_sectors(s, sectors, rules_inst, mode_index, n_max)
        ]
    if predicate is not None:
        states = [s for s in states if predicate(s)]

    states_t = tuple(states)
    return Basis(
        modes=sorted_modes,
        rules=rules_inst,
        states=states_t,
        lookup={s: i for i, s in enumerate(states_t)},
        dim=len(states_t),
        n_max=n_max,
        mode_index=mode_index,
    )


def _matches_sectors(
    state: State,
    sectors: Mapping[OpSum, complex],
    rules: AlgebraRules,
    mode_index: dict[Hashable, int],
    n_max: int | None,
) -> bool:
    for qn_op, target in sectors.items():
        if type(qn_op.rules) is not type(rules):
            msg = (
                f"Sector operator rules {qn_op.rules!r} do not match "
                f"basis rules {rules!r}."
            )
            raise ValueError(msg)
        for mode in op_modes(qn_op):
            if mode not in mode_index:
                msg = (
                    "Sector operator references mode "
                    f"{mode!r} which is not in the basis."
                )
                raise ValueError(msg)
        value = _diagonal_value(qn_op, state, rules, mode_index, n_max)
        if not cmath.isclose(value, complex(target), abs_tol=1e-10):
            return False
    return True


def _diagonal_value(
    op: OpSum,
    state: State,
    rules: AlgebraRules,
    mode_index: dict[Hashable, int],
    n_max: int | None,
) -> complex:
    total: complex = 0j
    for mono, coeff in op.terms.items():
        amp: complex = as_complex(coeff)
        cur: State = state
        for op_single in reversed(mono.ops):
            res = apply_op(rules, op_single, mode_index[op_single.mode], cur, n_max)
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
