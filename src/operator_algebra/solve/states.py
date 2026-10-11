"""Per-algebra dispatch: local Fock-space enumeration and operator action.

For each algebra (fermion, boson, Pauli) we provide:

- ``enumerate_states`` — list every basis state of the full Fock space.
- ``apply_op`` — apply a single ``Op`` to a basis state, returning
  ``(amplitude, new_state)`` or ``None`` if annihilated.

Basis state encodings
---------------------
- Fermion: ``int`` bitstring; bit ``i`` is mode ``i`` in canonical order.
- Pauli: ``int`` bitstring; bit ``i`` is site ``i``; ``0 = down, 1 = up``.
- Boson: ``tuple[int, ...]`` of occupations, one per mode.
"""

from __future__ import annotations

from itertools import product

from operator_algebra.algebra import (
    AlgebraRules,
    BosonRules,
    FermionRules,
    PauliRules,
)
from operator_algebra.operators import Op

type FermionState = int
type PauliState = int
type BosonState = tuple[int, ...]
type State = int | tuple[int, ...]

_UNKNOWN_RULES = "Unsupported AlgebraRules: {!r}"
_BOSONS_NEED_CUTOFF = "Bosons need a maximum Fock cutoff for each mode"


def local_dim(rules: AlgebraRules, n_max: int | None) -> int:
    """Return the per-mode Hilbert space dimension."""
    if isinstance(rules, FermionRules | PauliRules):
        return 2
    if isinstance(rules, BosonRules):
        if n_max is None:
            raise ValueError(_BOSONS_NEED_CUTOFF)
        return n_max + 1  # +1 for state 0 (vacuum)
    raise TypeError(_UNKNOWN_RULES.format(rules))


def enumerate_states(
    rules: AlgebraRules, num_modes: int, n_max: int | None
) -> list[State]:
    """Enumerate every basis state of the full Fock space."""
    if isinstance(rules, FermionRules | PauliRules):
        return list(range(1 << num_modes))
    if isinstance(rules, BosonRules):
        if n_max is None:
            raise ValueError(_BOSONS_NEED_CUTOFF)
        return list(product(range(n_max + 1), repeat=num_modes))
    raise TypeError(_UNKNOWN_RULES.format(rules))


def apply_op(
    rules: AlgebraRules,
    op: Op,
    mode_index: int,
    state: State,
    n_max: int | None,
) -> tuple[complex, State] | None:
    """Apply a single ``Op`` to ``state``.

    Returns ``(amplitude, new_state)`` or ``None`` if the result is zero.
    """
    if isinstance(rules, FermionRules):
        assert isinstance(state, int)
        return _apply_fermion(op, mode_index, state)
    if isinstance(rules, BosonRules):
        assert isinstance(state, tuple)
        assert n_max is not None
        return _apply_boson(op, mode_index, state, n_max)
    if isinstance(rules, PauliRules):
        assert isinstance(state, int)
        return _apply_pauli(op, mode_index, state)
    raise TypeError(_UNKNOWN_RULES.format(rules))


def _jw_sign(state: int, bit: int) -> int:
    """Jordan-Wigner string."""
    return -1 if (state & (bit - 1)).bit_count() & 1 else 1


def _apply_fermion(op: Op, mode_index: int, state: int) -> tuple[complex, int] | None:
    bit = 1 << mode_index
    occupied = bool(state & bit)
    if op.kind is True:
        if occupied:
            return None
        return complex(_jw_sign(state, bit)), state | bit
    if not occupied:
        return None
    return complex(_jw_sign(state, bit)), state ^ bit


def _apply_boson(
    op: Op, mode_index: int, state: tuple[int, ...], n_max: int
) -> tuple[complex, tuple[int, ...]] | None:
    n = state[mode_index]
    if op.kind is True:
        if n == n_max:
            return None
        new = (*state[:mode_index], n + 1, *state[mode_index + 1 :])
        return complex((n + 1) ** 0.5), new
    if n == 0:
        return None
    new = (*state[:mode_index], n - 1, *state[mode_index + 1 :])
    return complex(n**0.5), new


def _apply_pauli(op: Op, mode_index: int, state: int) -> tuple[complex, int] | None:
    bit = 1 << mode_index
    is_up = bool(state & bit)
    kind = op.kind
    if kind == "x":
        return 1 + 0j, state ^ bit
    if kind == "z":
        return (1 + 0j if is_up else -1 + 0j), state
    if kind == "y":
        return (1j if is_up else -1j), state ^ bit
    msg = f"Unsupported Pauli kind: {kind!r}"
    raise ValueError(msg)
