"""Single-mode Hilbert spaces and how operators act on them.

A basis state is a tuple of levels, one per mode. Each algebra provides a
``LocalSpace`` describing one mode; the basis is their product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from operator_algebra.algebra import BOSON, FERMION, PAULI, AlgebraRules, Ladder, Pauli
from operator_algebra.operators import Op

type State = tuple[int, ...]


class LocalSpace(Protocol):
    """Hilbert space of a single mode."""

    @property
    def rules(self) -> AlgebraRules:
        """Algebra of the operators acting on this space."""
        ...

    @property
    def dim(self) -> int:
        """Number of levels."""
        ...

    def apply(self, op: Op, level: int) -> tuple[complex, int] | None:
        """Return ``(amplitude, new_level)``, or None if ``op`` gives zero."""
        ...

    def string_phase(self, op: Op, earlier: State) -> complex:
        """Phase ``op`` picks up from the levels of the modes before it."""
        ...


@dataclass(frozen=True)
class FermionSpace:
    """Fermion mode: level 0 empty, 1 occupied."""

    rules: ClassVar[AlgebraRules] = FERMION
    dim: ClassVar[int] = 2

    def apply(self, op: Op, level: int) -> tuple[complex, int] | None:
        """Apply c† or c."""
        if op.kind is Ladder.CREATE:
            return None if level else (1, 1)
        return (1, 0) if level else None

    def string_phase(self, op: Op, earlier: State) -> complex:  # noqa: ARG002
        """Jordan-Wigner sign."""
        return -1 if sum(earlier) & 1 else 1


@dataclass(frozen=True)
class BosonSpace:
    """Boson mode truncated at ``n_max`` quanta."""

    n_max: int
    rules: ClassVar[AlgebraRules] = BOSON

    @property
    def dim(self) -> int:
        """Number of levels, ``0..n_max``."""
        return self.n_max + 1

    def apply(self, op: Op, level: int) -> tuple[complex, int] | None:
        """Apply b† or b."""
        if op.kind is Ladder.CREATE:
            if level == self.n_max:
                return None
            return (level + 1) ** 0.5, level + 1
        if level == 0:
            return None
        return level**0.5, level - 1

    def string_phase(self, op: Op, earlier: State) -> complex:  # noqa: ARG002
        """Bosons commute across modes."""
        return 1


@dataclass(frozen=True)
class PauliSpace:
    """Spin-1/2 site: level 0 down, 1 up."""

    rules: ClassVar[AlgebraRules] = PAULI
    dim: ClassVar[int] = 2

    def apply(self, op: Op, level: int) -> tuple[complex, int] | None:
        """Apply σx, σy or σz."""
        if op.kind is Pauli.X:
            return 1, 1 - level
        if op.kind is Pauli.Y:
            return (1j if level else -1j), 1 - level
        if op.kind is Pauli.Z:
            return (1 if level else -1), level
        msg = f"Unsupported Pauli kind: {op.kind!r}"
        raise ValueError(msg)

    def string_phase(self, op: Op, earlier: State) -> complex:  # noqa: ARG002
        """Spins on different sites commute."""
        return 1


def apply_op(
    space: LocalSpace, op: Op, index: int, state: State, start: int
) -> tuple[complex, State] | None:
    """Apply ``op`` to the mode at ``index``.

    ``start`` is the first mode of the same space, so the string phase only
    sees modes of that space. Returns ``(amplitude, new_state)``, or None if
    the result is zero.
    """
    res = space.apply(op, state[index])
    if res is None:
        return None
    amp, level = res
    phase = space.string_phase(op, state[start:index])
    return amp * phase, (*state[:index], level, *state[index + 1 :])
