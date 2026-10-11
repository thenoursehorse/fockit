"""Single-mode Hilbert spaces and how operators act on them.

A basis state is a row of levels, one per mode. Each algebra provides a
``LocalSpace`` describing one mode; the basis is their product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from operator_algebra.algebra import BOSON, FERMION, PAULI, AlgebraRules, Ladder, Pauli
from operator_algebra.operators import Op

type IntArray = NDArray[np.int64]
type ComplexArray = NDArray[np.complex128]


class LocalSpace(Protocol):
    """Hilbert space of a single mode.

    Each operator kind must send a level to at most one level. Build other
    local operators as sums in constructors, as ``sp``/``sm`` do.
    """

    @property
    def rules(self) -> AlgebraRules:
        """Algebra of the operators acting on this space."""
        ...

    @property
    def dim(self) -> int:
        """Number of levels."""
        ...

    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """For each level: the level ``op`` sends it to, and the amplitude.

        An amplitude of 0 means ``op`` annihilates that level.
        """
        ...

    def string_factor(self, op: Op) -> ComplexArray:
        """Factor ``op`` picks up from each earlier mode of this space.

        Indexed by the level of that earlier mode.
        """
        ...


@dataclass(frozen=True)
class FermionSpace:
    """Fermion mode: level 0 empty, 1 occupied."""

    rules: ClassVar[AlgebraRules] = FERMION
    dim: ClassVar[int] = 2

    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Apply c† or c."""
        if op.kind is Ladder.CREATE:
            return np.array([1, 1]), np.array([1, 0], dtype=complex)
        return np.array([0, 0]), np.array([0, 1], dtype=complex)

    def string_factor(self, op: Op) -> ComplexArray:  # noqa: ARG002
        """Jordan-Wigner sign from each occupied earlier mode."""
        return np.array([1, -1], dtype=complex)


@dataclass(frozen=True)
class BosonSpace:
    """Boson mode truncated at ``n_max`` quanta."""

    n_max: int
    rules: ClassVar[AlgebraRules] = BOSON

    @property
    def dim(self) -> int:
        """Number of levels, ``0..n_max``."""
        return self.n_max + 1

    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Apply b† or b; b† on ``n_max`` gives zero."""
        level = np.arange(self.dim)
        if op.kind is Ladder.CREATE:
            amp = np.sqrt(level + 1).astype(complex)
            amp[-1] = 0
            return np.minimum(level + 1, self.n_max), amp
        return np.maximum(level - 1, 0), np.sqrt(level).astype(complex)

    def string_factor(self, op: Op) -> ComplexArray:  # noqa: ARG002
        """Bosons commute across modes."""
        return np.ones(self.dim, dtype=complex)


@dataclass(frozen=True)
class PauliSpace:
    """Spin-1/2 site: level 0 down, 1 up."""

    rules: ClassVar[AlgebraRules] = PAULI
    dim: ClassVar[int] = 2

    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Apply σx, σy or σz."""
        if op.kind is Pauli.X:
            return np.array([1, 0]), np.array([1, 1], dtype=complex)
        if op.kind is Pauli.Y:
            return np.array([1, 0]), np.array([-1j, 1j])
        if op.kind is Pauli.Z:
            return np.array([0, 1]), np.array([-1, 1], dtype=complex)
        msg = f"Unsupported Pauli kind: {op.kind!r}"
        raise ValueError(msg)

    def string_factor(self, op: Op) -> ComplexArray:  # noqa: ARG002
        """Spins on different sites commute."""
        return np.ones(2, dtype=complex)
