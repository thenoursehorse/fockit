"""Single-mode Hilbert spaces and how operators act on them.

A basis state is a row of levels, one per mode. Each algebra provides a
``LocalSpace`` describing one mode; the basis is their product.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, override

import numpy as np
from numpy.typing import NDArray

from fockit.algebra.operators import Op
from fockit.algebra.rules import BOSON, FERMION, PAULI, AlgebraRules, Ladder, Pauli
from fockit.algebra.superoperators import ConjugateRules

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]
type Levels = NDArray[np.integer[Any]]
type ComplexArray = NDArray[np.complex128]


class LocalSpace(Protocol):
    """Hilbert space of a single mode.

    Each operator kind must send a level to at most one level. Build other
    local operators as sums in constructors, as ``sp``/``sm`` do. Subclass
    this for your own spaces, so they can be mixed with the built-in ones.
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

    def exchange_factor(self) -> ComplexArray:
        """Phase from swapping two modes of this space, indexed by their levels."""
        ...


@dataclass(frozen=True)
class FermionSpace(LocalSpace):
    """Fermion mode: level 0 empty, 1 occupied."""

    rules: ClassVar[AlgebraRules] = FERMION
    dim: ClassVar[int] = 2

    @override
    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Apply c† or c."""
        if op.kind is Ladder.CREATE:
            return np.array([1, 1]), np.array([1, 0], dtype=complex)
        return np.array([0, 0]), np.array([0, 1], dtype=complex)

    @override
    def string_factor(self, op: Op) -> ComplexArray:
        """Jordan-Wigner sign from each occupied earlier mode."""
        return np.array([1, -1], dtype=complex)

    @override
    def exchange_factor(self) -> ComplexArray:
        """Give -1 for swapping two occupied modes."""
        return np.array([[1, 1], [1, -1]], dtype=complex)


@dataclass(frozen=True)
class BosonSpace(LocalSpace):
    """Boson mode truncated at ``n_max`` quanta."""

    n_max: int
    rules: ClassVar[AlgebraRules] = BOSON

    @property
    @override
    def dim(self) -> int:
        """Number of levels, ``0..n_max``."""
        return self.n_max + 1

    @override
    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Apply b† or b; b† on ``n_max`` gives zero."""
        level = np.arange(self.dim)
        if op.kind is Ladder.CREATE:
            amp = np.sqrt(level + 1).astype(complex)
            amp[-1] = 0
            return np.minimum(level + 1, self.n_max), amp
        return np.maximum(level - 1, 0), np.sqrt(level).astype(complex)

    @override
    def string_factor(self, op: Op) -> ComplexArray:
        """Bosons commute across modes."""
        return np.ones(self.dim, dtype=complex)

    @override
    def exchange_factor(self) -> ComplexArray:
        """Bosons commute across modes."""
        return np.ones((self.dim, self.dim), dtype=complex)


@dataclass(frozen=True)
class PauliSpace(LocalSpace):
    """Spin-1/2 site: level 0 down, 1 up."""

    rules: ClassVar[AlgebraRules] = PAULI
    dim: ClassVar[int] = 2

    @override
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

    @override
    def string_factor(self, op: Op) -> ComplexArray:
        """Spins on different sites commute."""
        return np.ones(2, dtype=complex)

    @override
    def exchange_factor(self) -> ComplexArray:
        """Spins on different sites commute."""
        return np.ones((2, 2), dtype=complex)


@dataclass(frozen=True)
class Conjugate(LocalSpace):
    """Wraps a space for operators that multiply a density matrix from the right.

    The wrapped operators act like those of ``wrapped_space``, with
    complex-conjugated matrices. ``fockit.algebra.right`` makes them.
    """

    wrapped_space: LocalSpace

    @property
    @override
    def rules(self) -> AlgebraRules:
        """The wrapped space's algebra, wrapped in ``ConjugateRules``."""
        return ConjugateRules(self.wrapped_space.rules)

    @property
    @override
    def dim(self) -> int:
        """Same number of levels as the wrapped space."""
        return self.wrapped_space.dim

    @override
    def action(self, op: Op) -> tuple[IntArray, ComplexArray]:
        """Act like the wrapped space, with conjugated amplitudes."""
        new_level, amp = self.wrapped_space.action(self._unwrapped(op))
        return new_level, amp.conj()

    @override
    def string_factor(self, op: Op) -> ComplexArray:
        """Conjugate the wrapped space's string factor."""
        return self.wrapped_space.string_factor(self._unwrapped(op)).conj()

    @override
    def exchange_factor(self) -> ComplexArray:
        """Conjugate the wrapped space's exchange phase."""
        return self.wrapped_space.exchange_factor().conj()

    def _unwrapped(self, op: Op) -> Op:
        return Op(op.mode, op.kind, self.wrapped_space.rules)


def doubled[S: LocalSpace](
    spaces: Mapping[S, Sequence[Hashable]],
) -> dict[LocalSpace, Sequence[Hashable]]:
    """Add a ``Conjugate`` of each space, to build a basis for density matrices.

    For example ``build_basis(doubled({FermionSpace(): sites}))``.
    """
    out: dict[LocalSpace, Sequence[Hashable]] = dict(spaces.items())
    out.update({Conjugate(space): labels for space, labels in spaces.items()})
    return out
