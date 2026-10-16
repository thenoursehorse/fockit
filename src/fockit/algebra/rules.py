"""Algebra rules protocol and concrete implementations."""

from __future__ import annotations

import functools
import operator
from collections.abc import Hashable
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum, StrEnum
from typing import (
    TYPE_CHECKING,
    Protocol,
    SupportsIndex,
    cast,
    override,
    runtime_checkable,
)

import numpy as np

from fockit._types import Coefficient

if TYPE_CHECKING:
    from fockit.algebra.operators import Op

_MODE_MSG = (
    "Mode labels must be int, float, str, bool, None, an Enum, a dataclass "
    "with ==, or a tuple of these, got {!r} of type {}"
)


class Ladder(StrEnum):
    """Operator kinds for fermions and bosons."""

    CREATE = "+"
    ANNIHILATE = "-"


class Pauli(StrEnum):
    """Operator kinds for Pauli matrices."""

    X = "x"
    Y = "y"
    Z = "z"


@dataclass(frozen=True)
class SwapResult:
    """Result of swapping two adjacent operators.

    Each entry in ``terms`` is a ``(Coefficient, tuple[Op, ...])`` pair,
    where the tuple replaces the original two-operator subsequence.

    Parameters
    ----------
    terms : tuple[tuple[Coefficient, tuple[Op, ...]], ...]
        The replacement terms.

    Notes
    -----
    For the normal ordering algorithm to terminate, each replacement must
    either shorten the pair (length 0 or 1) or reduce the number of
    inversions (length 2).

    """

    terms: tuple[tuple[Coefficient, tuple[Op, ...]], ...]


@runtime_checkable
class AlgebraRules(Protocol):
    """Protocol defining the (anti)commutation relations for an algebra.

    Methods only ever receive operators of this algebra. Operators of
    different algebras commute.

    Notes
    -----
    Implementations MUST define ``__eq__`` and ``__hash__`` so that two
    instances representing the same algebra compare equal and are usable
    as ``lru_cache`` keys.

    """

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Define what happens when ``left`` and ``right`` are exchanged.

        Parameters
        ----------
        left : Op
        right : Op

        Returns
        -------
        SwapResult

        """
        ...

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Return True if ``(left, right)`` is already in normal order.

        Parameters
        ----------
        left : Op
        right : Op

        Returns
        -------
        bool

        """
        ...

    def dagger(self, op: Op) -> Op:
        """Return the Hermitian conjugate of a single operator.

        Parameters
        ----------
        op : Op

        Returns
        -------
        Op

        """
        ...

    @override
    def __hash__(self) -> int:
        """Return hash for use as ``lru_cache`` key."""
        ...


class FermionRules:
    """Canonical anticommutation relations: {c_a, c†_b} = delta_{ab}."""

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Apply fermion swap rules."""
        if left == right:
            return SwapResult(terms=())  # c c = 0
        swapped = ((-1, (right, left)),)
        if (
            left.kind is Ladder.ANNIHILATE
            and right.kind is Ladder.CREATE
            and left.mode == right.mode
        ):
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Check if a pair is in fermionic normal order.

        Creators left, canonical mode order within blocks.
        """
        if left == right:
            return False
        if left.kind is Ladder.CREATE and right.kind is Ladder.ANNIHILATE:
            return True
        if left.kind is Ladder.ANNIHILATE and right.kind is Ladder.CREATE:
            return False
        return mode_sort_key(left.mode) <= mode_sort_key(right.mode)

    def dagger(self, op: Op) -> Op:
        """Flip creation ↔ annihilation."""
        flipped = Ladder.CREATE if op.kind is Ladder.ANNIHILATE else Ladder.ANNIHILATE
        return replace(op, kind=flipped)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, FermionRules)

    @override
    def __hash__(self) -> int:
        return hash(type(self))

    @override
    def __repr__(self) -> str:
        return "FermionRules()"


class BosonRules:
    """Canonical commutation relations: [b_a, b†_b] = delta_{ab}."""

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Apply boson swap rules."""
        swapped = ((+1, (right, left)),)
        if (
            left.kind is Ladder.ANNIHILATE
            and right.kind is Ladder.CREATE
            and left.mode == right.mode
        ):
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Check normal ordering for bosonic operators."""
        if left.kind is Ladder.CREATE and right.kind is Ladder.ANNIHILATE:
            return True
        if left.kind is Ladder.ANNIHILATE and right.kind is Ladder.CREATE:
            return False
        return mode_sort_key(left.mode) <= mode_sort_key(right.mode)

    def dagger(self, op: Op) -> Op:
        """Flip creation ↔ annihilation."""
        flipped = Ladder.CREATE if op.kind is Ladder.ANNIHILATE else Ladder.ANNIHILATE
        return replace(op, kind=flipped)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BosonRules)

    @override
    def __hash__(self) -> int:
        return hash(type(self))

    @override
    def __repr__(self) -> str:
        return "BosonRules()"


_PAULI_PRODUCT: dict[tuple[Hashable, Hashable], tuple[complex, Hashable]] = {
    (Pauli.X, Pauli.Y): (1j, Pauli.Z),
    (Pauli.Y, Pauli.X): (-1j, Pauli.Z),
    (Pauli.Y, Pauli.Z): (1j, Pauli.X),
    (Pauli.Z, Pauli.Y): (-1j, Pauli.X),
    (Pauli.Z, Pauli.X): (1j, Pauli.Y),
    (Pauli.X, Pauli.Z): (-1j, Pauli.Y),
}


class PauliRules:
    """Pauli algebra: σ_a σ_b = δ_ab + i ε_abc σ_c on a site, commute across sites."""

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Commute across sites, multiply out on the same site."""
        if left.mode != right.mode:
            return SwapResult(terms=((+1, (right, left)),))
        if left.kind == right.kind:
            return SwapResult(terms=((1, ()),))
        phase, kind = _PAULI_PRODUCT[(left.kind, right.kind)]
        return SwapResult(terms=((phase, (replace(left, kind=kind),)),))

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Sites in mode order, at most one operator per site."""
        if left.mode == right.mode:
            return False
        return mode_sort_key(left.mode) <= mode_sort_key(right.mode)

    def dagger(self, op: Op) -> Op:
        """Pauli operators are Hermitian: dagger is identity."""
        return op

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PauliRules)

    @override
    def __hash__(self) -> int:
        return hash(type(self))

    @override
    def __repr__(self) -> str:
        return "PauliRules()"


FERMION = FermionRules()
BOSON = BosonRules()
PAULI = PauliRules()


def mode_sort_key(mode: Hashable) -> tuple[object, ...]:
    """Canonical sort key for modes.

    Stratifies by type first, then compares within type. Equal modes get
    equal keys, unequal modes different keys, and the order is the same on
    every run.

    Parameters
    ----------
    mode : Hashable
        The mode label: int, float, str, bool, None, an Enum, a dataclass
        with ``==``, or a tuple of these.

    Returns
    -------
    tuple
        A universally comparable sort key.

    Raises
    ------
    TypeError
        If ``mode`` is not one of the types above.

    Notes
    -----
    ``bool`` is a subclass of ``int``, so ``True`` and ``1`` produce
    identical sort keys. Avoid using ``bool`` values as modes.

    """
    match mode:
        case int() | float():
            return (0, mode)
        case str():
            return (1, mode)
        case tuple():
            return (2, *(mode_sort_key(m) for m in mode))
        case Enum():
            return (3, type(mode).__module__, type(mode).__qualname__, mode.name)
        case None:
            return (4,)
        case _ if is_dataclass(type(mode)):
            cls = type(mode)
            values = (getattr(mode, name) for name in _compared_fields(cls))
            return (
                5,
                cls.__module__,
                cls.__qualname__,
                *(mode_sort_key(canonical_mode(v)) for v in values),
            )
        case _:
            raise TypeError(_MODE_MSG.format(mode, type(mode).__name__))


def canonical_mode(mode: Hashable) -> Hashable:
    """Convert numpy scalars and other integer types to Python types.

    Parameters
    ----------
    mode : Hashable
        The mode label: int, float, str, bool, None, an Enum, a dataclass
        with ``==``, or a tuple of these. Numpy scalars and other types
        with ``__index__`` are converted.

    Returns
    -------
    Hashable

    Raises
    ------
    TypeError
        If ``mode`` is not one of the types above.

    """
    match mode:
        case int() | float() | str() | None | Enum():
            return mode
        case tuple():
            return tuple(canonical_mode(m) for m in mode)
        case _ if is_dataclass(type(mode)) and type(mode).__eq__ is not object.__eq__:
            for name in _compared_fields(type(mode)):
                canonical_mode(getattr(mode, name))
            return mode
        case np.generic():
            return canonical_mode(cast(Hashable, mode.item()))
        case SupportsIndex():
            return operator.index(mode)
        case _:
            raise TypeError(_MODE_MSG.format(mode, type(mode).__name__))


@functools.cache
def _compared_fields(cls: type) -> tuple[str, ...]:
    """Return the names of the fields ``==`` compares, in order."""
    return tuple(f.name for f in fields(cls) if f.compare)
