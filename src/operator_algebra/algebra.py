"""Algebra rules protocol and concrete implementations."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from operator_algebra.types import LadderOp

type Scalar = Any


@dataclass(frozen=True)
class SwapResult:
    """Result of swapping two adjacent operators.

    Each entry in ``terms`` is a ``(Scalar, tuple[LadderOp, ...])`` pair,
    where the tuple replaces the original two-operator subsequence.

    Parameters
    ----------
    terms : tuple[tuple[Scalar, tuple[LadderOp, ...]], ...]
        The replacement terms.

    Notes
    -----
    For the normal ordering algorithm to terminate, implementations must
    ensure that the total number of inversions strictly decreases across
    the replacement terms. Tuples of length 0 or 2 are safe.
    """

    terms: tuple[tuple[Scalar, tuple[LadderOp, ...]], ...]


class AlgebraRules(Protocol):
    """Protocol defining the (anti)commutation relations for an algebra.

    Notes
    -----
    Implementations MUST define ``__eq__`` so that two instances
    representing the same algebra compare equal.
    """

    def swap(self, left: LadderOp, right: LadderOp) -> SwapResult:
        """Define what happens when ``left`` and ``right`` are exchanged.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        SwapResult
        """
        ...

    def is_normal_ordered_pair(self, left: LadderOp, right: LadderOp) -> bool:
        """Return True if ``(left, right)`` is already in normal order.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        bool
        """
        ...

    def is_zero_monomial(self, ops: tuple[LadderOp, ...]) -> bool:
        """Return True if this monomial is identically zero.

        Parameters
        ----------
        ops : tuple[LadderOp, ...]

        Returns
        -------
        bool
        """
        ...


def _mode_sort_key(mode: Hashable) -> tuple[Any, ...]:
    """Canonical sort key for modes.

    Stratifies by type first, then compares within type.

    Parameters
    ----------
    mode : Hashable
        The mode label.

    Returns
    -------
    tuple
        A universally comparable sort key.

    Notes
    -----
    ``bool`` is a subclass of ``int``, so ``True`` and ``1`` produce
    identical sort keys. Avoid using ``bool`` values as modes.
    """
    match mode:
        case int():
            return (0, mode)
        case str():
            return (1, mode)
        case tuple():
            return (2, *(_mode_sort_key(m) for m in mode))
        case _:
            return (3, repr(mode))


class FermionRules:
    """Canonical anticommutation relations: {c_a, c†_b} = delta_{ab}."""

    def swap(self, left: LadderOp, right: LadderOp) -> SwapResult:
        """Fermion swap rules.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        SwapResult
        """
        swapped = ((-1, (right, left)),)
        if not left.is_creator and right.is_creator and left.mode == right.mode:
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: LadderOp, right: LadderOp) -> bool:
        """Normal order: creators left, canonical mode order within blocks.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        bool
        """
        if left.is_creator and not right.is_creator:
            return True
        if not left.is_creator and right.is_creator:
            return False
        return _mode_sort_key(left.mode) <= _mode_sort_key(right.mode)

    def is_zero_monomial(self, ops: tuple[LadderOp, ...]) -> bool:
        """Zero if any adjacent pair of identical LadderOps exists.

        Only adjacent duplicates are checked because non-adjacent duplicates
        may still yield nonzero contributions after expanding commutation
        relations (e.g., c†_1 c_1 c†_1 = c†_1).

        Parameters
        ----------
        ops : tuple[LadderOp, ...]

        Returns
        -------
        bool
        """
        for i in range(len(ops) - 1):
            if ops[i] == ops[i + 1]:
                return True
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FermionRules)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return "FermionRules()"


class BosonRules:
    """Canonical commutation relations: [b_a, b†_b] = delta_{ab}."""

    def swap(self, left: LadderOp, right: LadderOp) -> SwapResult:
        """Boson swap rules.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        SwapResult
        """
        swapped = ((+1, (right, left)),)
        if not left.is_creator and right.is_creator and left.mode == right.mode:
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: LadderOp, right: LadderOp) -> bool:
        """Check normal ordering for bosonic operators.

        Parameters
        ----------
        left : LadderOp
        right : LadderOp

        Returns
        -------
        bool
        """
        if left.is_creator and not right.is_creator:
            return True
        if not left.is_creator and right.is_creator:
            return False
        return _mode_sort_key(left.mode) <= _mode_sort_key(right.mode)

    def is_zero_monomial(self, ops: tuple[LadderOp, ...]) -> bool:
        """Bosons never square to zero.

        Parameters
        ----------
        ops : tuple[LadderOp, ...]

        Returns
        -------
        bool
        """
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BosonRules)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return "BosonRules()"


FERMION = FermionRules()
BOSON = BosonRules()
