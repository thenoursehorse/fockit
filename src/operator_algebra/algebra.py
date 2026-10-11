"""Algebra rules protocol and concrete implementations."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from operator_algebra._types import Coefficient

if TYPE_CHECKING:
    from operator_algebra.operators import Op


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

    def format_op_text(self, op: Op) -> str:
        """Format a single operator for human-readable text display.

        Parameters
        ----------
        op : Op

        Returns
        -------
        str

        """
        ...

    def format_op_latex(self, op: Op) -> str:
        r"""Format a single operator for LaTeX display.

        Parameters
        ----------
        op : Op

        Returns
        -------
        str

        """
        ...

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

    def format_op_text(self, op: Op) -> str:
        """Format as ``c†(k)`` or ``c(k)``."""
        symbol = "c†" if op.kind is Ladder.CREATE else "c"
        return f"{symbol}({_format_mode(op.mode)})"

    def format_op_latex(self, op: Op) -> str:
        r"""Format as ``c^{\dagger}_{k}`` or ``c_{k}``."""
        symbol = r"c^{\dagger}" if op.kind is Ladder.CREATE else "c"
        return f"{symbol}_{{{_format_mode_latex(op.mode)}}}"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, FermionRules)

    def __hash__(self) -> int:  # noqa: D105
        return hash(type(self))

    def __repr__(self) -> str:  # noqa: D105
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

    def format_op_text(self, op: Op) -> str:
        """Format as ``b†(k)`` or ``b(k)``."""
        symbol = "b†" if op.kind is Ladder.CREATE else "b"
        return f"{symbol}({_format_mode(op.mode)})"

    def format_op_latex(self, op: Op) -> str:
        r"""Format as ``b^{\dagger}_{k}`` or ``b_{k}``."""
        symbol = r"b^{\dagger}" if op.kind is Ladder.CREATE else "b"
        return f"{symbol}_{{{_format_mode_latex(op.mode)}}}"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, BosonRules)

    def __hash__(self) -> int:  # noqa: D105
        return hash(type(self))

    def __repr__(self) -> str:  # noqa: D105
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

    def format_op_text(self, op: Op) -> str:
        """Format as ``σ_x(k)``."""
        return f"σ_{op.kind}({_format_mode(op.mode)})"

    def format_op_latex(self, op: Op) -> str:
        r"""Format as ``\sigma^{x}_{k}``."""
        return rf"\sigma^{{{op.kind}}}_{{{_format_mode_latex(op.mode)}}}"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, PauliRules)

    def __hash__(self) -> int:  # noqa: D105
        return hash(type(self))

    def __repr__(self) -> str:  # noqa: D105
        return "PauliRules()"


FERMION = FermionRules()
BOSON = BosonRules()
PAULI = PauliRules()


def mode_sort_key(mode: Hashable) -> tuple[object, ...]:
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
            return (2, *(mode_sort_key(m) for m in mode))
        case _:
            return (3, repr(mode))


def _format_mode(mode: Hashable) -> str:
    """Format a mode label for display."""
    if isinstance(mode, tuple):
        return ", ".join(str(m) for m in mode)
    return str(mode)


def _format_mode_latex(mode: Hashable) -> str:
    """Format a mode label for LaTeX subscript."""
    raw = _format_mode(mode)
    return raw.replace("up", r"\uparrow").replace("dn", r"\downarrow")
