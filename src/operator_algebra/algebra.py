"""Algebra rules protocol and concrete implementations."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from operator_algebra._types import Coefficient

if TYPE_CHECKING:
    from operator_algebra.operators import Op


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
    For the normal ordering algorithm to terminate, implementations must
    ensure that the total number of inversions strictly decreases across
    the replacement terms. Tuples of length 0 or 2 are safe.

    """

    terms: tuple[tuple[Coefficient, tuple[Op, ...]], ...]


@runtime_checkable
class AlgebraRules(Protocol):
    """Protocol defining the (anti)commutation relations for an algebra.

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

    def is_zero_monomial(self, ops: tuple[Op, ...]) -> bool:
        """Return True if this monomial is identically zero.

        Parameters
        ----------
        ops : tuple[Op, ...]

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


def _mode_sort_key(mode: Hashable) -> tuple[object, ...]:
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


def _format_mode(mode: Hashable) -> str:
    """Format a mode label for display."""
    if isinstance(mode, tuple):
        return ", ".join(str(m) for m in mode)
    return str(mode)


def _format_mode_latex(mode: Hashable) -> str:
    """Format a mode label for LaTeX subscript."""
    raw = _format_mode(mode)
    return raw.replace("up", r"\uparrow").replace("dn", r"\downarrow")


class FermionRules:
    """Canonical anticommutation relations: {c_a, c†_b} = delta_{ab}."""

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Apply fermion swap rules."""
        swapped = ((-1, (right, left)),)
        if left.kind is False and right.kind is True and left.mode == right.mode:
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Check if a pair is in fermionic normal order.

        Creators left, canonical mode order within blocks.
        """
        if left.kind is True and right.kind is False:
            return True
        if left.kind is False and right.kind is True:
            return False
        return _mode_sort_key(left.mode) <= _mode_sort_key(right.mode)

    def is_zero_monomial(self, ops: tuple[Op, ...]) -> bool:
        """Return True if any adjacent identical ops exist."""
        return any(ops[i] == ops[i + 1] for i in range(len(ops) - 1))

    def dagger(self, op: Op) -> Op:
        """Flip creation ↔ annihilation."""
        from operator_algebra.operators import Op as _Op

        return _Op(op.mode, op.kind is False)

    def format_op_text(self, op: Op) -> str:
        """Format as ``c†(k)`` or ``c(k)``."""
        symbol = "c†" if op.kind is True else "c"
        return f"{symbol}({_format_mode(op.mode)})"

    def format_op_latex(self, op: Op) -> str:
        r"""Format as ``c^{\dagger}_{k}`` or ``c_{k}``."""
        symbol = r"c^{\dagger}" if op.kind is True else "c"
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
        if left.kind is False and right.kind is True and left.mode == right.mode:
            return SwapResult(terms=((1, ()), *swapped))
        return SwapResult(terms=swapped)

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Check normal ordering for bosonic operators."""
        if left.kind is True and right.kind is False:
            return True
        if left.kind is False and right.kind is True:
            return False
        return _mode_sort_key(left.mode) <= _mode_sort_key(right.mode)

    def is_zero_monomial(self, ops: tuple[Op, ...]) -> bool:  # noqa: ARG002
        """Return False; bosons never square to zero."""
        return False

    def dagger(self, op: Op) -> Op:
        """Flip creation ↔ annihilation."""
        from operator_algebra.operators import Op as _Op

        return _Op(op.mode, op.kind is False)

    def format_op_text(self, op: Op) -> str:
        """Format as ``b†(k)`` or ``b(k)``."""
        symbol = "b†" if op.kind is True else "b"
        return f"{symbol}({_format_mode(op.mode)})"

    def format_op_latex(self, op: Op) -> str:
        r"""Format as ``b^{\dagger}_{k}`` or ``b_{k}``."""
        symbol = r"b^{\dagger}" if op.kind is True else "b"
        return f"{symbol}_{{{_format_mode_latex(op.mode)}}}"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, BosonRules)

    def __hash__(self) -> int:  # noqa: D105
        return hash(type(self))

    def __repr__(self) -> str:  # noqa: D105
        return "BosonRules()"


_PAULI_KIND_ORDER: dict[Hashable, int] = {"x": 0, "y": 1, "z": 2}


class PauliRules:
    """Pauli algebra.

    Anticommutes on the same site, commutes across sites.
    """

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Apply Pauli swap rules.

        Same site, same kind: σ_i² = I → identity.
        Same site, different kind: {σ_i, σ_j} = 0 → anticommute.
        Different site: commute.
        """
        if left.mode != right.mode:
            return SwapResult(terms=((+1, (right, left)),))
        if left.kind == right.kind:
            return SwapResult(terms=((1, ()),))
        return SwapResult(terms=((-1, (right, left)),))

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Check Pauli normal order.

        Uses mode order, then ``x < y < z`` within a site.
        """
        if left.mode != right.mode:
            return _mode_sort_key(left.mode) <= _mode_sort_key(right.mode)
        if left.kind == right.kind:
            return False  # σ_i σ_i must reduce to identity
        return _PAULI_KIND_ORDER[left.kind] < _PAULI_KIND_ORDER[right.kind]

    def is_zero_monomial(self, ops: tuple[Op, ...]) -> bool:  # noqa: ARG002
        """Return False; no Pauli monomial is identically zero."""
        return False

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
