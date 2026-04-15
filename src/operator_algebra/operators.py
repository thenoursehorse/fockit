"""Core data types and helpers: Op, Monomial, OpSum."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from operator_algebra.algebra import FERMION, AlgebraRules

type Mode = Hashable
type Scalar = Any

_POW_MSG = "Exponent must be a non-negative integer, got {}"


@dataclass(frozen=True, slots=True)
class Op:
    """A single operator acting on one mode.

    Parameters
    ----------
    mode : Hashable
        Any hashable Python object identifying the single-particle state.
    kind : Hashable
        Operator variant within the mode. ``True``/``False`` for ladder
        (creation/annihilation), ``"x"``/``"y"``/``"z"`` for Pauli, etc.

    """

    mode: Hashable
    kind: Hashable

    def __repr__(self) -> str:
        """Return a generic representation like ``Op(k, True)``."""
        return f"Op({format_mode(self.mode)}, {self.kind!r})"


@dataclass(frozen=True, slots=True)
class Monomial:
    """An ordered product of operators. No coefficient.

    Parameters
    ----------
    ops : tuple[Op, ...]
        The ordered sequence of operators in this product.

    """

    ops: tuple[Op, ...]

    def __len__(self) -> int:
        """Return the number of ladder operators in the product."""
        return len(self.ops)

    def __hash__(self) -> int:
        """Hash by the operator tuple.

        Monomials with the same operators in the same order are identical.
        """
        return hash(self.ops)

    def __eq__(self, other: object) -> bool:
        """Compare by operator sequence.

        Two monomials are equal iff they have the same operators in order.
        """
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.ops == other.ops

    def __repr__(self) -> str:
        """Show the operator product, or ``I`` for the empty monomial (identity)."""
        if not self.ops:
            return "I"
        return " ".join(repr(op) for op in self.ops)


class OpSum:
    """A linear combination of monomials: sum_i coeff_i * monomial_i.

    Parameters
    ----------
    terms : dict[Monomial, Scalar] | None
        Mapping from monomials to coefficients.
    rules : AlgebraRules | None
        Algebra rules. Default: ``FERMION``.

    """

    def __init__(
        self,
        terms: dict[Monomial, Scalar] | None = None,
        rules: AlgebraRules | None = None,
    ) -> None:
        self.rules: AlgebraRules = rules if rules is not None else FERMION
        if terms is None:
            self._terms: dict[Monomial, Scalar] = {}
        else:
            self._terms = {m: c for m, c in terms.items() if not is_zero(c)}

    def __hash__(self) -> int:
        """Hash by the frozen set of (monomial, coefficient) pairs."""
        return hash(frozenset(self._terms.items()))

    @property
    def terms(self) -> MappingProxyType[Monomial, Scalar]:
        """Read-only view of the internal terms dict."""
        return MappingProxyType(self._terms)

    def __add__(self, other: object) -> OpSum:
        """Merge terms from both sums, or promote a scalar via identity."""
        if isinstance(other, OpSum):
            if not rules_compatible(self.rules, other.rules):
                msg = f"Cannot add OpSum with {self.rules} and {other.rules}"
                raise ValueError(msg)
            merged: dict[Monomial, Scalar] = dict(self._terms)
            for mono, coeff in other._terms.items():
                if mono in merged:
                    merged[mono] = merged[mono] + coeff
                else:
                    merged[mono] = coeff
            return OpSum(merged, rules=self.rules)
        if is_scalar(other):
            return self + identity(self.rules) * other
        return NotImplemented

    def __radd__(self, other: object) -> OpSum:
        """Support ``scalar + OpSum`` by delegating to ``__add__``."""
        if is_scalar(other):
            return self + other
        return NotImplemented

    def __sub__(self, other: object) -> OpSum:
        """Subtract by negating and adding."""
        if isinstance(other, OpSum):
            return self + (-other)
        if is_scalar(other):
            return self - identity(self.rules) * other
        return NotImplemented

    def __rsub__(self, other: object) -> OpSum:
        """Support ``scalar - OpSum``."""
        if is_scalar(other):
            return (-self) + other
        return NotImplemented

    def __neg__(self) -> OpSum:
        """Negate all coefficients."""
        return OpSum({m: -c for m, c in self._terms.items()}, rules=self.rules)

    def __mul__(self, other: object) -> OpSum:
        """Distribute over terms; monomial products concatenate ops."""
        if isinstance(other, OpSum):
            if not rules_compatible(self.rules, other.rules):
                msg = f"Cannot multiply OpSum with {self.rules} and {other.rules}"
                raise ValueError(msg)
            result: dict[Monomial, Scalar] = {}
            for m1, c1 in self._terms.items():
                for m2, c2 in other._terms.items():
                    new_mono = Monomial(m1.ops + m2.ops)
                    new_coeff = c1 * c2
                    if new_mono in result:
                        result[new_mono] = result[new_mono] + new_coeff
                    else:
                        result[new_mono] = new_coeff
            return OpSum(result, rules=self.rules)
        if is_scalar(other):
            return OpSum(
                {m: c * other for m, c in self._terms.items()}, rules=self.rules
            )
        return NotImplemented

    def __rmul__(self, other: object) -> OpSum:
        """Scale each coefficient from the left for ``scalar * OpSum``."""
        if is_scalar(other):
            return OpSum(
                {m: other * c for m, c in self._terms.items()}, rules=self.rules
            )
        return NotImplemented

    def __pow__(self, n: int) -> OpSum:
        """Raise to a non-negative integer power.

        Parameters
        ----------
        n : int
            Non-negative integer exponent.

        Returns
        -------
        OpSum

        Notes
        -----
        ``zero() ** 0`` returns ``identity()``. The empty product is the
        identity operator.

        """
        if not isinstance(n, int):
            raise ValueError(_POW_MSG.format(n))
        if n < 0:
            raise ValueError(_POW_MSG.format(n))
        if n == 0:
            return identity(self.rules)
        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        """Two OpSums are equal iff they have the same monomial-coefficient pairs."""
        if not isinstance(other, OpSum):
            return NotImplemented
        return self._terms == other._terms

    def __bool__(self) -> bool:
        """Return False when no terms remain (zero operator)."""
        return bool(self._terms)

    def __repr__(self) -> str:
        """Delegate to ``format_opsum`` for human-readable display."""
        from operator_algebra.display import to_text

        return to_text(self)

    def __str__(self) -> str:
        """Delegate to ``__repr__``."""
        return repr(self)


# Public helpers


def format_mode(mode: Hashable) -> str:
    """Format a mode label for display."""
    if isinstance(mode, tuple):
        return ", ".join(str(m) for m in mode)
    return str(mode)


def is_zero(coeff: Scalar, tol: float = 0.0) -> bool:
    """Check if a coefficient is zero, with tolerance for floats.

    Parameters
    ----------
    coeff : Scalar
        The coefficient to check.
    tol : float
        Tolerance for floating-point comparison. Default 0.0 (exact).

    Returns
    -------
    bool

    """
    if coeff == 0:
        return True
    if isinstance(coeff, float | complex):
        return abs(coeff) <= tol
    return False


def is_scalar(obj: object) -> bool:
    """Check if an object can be used as a scalar coefficient.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool

    """
    if isinstance(obj, OpSum):
        return False
    if isinstance(obj, int | float | complex):
        return True
    if isinstance(obj, str | bytes | list | tuple | set | frozenset | dict):
        return False
    return hasattr(obj, "__mul__") and hasattr(obj, "__add__")


def rules_compatible(a: AlgebraRules, b: AlgebraRules) -> bool:
    """Check if two AlgebraRules instances are compatible.

    Parameters
    ----------
    a : AlgebraRules
    b : AlgebraRules

    Returns
    -------
    bool

    """
    if a == b:
        return True
    return type(a) is type(b)


def identity(rules: AlgebraRules = FERMION) -> OpSum:
    """Return the identity operator (empty product with coefficient 1).

    Parameters
    ----------
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    return OpSum({Monomial(()): 1}, rules=rules)


def zero(rules: AlgebraRules = FERMION) -> OpSum:
    """Return the zero operator (empty sum).

    Parameters
    ----------
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    return OpSum({}, rules=rules)
