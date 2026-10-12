"""Core data types and helpers: Op, Monomial, OpSum."""

from __future__ import annotations

import operator as py_operator
from collections.abc import Hashable
from dataclasses import dataclass, field
from types import MappingProxyType

from operator_algebra._types import Coefficient, as_coefficient
from operator_algebra.algebra import AlgebraRules

_POW_MSG = "Exponent must be a non-negative integer, got {}"


@dataclass(frozen=True, slots=True)
class Op:
    """A single operator acting on one mode.

    Parameters
    ----------
    mode : Hashable
        Any hashable Python object identifying the single-particle state.
    kind : Hashable
        Which operator of the algebra, e.g. ``Ladder.CREATE`` or
        ``Pauli.X``.
    rules : AlgebraRules
        Algebra the operator belongs to.

    """

    mode: Hashable
    kind: Hashable
    rules: AlgebraRules
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compute the hash once; the operator is frozen so it never changes.

        The algebra is hashed by its type. Equal rules have equal types, so
        this agrees with ``__eq__``, and hashing a type runs in C instead of
        the rules' own Python ``__hash__``.
        """
        object.__setattr__(
            self, "_hash", hash((self.mode, self.kind, type(self.rules)))
        )

    def __hash__(self) -> int:
        """Return the hash computed at creation."""
        return self._hash

    def __repr__(self) -> str:
        """Show mode, kind and algebra, e.g. ``Op(0, <Ladder.CREATE: '+'>, ...)``."""
        return f"Op({format_mode(self.mode)}, {self.kind!r}, {self.rules!r})"


@dataclass(frozen=True, slots=True)
class Monomial:
    """An ordered product of operators. No coefficient.

    Parameters
    ----------
    ops : tuple[Op, ...]
        The ordered sequence of operators in this product.

    """

    ops: tuple[Op, ...]
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compute the hash once; tuples don't cache their own hash."""
        object.__setattr__(self, "_hash", hash(self.ops))

    def __len__(self) -> int:
        """Return the number of ladder operators in the product."""
        return len(self.ops)

    def __hash__(self) -> int:
        """Return the hash of the operator tuple, computed at creation."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Compare by operator sequence.

        Two monomials are equal iff they have the same operators in order.
        """
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.ops == other.ops

    def __repr__(self) -> str:
        """Show the operator product, or ``I`` for the empty monomial."""
        if not self.ops:
            return "I"
        return " ".join(repr(op) for op in self.ops)


class OpSum:
    """A linear combination of monomials: sum_i coeff_i * monomial_i.

    Parameters
    ----------
    terms : dict[Monomial, Coefficient] | None
        Mapping from monomials to coefficients.

    """

    def __init__(self, terms: dict[Monomial, Coefficient] | None = None) -> None:
        if terms is None:
            self._terms: dict[Monomial, Coefficient] = {}
        else:
            self._terms = {m: c for m, c in terms.items() if not is_zero(c)}

    def __hash__(self) -> int:
        """Hash by the frozen set of (monomial, coefficient) pairs."""
        return hash(frozenset(self._terms.items()))

    @property
    def terms(self) -> MappingProxyType[Monomial, Coefficient]:
        """Read-only view of the internal terms dict."""
        return MappingProxyType(self._terms)

    def __add__(self, other: object) -> OpSum:
        """Merge terms from both sums, or promote a scalar via identity."""
        if isinstance(other, OpSum):
            merged: dict[Monomial, Coefficient] = dict(self._terms)
            for mono, coeff in other._terms.items():
                if mono in merged:
                    merged[mono] = as_coefficient(merged[mono] + coeff)
                else:
                    merged[mono] = coeff
            return OpSum(merged)
        if is_scalar(other):
            return self + identity() * other
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
            return self - identity() * other
        return NotImplemented

    def __rsub__(self, other: object) -> OpSum:
        """Support ``scalar - OpSum``."""
        if is_scalar(other):
            return (-self) + other
        return NotImplemented

    def __neg__(self) -> OpSum:
        """Negate all coefficients."""
        return OpSum({m: as_coefficient(-c) for m, c in self._terms.items()})

    def __mul__(self, other: object) -> OpSum:
        """Distribute over terms; monomial products concatenate ops."""
        if isinstance(other, OpSum):
            result: dict[Monomial, Coefficient] = {}
            for m1, c1 in self._terms.items():
                for m2, c2 in other._terms.items():
                    new_mono = Monomial(m1.ops + m2.ops)
                    new_coeff = as_coefficient(c1 * c2)
                    if new_mono in result:
                        result[new_mono] = as_coefficient(result[new_mono] + new_coeff)
                    else:
                        result[new_mono] = new_coeff
            return OpSum(result)
        if is_scalar(other):
            return OpSum(
                {
                    m: as_coefficient(py_operator.mul(c, other))
                    for m, c in self._terms.items()
                }
            )
        return NotImplemented

    def __rmul__(self, other: object) -> OpSum:
        """Scale each coefficient from the left for ``scalar * OpSum``."""
        if is_scalar(other):
            return OpSum(
                {
                    m: as_coefficient(py_operator.mul(other, c))
                    for m, c in self._terms.items()
                }
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
            return identity()
        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        """Two OpSums are equal iff monomial-coefficient pairs match."""
        if not isinstance(other, OpSum):
            return NotImplemented
        return self._terms == other._terms

    def __bool__(self) -> bool:
        """Return False when no terms remain (zero operator)."""
        return bool(self._terms)

    def __repr__(self) -> str:
        """Delegate to ``to_text`` for human-readable display."""
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


def is_zero(coeff: Coefficient, tol: float = 0.0) -> bool:
    """Check if a coefficient is zero, with tolerance for floats.

    Parameters
    ----------
    coeff : Coefficient
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


def identity() -> OpSum:
    """Return the identity operator (empty product with coefficient 1).

    Returns
    -------
    OpSum

    """
    return OpSum({Monomial(()): 1})


def zero() -> OpSum:
    """Return the zero operator (empty sum).

    Returns
    -------
    OpSum

    """
    return OpSum({})
