"""Core data types: LadderOp, Monomial, OpSum."""

from __future__ import annotations

import types as _builtin_types
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from operator_algebra.algebra import FERMION, AlgebraRules

type Mode = Hashable
type Scalar = Any


def _is_zero(coeff: Scalar, tol: float = 0.0) -> bool:
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
    if isinstance(coeff, (float, complex)):
        return abs(coeff) <= tol
    return False


def _is_scalar(obj: object) -> bool:
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
    if isinstance(obj, (int, float, complex)):
        return True
    if isinstance(obj, (str, bytes, list, tuple, set, frozenset, dict)):
        return False
    return hasattr(obj, "__mul__") and hasattr(obj, "__add__")


def _rules_compatible(a: AlgebraRules, b: AlgebraRules) -> bool:
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


def _format_mode(mode: Hashable) -> str:
    """Format a mode label for display."""
    if isinstance(mode, tuple):
        return ", ".join(str(m) for m in mode)
    return str(mode)


@dataclass(frozen=True, slots=True)
class LadderOp:
    """A single creation or annihilation operator acting on one mode.

    Parameters
    ----------
    mode : Hashable
        Any hashable Python object identifying the single-particle state.
    is_creator : bool
        True for creation operator, False for annihilation operator.

    """

    mode: Hashable
    is_creator: bool

    def __repr__(self) -> str:
        symbol = "c†" if self.is_creator else "c"
        return f"{symbol}({_format_mode(self.mode)})"


@dataclass(frozen=True, slots=True)
class Monomial:
    """An ordered product of ladder operators. No coefficient.

    Parameters
    ----------
    ops : tuple[LadderOp, ...]
        The ordered sequence of ladder operators in this product.

    """

    ops: tuple[LadderOp, ...]

    def __len__(self) -> int:
        return len(self.ops)

    def __hash__(self) -> int:
        return hash(self.ops)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.ops == other.ops

    def __repr__(self) -> str:
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
            self._terms = {m: c for m, c in terms.items() if not _is_zero(c)}

    @property
    def terms(self) -> _builtin_types.MappingProxyType[Monomial, Scalar]:
        """Read-only view of the internal terms dict."""
        return _builtin_types.MappingProxyType(self._terms)

    def __add__(self, other: object) -> OpSum:
        if isinstance(other, OpSum):
            if not _rules_compatible(self.rules, other.rules):
                raise ValueError(
                    f"Cannot add OpSum with {self.rules} and {other.rules}"
                )
            merged: dict[Monomial, Scalar] = dict(self._terms)
            for mono, coeff in other._terms.items():
                if mono in merged:
                    merged[mono] = merged[mono] + coeff
                else:
                    merged[mono] = coeff
            return OpSum(merged, rules=self.rules)
        if _is_scalar(other):
            from operator_algebra.constructors import identity

            return self + other * identity(self.rules)
        return NotImplemented

    def __radd__(self, other: object) -> OpSum:
        if _is_scalar(other):
            return self + other
        return NotImplemented

    def __sub__(self, other: object) -> OpSum:
        if isinstance(other, OpSum):
            return self + (-other)
        if _is_scalar(other):
            return self + (-other)  # type: ignore[operator]
        return NotImplemented

    def __rsub__(self, other: object) -> OpSum:
        if _is_scalar(other):
            return (-self) + other
        return NotImplemented

    def __neg__(self) -> OpSum:
        return OpSum({m: -c for m, c in self._terms.items()}, rules=self.rules)

    def __mul__(self, other: object) -> OpSum:
        if isinstance(other, OpSum):
            if not _rules_compatible(self.rules, other.rules):
                raise ValueError(
                    f"Cannot multiply OpSum with {self.rules} and {other.rules}"
                )
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
        if _is_scalar(other):
            return OpSum(
                {m: c * other for m, c in self._terms.items()}, rules=self.rules
            )
        return NotImplemented

    def __rmul__(self, other: object) -> OpSum:
        if _is_scalar(other):
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
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"Exponent must be a non-negative integer, got {n}")
        from operator_algebra.constructors import identity

        if n == 0:
            return identity(self.rules)
        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSum):
            return NotImplemented
        return self._terms == other._terms

    def __bool__(self) -> bool:
        return bool(self._terms)

    def __repr__(self) -> str:
        from operator_algebra.display import format_opsum

        return format_opsum(self)

    def __str__(self) -> str:
        return self.__repr__()
