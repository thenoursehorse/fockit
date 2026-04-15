"""Simplification utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from operator_algebra._types import Coefficient, as_coefficient
from operator_algebra.operators import Monomial, OpSum


def drop_zeros(op: OpSum, tol: float = 1e-15) -> OpSum:
    """Drop terms with negligible coefficients.

    Parameters
    ----------
    op : OpSum
    tol : float, optional
        Tolerance for zero detection. Default: 1e-15.

    Returns
    -------
    OpSum

    """
    kept: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        if coeff == 0:
            continue
        if isinstance(coeff, float | complex) and abs(coeff) <= tol:
            continue
        if not isinstance(coeff, int | float | complex):
            try:
                magnitude = abs(coeff)
                if isinstance(magnitude, int | float) and magnitude <= tol:
                    continue
            except TypeError:
                pass
        kept[mono] = coeff
    return OpSum(kept, rules=op.rules)


def simplify_coeffs(op: OpSum) -> OpSum:
    """Simplify symbolic coefficients using sympy.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    OpSum

    Raises
    ------
    ImportError
        If sympy is not installed.

    """
    import sympy

    simplify = cast(Callable[[object], object], sympy.simplify)
    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        new_terms[mono] = as_coefficient(simplify(coeff))
    return OpSum(new_terms, rules=op.rules)
