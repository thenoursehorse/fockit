"""Simplification utilities."""

from __future__ import annotations

from operator_algebra.types import Monomial, OpSum, Scalar


def collect_terms(op: OpSum) -> OpSum:
    """Re-collect like monomials, drop zeros.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    OpSum
    """
    return OpSum(dict(op._terms), rules=op.rules)


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
    kept: dict[Monomial, Scalar] = {}
    for mono, coeff in op._terms.items():
        if coeff == 0:
            continue
        if isinstance(coeff, (float, complex)) and abs(coeff) <= tol:
            continue
        # For exotic types, try abs comparison but don't fail
        if not isinstance(coeff, (int, float, complex)):
            try:
                if abs(coeff) <= tol:  # type: ignore[arg-type]
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

    new_terms: dict[Monomial, Scalar] = {}
    for mono, coeff in op._terms.items():
        new_terms[mono] = sympy.simplify(coeff)
    return OpSum(new_terms, rules=op.rules)
