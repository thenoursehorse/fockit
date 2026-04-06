"""Simplification utilities."""

from __future__ import annotations

from operator_algebra.operators import Monomial, OpSum, Scalar


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
        if not isinstance(coeff, (int, float, complex)):
            try:
                if abs(coeff) <= tol:
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
    import sympy as _sympy  # noqa: PLC0415

    new_terms: dict[Monomial, Scalar] = {}
    for mono, coeff in op._terms.items():
        new_terms[mono] = _sympy.simplify(coeff)
    return OpSum(new_terms, rules=op.rules)
