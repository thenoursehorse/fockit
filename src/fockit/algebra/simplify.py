"""Simplification utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from fockit._types import Coefficient, as_coefficient
from fockit.algebra.operators import Monomial, OpSum, as_opsum

_LOGGER = logging.getLogger(__name__)


def drop_zeros(op: OpSum | complex, tol: float = 1e-15, log: bool = False) -> OpSum:
    """Drop terms with negligible coefficients.

    Float coefficients that should cancel leave round-off behind, e.g.
    ``0.1 * nc(0) + 0.2 * nc(0) - 0.3 * nc(0)`` keeps a ``5.6e-17`` term. Only
    exact zeros are removed automatically, so call this before comparing
    a result with ``zero()``.

    Parameters
    ----------
    op : OpSum
    tol : float, optional
        Absolute tolerance, so pick it for the scale of your coefficients.
        Default: 1e-15.
    log : bool, optional
        Log, at INFO level, how many terms were dropped and the largest
        dropped coefficient, to check ``tol`` didn't remove a real term.
        Default: False.

    Returns
    -------
    OpSum

    """
    op = as_opsum(op)
    kept: dict[Monomial, Coefficient] = {}
    dropped: list[float] = []
    for mono, coeff in op.terms.items():
        magnitude = _magnitude(coeff)
        if coeff == 0 or (magnitude is not None and magnitude <= tol):
            dropped.append(magnitude or 0.0)
        else:
            kept[mono] = coeff
    if log:
        _LOGGER.info(
            "Dropped %d of %d terms, largest |coefficient| %.3g",
            len(dropped),
            len(op.terms),
            max(dropped, default=0.0),
        )
    return OpSum(kept)


def simplify_coeffs(op: OpSum | complex) -> OpSum:
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
    op = as_opsum(op)
    import sympy

    simplify = cast(Callable[[object], object], sympy.simplify)
    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        new_terms[mono] = as_coefficient(simplify(coeff))
    return OpSum(new_terms)


def _magnitude(coeff: Coefficient) -> float | None:
    """Return ``abs(coeff)`` as a float, or None if it isn't a number."""
    try:
        magnitude = abs(coeff)
    except TypeError:
        return None
    if isinstance(magnitude, int | float):
        return float(magnitude)
    return None
