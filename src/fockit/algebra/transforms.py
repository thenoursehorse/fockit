"""Transforms for conjugation, commutators, and mode substitution."""

import warnings
from collections.abc import Hashable, Mapping
from dataclasses import replace

from fockit._types import Coefficient, as_coefficient
from fockit.algebra.operators import Monomial, Op, OpSum, as_opsum, identity
from fockit.algebra.ordering import normal_order


def _conjugate(coeff: Coefficient) -> Coefficient:
    """Complex-conjugate a coefficient.

    Parameters
    ----------
    coeff : Coefficient
        The coefficient to conjugate.

    Returns
    -------
    Coefficient

    Raises
    ------
    TypeError
        If the coefficient type does not support ``.conjugate()``.

    """
    try:
        return as_coefficient(coeff.conjugate())
    except AttributeError:
        type_name = type(coeff).__name__
        warnings.warn(
            f"Coefficient of type {type_name!r} does not support "
            f".conjugate(). If this type is real-valued, define a .conjugate() "
            f"method returning self. Raising TypeError.",
            stacklevel=3,
        )
        msg = (
            f"Cannot conjugate coefficient of type {type_name!r}: "
            f"no .conjugate() method."
        )
        raise TypeError(msg) from None


def hc(op: OpSum | complex) -> OpSum:
    """Hermitian conjugate of an operator expression.

    Reverses operator order, flips creation/annihilation, and
    complex-conjugates coefficients.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    OpSum

    """
    op = as_opsum(op)
    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        new_ops = tuple(
            single_op.rules.dagger(single_op) for single_op in reversed(mono.ops)
        )
        new_terms[Monomial(new_ops)] = _conjugate(coeff)
    return OpSum(new_terms)


def commutator(a: OpSum | complex, b: OpSum | complex) -> OpSum:
    """Commutator [A, B] = AB - BA.

    Parameters
    ----------
    a : OpSum
    b : OpSum

    Returns
    -------
    OpSum
        The commutator (NOT normal ordered).

    """
    a = as_opsum(a)
    b = as_opsum(b)
    return a * b - b * a


def anticommutator(a: OpSum | complex, b: OpSum | complex) -> OpSum:
    """Anticommutator {A, B} = AB + BA.

    Parameters
    ----------
    a : OpSum
    b : OpSum

    Returns
    -------
    OpSum
        The anticommutator (NOT normal ordered).

    """
    a = as_opsum(a)
    b = as_opsum(b)
    return a * b + b * a


def substitute(op: OpSum | complex, mapping: Mapping[OpSum, OpSum]) -> OpSum:
    """Replace single operators by expressions.

    Parameters
    ----------
    op : OpSum
        The operator expression.
    mapping : Mapping[OpSum, OpSum]
        Each key is a single operator, e.g. ``c(0)``, mapped to what replaces
        it, e.g. ``(gamma(0) + 1j * gamma(1)) / 2``. Operators not in the
        mapping are unchanged.

    Returns
    -------
    OpSum
        The result, not normal ordered.

    """
    op = as_opsum(op)
    images: dict[Op, OpSum] = {}
    for key, value in mapping.items():
        terms = list(key.terms.items())
        if len(terms) != 1 or len(terms[0][0].ops) != 1 or terms[0][1] != 1:
            msg = f"Each key must be a single operator, e.g. c(0), not {key}."
            raise ValueError(msg)
        images[terms[0][0].ops[0]] = value

    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        product = identity() * coeff
        for single_op in mono.ops:
            product = product * images.get(
                single_op, OpSum({Monomial((single_op,)): 1})
            )
        for new_mono, new_coeff in product.terms.items():
            if new_mono in new_terms:
                new_terms[new_mono] = as_coefficient(new_terms[new_mono] + new_coeff)
            else:
                new_terms[new_mono] = new_coeff
    return OpSum(new_terms)


def substitute_modes(
    op: OpSum | complex,
    mapping: Mapping[Hashable, Hashable],
    *,
    reorder: bool = False,
) -> OpSum:
    """Relabel modes according to a mapping.

    Parameters
    ----------
    op : OpSum
        The operator expression.
    mapping : Mapping[Hashable, Hashable]
        Mode relabeling map. Modes not in the mapping are unchanged.
    reorder : bool, optional
        If True, ``normal_order()`` is applied after substitution.
        Default: False.

    Returns
    -------
    OpSum

    """
    op = as_opsum(op)
    if len(set(mapping.values())) < len(mapping) and not reorder:
        warnings.warn(
            "Non-injective mode mapping detected. Result may contain "
            "structurally nonzero but algebraically zero terms. "
            "Set reorder=True or call normal_order() on the result.",
            stacklevel=2,
        )

    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        new_ops = tuple(
            replace(single_op, mode=mapping.get(single_op.mode, single_op.mode))
            for single_op in mono.ops
        )
        new_mono = Monomial(new_ops)
        if new_mono in new_terms:
            new_terms[new_mono] = as_coefficient(new_terms[new_mono] + coeff)
        else:
            new_terms[new_mono] = coeff

    result = OpSum(new_terms)
    if reorder:
        result = normal_order(result)
    return result
