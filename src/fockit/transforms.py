"""Transforms for conjugation, commutators, and mode substitution."""

import warnings
from collections.abc import Hashable, Mapping
from dataclasses import replace

from fockit._types import Coefficient, as_coefficient
from fockit.normal_order import normal_order
from fockit.operators import Monomial, OpSum


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


def hc(op: OpSum) -> OpSum:
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
    new_terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        new_ops = tuple(
            single_op.rules.dagger(single_op) for single_op in reversed(mono.ops)
        )
        new_terms[Monomial(new_ops)] = _conjugate(coeff)
    return OpSum(new_terms)


def commutator(a: OpSum, b: OpSum) -> OpSum:
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
    return a * b - b * a


def anticommutator(a: OpSum, b: OpSum) -> OpSum:
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
    return a * b + b * a


def substitute_modes(
    op: OpSum,
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
