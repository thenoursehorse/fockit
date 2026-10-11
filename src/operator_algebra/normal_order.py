"""Normal ordering algorithm."""

from __future__ import annotations

import functools

from operator_algebra._types import Coefficient, as_coefficient
from operator_algebra.algebra import SwapResult
from operator_algebra.operators import Monomial, Op, OpSum


def normal_order(op: OpSum) -> OpSum:
    """Normal-order an operator expression.

    Operators of different algebras commute and are grouped by algebra.

    Parameters
    ----------
    op : OpSum
        The operator to normal-order.

    Returns
    -------
    OpSum
        The normal-ordered operator.

    """
    terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        for ordered, c in _order(mono.ops, Monomial(())).items():
            _accumulate(terms, ordered, as_coefficient(coeff * c))
    return OpSum(terms)


def is_normal_ordered(op: OpSum) -> bool:
    """Check if an operator expression is in normal order.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    bool

    """
    for mono in op.terms:
        ops = mono.ops
        for i in range(len(ops) - 1):
            if not _is_ordered(ops[i], ops[i + 1]):
                return False
    return True


def _is_ordered(left: Op, right: Op) -> bool:
    if type(left.rules) is type(right.rules):
        return left.rules.is_normal_ordered_pair(left, right)
    return type(left.rules).__name__ < type(right.rules).__name__


def _swap(left: Op, right: Op) -> SwapResult:
    if type(left.rules) is type(right.rules):
        return left.rules.swap(left, right)
    return SwapResult(terms=((1, (right, left)),))


def _order(prefix: tuple[Op, ...], tail: Monomial) -> dict[Monomial, Coefficient]:
    """Normal-order ``prefix * tail``, where ``tail`` is already normal ordered."""
    terms: dict[Monomial, Coefficient] = {tail: 1}
    for op in reversed(prefix):
        new: dict[Monomial, Coefficient] = {}
        for t, c1 in terms.items():
            for m, c2 in _insert(op, t):
                _accumulate(new, m, as_coefficient(c1 * c2))
        terms = new
    return terms


@functools.lru_cache(maxsize=2**16)
def _insert(op: Op, tail: Monomial) -> tuple[tuple[Monomial, Coefficient], ...]:
    """Normal-order ``op * tail``, where ``tail`` is already normal ordered."""
    if not tail.ops or _is_ordered(op, tail.ops[0]):
        return ((Monomial((op, *tail.ops)), 1),)
    acc: dict[Monomial, Coefficient] = {}
    for coeff, repl in _swap(op, tail.ops[0]).terms:
        for m, c in _order(repl, Monomial(tail.ops[1:])).items():
            _accumulate(acc, m, as_coefficient(coeff * c))
    return tuple(acc.items())


def _accumulate(
    acc: dict[Monomial, Coefficient], mono: Monomial, coeff: Coefficient
) -> None:
    """Add coeff * mono into the accumulator dict.

    Parameters
    ----------
    acc : dict[Monomial, Coefficient]
    mono : Monomial
    coeff : Coefficient

    """
    if mono in acc:
        acc[mono] = as_coefficient(acc[mono] + coeff)
    else:
        acc[mono] = coeff
