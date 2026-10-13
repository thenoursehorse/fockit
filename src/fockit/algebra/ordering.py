"""Normal ordering algorithm.

Every algebra goes through the same path, using only its ``swap`` and
``is_normal_ordered_pair`` rules. Pauli strings could be multiplied faster site
by site, or stored as bitmasks, but that would treat one algebra differently
from the rest.
"""

from __future__ import annotations

import functools

from fockit._types import Coefficient, as_coefficient
from fockit.algebra.operators import Monomial, Op, OpSum, as_opsum
from fockit.algebra.rules import SwapResult


def normal_order(op: OpSum | complex) -> OpSum:
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
    op = as_opsum(op)
    terms: dict[Monomial, Coefficient] = {}
    for mono, coeff in op.terms.items():
        for ordered, c in _order(mono.ops, Monomial(())).items():
            _accumulate(terms, ordered, as_coefficient(coeff * c))
    return OpSum(terms)


def is_normal_ordered(op: OpSum | complex) -> bool:
    """Check if an operator expression is in normal order.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    bool

    """
    op = as_opsum(op)
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
    """Normal-order ``op * tail``, where ``tail`` is already normal ordered.

    ``op`` moves right past plain exchanges, picking up only their sign. The
    first swap that does anything else is expanded, and the operators ``op``
    passed are put back in front.
    """
    ops = tail.ops
    sign: Coefficient = 1
    i = 0
    while i < len(ops) and not _is_ordered(op, ops[i]):
        terms = _swap(op, ops[i]).terms
        if len(terms) != 1 or terms[0][1] != (ops[i], op):
            break
        sign = as_coefficient(sign * terms[0][0])
        i += 1

    passed = ops[:i]
    rest: dict[Monomial, Coefficient] = {}
    if i == len(ops) or _is_ordered(op, ops[i]):
        if not passed or _is_ordered(passed[-1], op):
            return ((Monomial((*passed, op, *ops[i:])), sign),)
        rest[Monomial((op, *ops[i:]))] = sign
    else:
        for coeff, repl in _swap(op, ops[i]).terms:
            scale = as_coefficient(sign * coeff)
            for m, c in _order(repl, Monomial(ops[i + 1 :])).items():
                _accumulate(rest, m, as_coefficient(scale * c))

    acc: dict[Monomial, Coefficient] = {}
    for m, c in rest.items():
        if not passed:
            _accumulate(acc, m, c)
        elif not m.ops or _is_ordered(passed[-1], m.ops[0]):
            _accumulate(acc, Monomial(passed + m.ops), c)
        else:
            for ordered, c2 in _order(passed, m).items():
                _accumulate(acc, ordered, as_coefficient(c * c2))
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
