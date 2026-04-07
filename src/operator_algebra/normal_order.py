"""Normal ordering algorithm."""

from __future__ import annotations

import functools
from collections import deque

from operator_algebra.algebra import AlgebraRules
from operator_algebra.operators import Monomial, OpSum, Scalar, zero


def normal_order(op: OpSum) -> OpSum:
    """Normal-order an operator expression.

    Parameters
    ----------
    op : OpSum
        The operator to normal-order.

    Returns
    -------
    OpSum
        The normal-ordered operator.

    Notes
    -----
    For monomials of length n, the algorithm performs O(n²) adjacent swaps.
    Each swap can branch into multiple terms. For typical condensed matter
    Hamiltonians (monomials of length 2 or 4), this is fast.

    """
    result = zero(op.rules)
    for mono, coeff in op.terms.items():
        ordered = normal_order_monomial(mono, op.rules)
        result = result + _scale(ordered, coeff)
    return result


@functools.lru_cache(maxsize=4096)
def normal_order_monomial(mono: Monomial, rules: AlgebraRules) -> OpSum:
    """Normal-order a single monomial using iterative worklist."""
    worklist: deque[tuple[Monomial, Scalar]] = deque([(mono, 1)])
    accumulator: dict[Monomial, Scalar] = {}

    while worklist:
        current_mono, current_coeff = worklist.popleft()
        ops = current_mono.ops

        if len(ops) <= 1:
            _accumulate(accumulator, current_mono, current_coeff)
            continue

        if rules.is_zero_monomial(ops):
            continue

        swap_index = None
        for i in range(len(ops) - 1):
            if not rules.is_normal_ordered_pair(ops[i], ops[i + 1]):
                swap_index = i
                break

        if swap_index is None:
            _accumulate(accumulator, current_mono, current_coeff)
            continue

        i = swap_index
        left = ops[i]
        right = ops[i + 1]
        prefix = ops[:i]
        suffix = ops[i + 2 :]

        swap_result = rules.swap(left, right)

        for coeff, replacement_ops in swap_result.terms:
            new_ops = prefix + replacement_ops + suffix
            new_mono = Monomial(new_ops)
            worklist.append((new_mono, current_coeff * coeff))

    return OpSum(accumulator, rules=rules)


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
            if not op.rules.is_normal_ordered_pair(ops[i], ops[i + 1]):
                return False
    return True


def _scale(op: OpSum, coeff: Scalar) -> OpSum:
    """Multiply all coefficients in op by coeff."""
    return OpSum({m: c * coeff for m, c in op.terms.items()}, rules=op.rules)


def _accumulate(acc: dict[Monomial, Scalar], mono: Monomial, coeff: Scalar) -> None:
    """Add coeff * mono into the accumulator dict.

    Parameters
    ----------
    acc : dict[Monomial, Scalar]
    mono : Monomial
    coeff : Scalar

    """
    if mono in acc:
        acc[mono] = acc[mono] + coeff
    else:
        acc[mono] = coeff
