"""Inspection utilities for operator expressions."""

from __future__ import annotations

from collections.abc import Hashable, Iterator

from operator_algebra.operators import LadderOp, OpSum, Scalar


def monomials(op: OpSum) -> Iterator[tuple[Scalar, tuple[LadderOp, ...]]]:
    """Iterate over terms as (coefficient, operators) pairs.

    Parameters
    ----------
    op : OpSum

    Yields
    ------
    tuple[Scalar, tuple[LadderOp, ...]]

    """
    for mono, coeff in op.terms.items():
        yield coeff, mono.ops


def modes(op: OpSum) -> set[Hashable]:
    """Return the set of all mode labels appearing in the operator.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    set[Hashable]

    """
    result: set[Hashable] = set()
    for mono in op.terms:
        for ladder_op in mono.ops:
            result.add(ladder_op.mode)
    return result


def degree(op: OpSum) -> int:
    """Maximum monomial length (number of ladder operators).

    Returns 0 for the zero operator.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    int

    """
    if not op._terms:
        return 0
    return max(len(mono) for mono in op._terms)


def num_terms(op: OpSum) -> int:
    """Return the number of nonzero terms.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    int

    """
    return len(op._terms)
