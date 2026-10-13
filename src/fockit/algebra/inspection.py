"""Inspection utilities for operator expressions."""

from __future__ import annotations

from collections.abc import Hashable

from fockit.algebra.operators import OpSum, as_opsum


def modes(op: OpSum | complex) -> set[Hashable]:
    """Return the set of all mode labels appearing in the operator.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    set[Hashable]

    """
    op = as_opsum(op)
    result: set[Hashable] = set()
    for mono in op.terms:
        for ladder_op in mono.ops:
            result.add(ladder_op.mode)
    return result


def degree(op: OpSum | complex) -> int:
    """Maximum monomial length (number of ladder operators).

    Returns 0 for the zero operator.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    int

    """
    op = as_opsum(op)
    if not op.terms:
        return 0
    return max(len(mono) for mono in op.terms)


def num_terms(op: OpSum | complex) -> int:
    """Return the number of nonzero terms.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    int

    """
    op = as_opsum(op)
    return len(op.terms)
