"""Tests for simplification utilities."""

import pytest

from operator_algebra import drop_zeros
from operator_algebra.operators import LadderOp, Monomial, OpSum

sympy = pytest.importorskip("sympy")


def test_drop_exact_zeros():
    terms = {
        Monomial((LadderOp(1, True),)): 1,
        Monomial((LadderOp(2, True),)): 0,
    }
    op = OpSum(terms)
    # Constructor already drops exact zeros
    assert len(op._terms) == 1


def test_drop_near_zero_floats():
    terms = {
        Monomial((LadderOp(1, True),)): 1.0,
        Monomial((LadderOp(2, True),)): 1e-16,
    }
    # Bypass constructor filtering by setting _terms directly
    op = OpSum(rules=None)
    op._terms = dict(terms)
    result = drop_zeros(op)
    assert len(result._terms) == 1


def test_drop_zeros_sympy_integer_zero():
    """``coeff == 0`` catches sympy exact zeros."""
    terms = {
        Monomial((LadderOp(1, True),)): 1,
        Monomial((LadderOp(2, True),)): sympy.Integer(0),
    }
    op = OpSum(rules=None)
    op._terms = dict(terms)
    result = drop_zeros(op)
    assert len(result._terms) == 1


def test_drop_zeros_does_not_simplify_symbolic():
    """Symbolic ``t - t`` IS dropped because sympy evaluates ``t - t == 0``."""
    t = sympy.Symbol("t")
    terms = {Monomial((LadderOp(1, True),)): t - t}
    op = OpSum(rules=None)
    op._terms = dict(terms)
    result = drop_zeros(op)
    assert len(result._terms) == 0
