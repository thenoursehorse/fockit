"""Tests for simplification utilities."""

import pytest

from fockit import cdag, drop_zeros, num_terms, zero

sympy = pytest.importorskip("sympy")


def test_drop_near_zero_floats():
    op = cdag(1) + 1e-16 * cdag(2)
    result = drop_zeros(op)
    assert result == cdag(1)


def test_drop_zeros_keeps_significant_terms():
    op = cdag(1) + 0.5 * cdag(2)
    result = drop_zeros(op)
    assert num_terms(result) == 2


def test_drop_zeros_sympy_zero():
    op = cdag(1) + sympy.Integer(0) * cdag(2)
    result = drop_zeros(op)
    assert result == cdag(1)


def test_drop_zeros_sympy_cancel():
    t = sympy.Symbol("t")
    op = (t - t) * cdag(1)
    result = drop_zeros(op)
    assert result == zero()
