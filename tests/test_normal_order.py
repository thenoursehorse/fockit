"""Tests for normal ordering."""

import pytest

from operator_algebra import (
    anticommutator,
    b,
    bdag,
    c,
    cdag,
    identity,
    n,
    normal_order,
    zero,
)
from operator_algebra.algebra import BOSON


class TestFermionNormalOrder:
    def test_identity(self):
        assert normal_order(identity()) == identity()

    def test_zero(self):
        assert normal_order(zero()) == zero()

    def test_single_creator(self):
        assert normal_order(cdag(1)) == cdag(1)

    def test_single_annihilator(self):
        assert normal_order(c(1)) == c(1)

    def test_already_normal_ordered(self):
        op = cdag(1) * c(2)
        assert normal_order(op) == op

    def test_simple_swap_same_mode(self):
        result = normal_order(c(1) * cdag(1))
        assert result == identity() - cdag(1) * c(1)

    def test_swap_different_modes(self):
        result = normal_order(c(1) * cdag(2))
        assert result == -cdag(2) * c(1)

    def test_nilpotency_annihilator(self):
        assert normal_order(c(1) * c(1)) == zero()

    def test_nilpotency_creator(self):
        assert normal_order(cdag(1) * cdag(1)) == zero()

    def test_number_operator_unchanged(self):
        op = n(1)
        assert normal_order(op) == op

    def test_canonical_ordering_creators(self):
        result = normal_order(cdag(2) * cdag(1))
        expected = -1 * (cdag(1) * cdag(2))
        assert result == expected

    def test_canonical_ordering_annihilators(self):
        result = normal_order(c(2) * c(1))
        expected = -1 * (c(1) * c(2))
        assert result == expected

    def test_anticommutation_same_mode(self):
        result = normal_order(anticommutator(c(1, "up"), cdag(1, "up")))
        assert result == identity()

    def test_anticommutation_different_modes(self):
        result = normal_order(anticommutator(c(1, "up"), cdag(2, "up")))
        assert result == zero()

    def test_idempotency(self):
        for x in [c(1) * cdag(2), cdag(1) * cdag(2) * c(3) * c(4)]:
            once = normal_order(x)
            twice = normal_order(once)
            assert once == twice

    def test_linearity(self):
        x = c(1) * cdag(2)
        y = cdag(3) * c(4)
        lhs = normal_order(3 * x + 2 * y)
        rhs = 3 * normal_order(x) + 2 * normal_order(y)
        assert lhs == rhs

    def test_symbolic_coefficients(self):
        sympy = pytest.importorskip("sympy")
        t = sympy.Symbol("t")
        result = normal_order(t * c(1) * cdag(1))
        assert result == t * identity() - t * cdag(1) * c(1)


class TestBosonNormalOrder:
    def test_simple_swap_same_mode(self):
        result = normal_order(b(1) * bdag(1))
        assert result == identity(BOSON) + bdag(1) * b(1)

    def test_not_nilpotent(self):
        result = normal_order(b(1) * b(1))
        assert result != zero(BOSON)

    def test_swap_different_modes(self):
        result = normal_order(b(1) * bdag(2))
        assert result == bdag(2) * b(1)
