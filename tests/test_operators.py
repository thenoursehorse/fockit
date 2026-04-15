"""Tests for core data types."""

import pytest

from operator_algebra import bdag, cdag, identity, num_terms, zero


class TestOpSum:
    def test_addition_merges_terms(self):
        result = cdag(1) + cdag(2)
        assert num_terms(result) == 2

    def test_addition_combines_like_terms(self):
        assert cdag(1) + cdag(1) == 2 * cdag(1)

    def test_subtraction_cancels(self):
        assert cdag(1) - cdag(1) == zero()

    def test_negation(self):
        assert -cdag(1) == -1 * cdag(1)

    def test_scalar_multiplication_commutes(self):
        assert 3 * cdag(1) == cdag(1) * 3

    def test_scalar_acts_as_identity_scaled(self):
        for expr, expected in [
            (cdag(1) + 5, cdag(1) + 5 * identity()),
            (5 + cdag(1), 5 * identity() + cdag(1)),
            (cdag(1) - 3, cdag(1) + (-3) * identity()),
            (3 - cdag(1), 3 * identity() - cdag(1)),
        ]:
            assert expr == expected

    def test_pow(self):
        op = cdag(1)
        assert op**0 == identity()
        assert op**2 == op * op
        assert op**3 == op * op * op
        assert zero() ** 0 == identity()

    def test_builtin_sum(self):
        assert sum([cdag(1), cdag(2), cdag(3)]) == cdag(1) + cdag(2) + cdag(3)

    def test_mixing_rules_raises(self):
        with pytest.raises(ValueError, match="Cannot multiply"):
            cdag(1) * bdag(2)
        with pytest.raises(ValueError, match="Cannot add"):
            cdag(1) + bdag(2)

    def test_truthiness(self):
        assert not zero()
        assert cdag(1)
