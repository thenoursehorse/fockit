"""Tests for core data types."""

import pytest

from operator_algebra import (
    BOSON,
    LadderOp,
    Monomial,
    bdag,
    c,
    cdag,
    identity,
    zero,
)
from operator_algebra.operators import is_scalar, is_zero


class TestLadderOp:
    def test_frozen_and_hashable(self):
        op = LadderOp(1, True)
        assert hash(op) == hash(LadderOp(1, True))
        with pytest.raises(AttributeError):
            op.mode = 2  # type: ignore[misc]

    def test_various_mode_types(self):
        LadderOp(1, True)
        LadderOp("up", False)
        LadderOp((1, "up"), True)
        LadderOp(frozenset({1, 2}), False)

    def test_equality(self):
        assert LadderOp(1, True) == LadderOp(1, True)
        assert LadderOp(1, True) != LadderOp(2, True)
        assert LadderOp(1, True) != LadderOp(1, False)

    def test_bool_int_aliasing(self):
        assert LadderOp(True, True) == LadderOp(1, True)

    def test_repr(self):
        assert repr(LadderOp(3, False)) == "c(3)"
        assert repr(LadderOp(("Fe1", "up"), True)) == "c†(Fe1, up)"


class TestMonomial:
    def test_hashable_and_equal(self):
        m1 = Monomial((LadderOp(1, True), LadderOp(2, False)))
        m2 = Monomial((LadderOp(1, True), LadderOp(2, False)))
        assert m1 == m2
        assert hash(m1) == hash(m2)

    def test_len_returns_num_operators(self):
        m = Monomial((LadderOp(1, True), LadderOp(2, False)))
        assert len(m) == 2
        assert len(Monomial(())) == 0

    def test_repr(self):
        m = Monomial((LadderOp(1, True), LadderOp(2, False)))
        assert repr(m) == "c†(1) c(2)"
        assert repr(Monomial(())) == "I"


class TestOpSum:
    def test_addition_merges_terms(self):
        result = cdag(1) + cdag(2)
        assert len(result._terms) == 2

    def test_addition_combines_like_terms(self):
        result = cdag(1) + cdag(1)
        assert list(result._terms.values()) == [2]

    def test_subtraction(self):
        result = cdag(1) - cdag(1)
        assert result == zero()

    def test_negation(self):
        result = -cdag(1)
        assert list(result._terms.values()) == [-1]

    def test_scalar_mul_right(self):
        result = cdag(1) * 3
        assert list(result._terms.values()) == [3]

    def test_scalar_mul_left(self):
        result = 3 * cdag(1)
        assert list(result._terms.values()) == [3]

    def test_scalar_addition(self):
        result = cdag(1) + 5
        assert result == cdag(1) + 5 * identity()

    def test_scalar_radd(self):
        result = 5 + cdag(1)
        assert result == 5 * identity() + cdag(1)

    def test_scalar_sub(self):
        result = cdag(1) - 3
        assert result == cdag(1) + (-3) * identity()

    def test_scalar_rsub(self):
        result = 3 - cdag(1)
        assert result == 3 * identity() - cdag(1)

    def test_mul_opsum(self):
        result = cdag(1) * c(2)
        mono = next(iter(result._terms))
        assert len(mono) == 2

    def test_pow(self):
        op = cdag(1)
        assert op**0 == identity()
        assert op**1 == op
        assert op**2 == op * op
        assert op**3 == op * op * op

    def test_pow_preserves_rules(self):
        result = bdag(1) ** 0
        assert result.rules == BOSON

    def test_zero_pow_zero(self):
        assert zero() ** 0 == identity()

    def test_sum_builtin(self):
        ops = [cdag(1), cdag(2), cdag(3)]
        result = sum(ops, zero())
        assert len(result._terms) == 3

    def test_sum_builtin_radd_with_int_zero(self):
        """``sum([op1, op2])`` calls ``0 + op1`` via __radd__."""
        result = sum([cdag(1), cdag(2)])
        assert len(result._terms) == 2

    def test_structural_equality(self):
        assert cdag(1) == cdag(1)
        assert cdag(1) != cdag(2)

    def test_mixing_rules_mul_raises(self):
        with pytest.raises(ValueError, match="Cannot multiply"):
            cdag(1) * bdag(2)

    def test_mixing_rules_add_raises(self):
        with pytest.raises(ValueError, match="Cannot add"):
            cdag(1) + bdag(2)

    def test_bool_zero_is_falsy(self):
        assert not zero()

    def test_bool_nonzero_is_truthy(self):
        assert cdag(1)

    def test_scalar_list_raises(self):
        with pytest.raises(TypeError):
            cdag(1) * [1, 2, 3]  # type: ignore[operator]


class TestIsScalar:
    def test_int_float_complex(self):
        assert is_scalar(1)
        assert is_scalar(1.0)
        assert is_scalar(1j)

    def test_sympy_symbol(self):
        sympy = pytest.importorskip("sympy")
        assert is_scalar(sympy.Symbol("t"))

    def test_opsum_not_scalar(self):
        assert not is_scalar(cdag(1))

    def test_list_dict_set_not_scalar(self):
        assert not is_scalar([1])
        assert not is_scalar({1: 2})
        assert not is_scalar({1})


class TestIsZero:
    def test_exact_zeros(self):
        assert is_zero(0)
        assert is_zero(0.0)
        assert is_zero(0j)

    def test_nonzero(self):
        assert not is_zero(1)
        assert not is_zero(1e-10)

    def test_tolerance(self):
        assert is_zero(1e-16, tol=1e-15)
        assert not is_zero(1e-14, tol=1e-15)
