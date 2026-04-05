"""Tests for transforms."""

import warnings

import pytest

from operator_algebra import (
    c,
    cdag,
    commutator,
    hc,
    n,
    normal_order,
    substitute_modes,
    zero,
)


class TestHC:
    def test_hc_creator(self):
        assert hc(cdag(1)) == c(1)

    def test_hc_annihilator(self):
        assert hc(c(1)) == cdag(1)

    def test_hc_product(self):
        assert hc(cdag(1) * c(2)) == cdag(2) * c(1)

    def test_hc_involution(self):
        for x in [cdag(1), c(1) * cdag(2), 3 * cdag(1) + 2 * c(2)]:
            assert hc(hc(x)) == x

    def test_hc_complex_coeff(self):
        assert hc(1j * cdag(1)) == -1j * c(1)

    def test_hc_exotic_no_conjugate_raises(self):
        """Coefficient type without .conjugate() raises TypeError."""

        class NoConj:
            def __mul__(self, other):
                return self

            def __add__(self, other):
                return self

            def __neg__(self):
                return self

            def __eq__(self, other):
                return False

        from operator_algebra.types import LadderOp, Monomial, OpSum

        op = OpSum({Monomial((LadderOp(1, True),)): NoConj()})
        with pytest.raises(TypeError, match="conjugate"):
            hc(op)


class TestCommutator:
    def test_n_cdag(self):
        result = normal_order(commutator(n(1), cdag(1)))
        assert result == cdag(1)

    def test_n_c(self):
        result = normal_order(commutator(n(1), c(1)))
        assert result == -c(1)


class TestSubstituteModes:
    def test_simple_relabel(self):
        op = cdag(1) * c(2)
        result = substitute_modes(op, {1: "a", 2: "b"})
        assert result == cdag("a") * c("b")

    def test_non_injective_warns(self):
        op = cdag(1) * cdag(2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = substitute_modes(op, {1: "x", 2: "x"})
            assert len(w) == 1
            assert "Non-injective" in str(w[0].message)
        # Structurally nonzero but algebraically zero
        assert result != zero()
        assert normal_order(result) == zero()

    def test_non_injective_with_reorder(self):
        op = cdag(1) * cdag(2)
        result = substitute_modes(op, {1: "x", 2: "x"}, reorder=True)
        assert result == zero()
