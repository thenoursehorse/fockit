"""Tests for transforms."""

import warnings

import pytest

from fockit import (
    c,
    cdag,
    commutator,
    hc,
    identity,
    n,
    normal_order,
    sigma_x,
    sigma_y,
    sigma_z,
    sm,
    sp,
    substitute,
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
        assert result != zero()
        assert normal_order(result) == zero()

    def test_non_injective_with_reorder(self):
        op = cdag(1) * cdag(2)
        result = substitute_modes(op, {1: "x", 2: "x"}, reorder=True)
        assert result == zero()


def test_jordan_wigner_turns_a_fermion_chain_into_the_xy_chain():
    sites = 5
    jordan_wigner = {}
    for j in range(sites):
        string = identity()
        for k in range(j):
            string = string * sigma_z(k)
        jordan_wigner[c(j)] = string * sm(j)
        jordan_wigner[cdag(j)] = string * sp(j)
    hopping = -sum(cdag(j + 1) * c(j) + cdag(j) * c(j + 1) for j in range(sites - 1))
    xy = 0.5 * sum(
        sigma_x(j) * sigma_x(j + 1) + sigma_y(j) * sigma_y(j + 1)
        for j in range(sites - 1)
    )
    assert normal_order(substitute(hopping, jordan_wigner)) == normal_order(xy)


def test_substitute_keys_must_be_single_operators():
    with pytest.raises(ValueError, match="single operator"):
        substitute(n(0), {n(0): cdag(0)})
