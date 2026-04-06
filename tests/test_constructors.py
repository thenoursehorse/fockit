"""Tests for constructor functions."""

from operator_algebra import c, cdag, identity, n, sm, sp, zero
from operator_algebra.algebra import FERMION
from operator_algebra.constructors import create, sigma_minus, sigma_plus


def test_cdag_tuple_mode():
    assert cdag(1, "up") != cdag(1, "dn")
    assert cdag(1, "up") == cdag(1, "up")


def test_c_annihilation():
    assert c("foo") == c("foo")
    assert c("foo") != cdag("foo")


def test_n_equals_cdag_times_c():
    assert n(1, "up") == cdag(1, "up") * c(1, "up")


def test_create_with_fermion_equals_cdag():
    assert create(1, rules=FERMION) == cdag(1)


def test_identity_is_multiplicative_unit():
    op = cdag(1)
    assert identity() * op == op
    assert op * identity() == op


def test_zero_is_additive_unit():
    op = cdag(1)
    assert op + zero() == op
    assert not zero()


def test_sigma_plus_equals_sp():
    assert sigma_plus(0) == sp(0)


def test_sigma_minus_equals_sm():
    assert sigma_minus(0) == sm(0)
