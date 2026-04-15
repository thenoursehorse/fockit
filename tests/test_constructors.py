"""Tests for constructor functions."""

from operator_algebra import c, cdag, identity, n, zero
from operator_algebra.algebra import FERMION
from operator_algebra.constructors import create


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
