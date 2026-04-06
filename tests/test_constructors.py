"""Tests for constructor functions."""

from operator_algebra import (
    BOSON,
    FERMION,
    b,
    bdag,
    c,
    cdag,
    create,
    identity,
    n,
    num_terms,
    zero,
)


def test_cdag_creates_single_term():
    assert num_terms(cdag(1)) == 1


def test_cdag_tuple_mode():
    assert cdag(1, "up") != cdag(1, "dn")
    assert cdag(1, "up") == cdag(1, "up")


def test_c_annihilation():
    assert c("foo") == c("foo")
    assert c("foo") != cdag("foo")


def test_n_equals_cdag_times_c():
    assert n(1, "up") == cdag(1, "up") * c(1, "up")


def test_bdag_has_boson_rules():
    assert bdag(1).rules == BOSON


def test_b_has_boson_rules():
    assert b(1).rules == BOSON


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
