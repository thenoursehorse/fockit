"""Tests for constructor functions."""

from operator_algebra import (
    BOSON,
    FERMION,
    LadderOp,
    Monomial,
    b,
    bdag,
    c,
    cdag,
    create,
    identity,
    n,
    zero,
)


def test_cdag_single_mode():
    op = cdag(1)
    mono, coeff = next(iter(op.terms.items()))
    assert coeff == 1
    assert mono.ops == (LadderOp(1, True),)


def test_cdag_tuple_mode():
    op = cdag(1, "up")
    mono = next(iter(op.terms.keys()))
    assert mono.ops[0].mode == (1, "up")


def test_c_annihilation():
    op = c("foo")
    mono = next(iter(op.terms.keys()))
    assert mono.ops[0] == LadderOp("foo", False)


def test_n_equals_cdag_times_c():
    assert n(1, "up") == cdag(1, "up") * c(1, "up")


def test_bdag_has_boson_rules():
    assert bdag(1).rules == BOSON


def test_b_has_boson_rules():
    assert b(1).rules == BOSON


def test_create_with_fermion_equals_cdag():
    assert create(1, rules=FERMION) == cdag(1)


def test_identity():
    ident = identity()
    mono, coeff = next(iter(ident.terms.items()))
    assert mono == Monomial(())
    assert coeff == 1


def test_zero():
    z = zero()
    assert len(z._terms) == 0
