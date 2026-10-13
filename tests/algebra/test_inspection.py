"""Tests for inspection utilities."""

from fockit.algebra import c, cdag, degree, modes, num_terms, zero


def test_modes() -> None:
    op = cdag(1, "up") * c(2, "dn")
    assert modes(op) == {(1, "up"), (2, "dn")}


def test_degree() -> None:
    op = cdag(1) * c(2) + cdag(3)
    assert degree(op) == 2


def test_degree_zero() -> None:
    assert degree(zero()) == 0


def test_num_terms() -> None:
    op = cdag(1) + cdag(2)
    assert num_terms(op) == 2
