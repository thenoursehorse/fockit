"""Integration tests."""

import pytest

from operator_algebra import (
    anticommutator,
    c,
    cdag,
    hc,
    identity,
    n,
    normal_order,
    num_terms,
    zero,
)


def test_hubbard_model():
    sympy = pytest.importorskip("sympy")
    t, U = sympy.Symbol("t"), sympy.Symbol("U")
    up, dn = "up", "dn"

    H_hop = sum(
        -t * (cdag(i, s) * c(j, s) + cdag(j, s) * c(i, s))
        for i, j in [(0, 1)]
        for s in [up, dn]
    )
    H_int = sum(U * n(i, up) * n(i, dn) for i in [0, 1])
    H = H_hop + H_int

    H_no = normal_order(H)
    assert normal_order(H_no) == H_no  # idempotency


def test_hermiticity():
    sympy = pytest.importorskip("sympy")
    t = sympy.Symbol("t", real=True)
    U = sympy.Symbol("U", real=True)
    up, dn = "up", "dn"

    H_hop = sum(
        -t * (cdag(i, s) * c(j, s) + cdag(j, s) * c(i, s))
        for i, j in [(0, 1)]
        for s in [up, dn]
    )
    H_int = sum(U * n(i, up) * n(i, dn) for i in [0, 1])
    H = H_hop + H_int

    assert normal_order(H - hc(H)) == zero()


def test_anticommutation_relations():
    up, dn = "up", "dn"
    for s1, s2 in [(up, up), (up, dn), (dn, dn)]:
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            result = normal_order(anticommutator(c(i, s1), cdag(j, s2)))
            if i == j and s1 == s2:
                assert result == identity()
            else:
                assert result == zero()


def test_hopping_hamiltonian_term_count():
    H = sum(cdag(i) * c(i + 1) + cdag(i + 1) * c(i) for i in range(3))
    assert num_terms(H) == 6


def test_mixed_mode_types():
    """Mixing int and str modes should not raise."""
    op = cdag(0) * c("a")
    result = normal_order(op)
    assert result == op


def test_scalar_addition_in_hamiltonian():
    sympy = pytest.importorskip("sympy")
    E_0 = sympy.Symbol("E_0")
    H = cdag(0) * c(1) + E_0
    assert num_terms(H) == 2
