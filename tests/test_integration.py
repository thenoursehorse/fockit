"""Integration tests."""

import pytest

from operator_algebra import (
    anticommutator,
    c,
    cdag,
    commutator,
    hc,
    identity,
    n,
    normal_order,
    num_terms,
    sigma_x,
    sigma_y,
    sigma_z,
    sm,
    sp,
    zero,
)
from operator_algebra.algebra import PAULI


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


class TestPauliIntegration:
    def test_pauli_product_rule(self):
        x, y, z = sigma_x(0), sigma_y(0), sigma_z(0)
        one = identity(PAULI)
        cases = [
            (x, x, one),
            (y, y, one),
            (z, z, one),
            (x, y, 1j * z),
            (y, z, 1j * x),
            (z, x, 1j * y),
            (y, x, -1j * z),
            (z, y, -1j * x),
            (x, z, -1j * y),
        ]
        for a, b, expected in cases:
            assert normal_order(a * b) == expected
        assert normal_order(x * y * z) == 1j * one

    def test_sigma_plus_minus_ladder_relations(self):
        z = sigma_z(0)
        assert normal_order(sp(0) * sp(0)) == zero(PAULI)
        assert normal_order(sm(0) * sm(0)) == zero(PAULI)
        assert normal_order(commutator(sp(0), sm(0))) == z
        assert normal_order(commutator(z, sp(0))) == 2 * sp(0)
        assert normal_order(commutator(z, sm(0))) == -2 * sm(0)
        assert hc(sp(0)) == sm(0)

    def test_heisenberg_bond_in_ladder_form(self):
        zz = sigma_z(0) * sigma_z(1)
        xyz = sigma_x(0) * sigma_x(1) + sigma_y(0) * sigma_y(1) + zz
        ladder = 2 * (sp(0) * sm(1) + sm(0) * sp(1)) + zz
        assert normal_order(xyz) == normal_order(ladder)

    def test_heisenberg_conserves_total_sz(self):
        H = sum(
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
            for i in range(2)
        )
        Sz = sum(sigma_z(i) for i in range(3))
        assert normal_order(commutator(H, Sz)) == zero(PAULI)

    def test_anticommutator_same_gives_two_identity(self):
        for sigma in (sigma_x, sigma_y, sigma_z):
            result = normal_order(anticommutator(sigma(0), sigma(0)))
            assert result == 2 * identity(PAULI)

    def test_anticommutator_different_gives_zero(self):
        pairs = [(sigma_x, sigma_y), (sigma_x, sigma_z), (sigma_y, sigma_z)]
        for sa, sb in pairs:
            result = normal_order(anticommutator(sa(0), sb(0)))
            assert result == zero(PAULI)

    def test_different_sites_commute(self):
        result = normal_order(commutator(sigma_x(0), sigma_y(1)))
        assert result == zero(PAULI)

    def test_hc_pauli_is_self(self):
        for sigma in (sigma_x, sigma_y, sigma_z):
            assert hc(sigma(0)) == sigma(0)
