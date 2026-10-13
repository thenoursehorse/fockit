"""Integration tests."""

import logging
from collections.abc import Callable, Sequence

import pytest

from fockit.algebra import (
    OpSum,
    anticommutator,
    b,
    bdag,
    c,
    cdag,
    commutator,
    drop_zeros,
    hc,
    identity,
    nc,
    normal_order,
    num_terms,
    sigma_x,
    sigma_y,
    sigma_z,
    sm,
    sp,
    zero,
)


def test_hubbard_model() -> None:
    sympy = pytest.importorskip("sympy")
    t, U = sympy.Symbol("t"), sympy.Symbol("U")
    up, dn = "up", "dn"

    H_hop = sum(
        -t * (cdag(i, s) * c(j, s) + cdag(j, s) * c(i, s))
        for i, j in [(0, 1)]
        for s in [up, dn]
    )
    H_int = sum(U * nc(i, up) * nc(i, dn) for i in [0, 1])
    H = H_hop + H_int

    H_no = normal_order(H)
    assert normal_order(H_no) == H_no  # idempotency


def test_hermiticity() -> None:
    sympy = pytest.importorskip("sympy")
    t = sympy.Symbol("t", real=True)
    U = sympy.Symbol("U", real=True)
    up, dn = "up", "dn"

    H_hop = sum(
        -t * (cdag(i, s) * c(j, s) + cdag(j, s) * c(i, s))
        for i, j in [(0, 1)]
        for s in [up, dn]
    )
    H_int = sum(U * nc(i, up) * nc(i, dn) for i in [0, 1])
    H = H_hop + H_int

    assert normal_order(H - hc(H)) == zero()


def test_anticommutation_relations() -> None:
    up, dn = "up", "dn"
    for s1, s2 in [(up, up), (up, dn), (dn, dn)]:
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            result = normal_order(anticommutator(c(i, s1), cdag(j, s2)))
            if i == j and s1 == s2:
                assert result == identity()
            else:
                assert result == zero()


def test_hopping_hamiltonian_term_count() -> None:
    H = sum(cdag(i) * c(i + 1) + cdag(i + 1) * c(i) for i in range(3))
    assert num_terms(H) == 6


def test_mixed_mode_types() -> None:
    """Mixing int and str modes should not raise."""
    op = cdag(0) * c("a")
    result = normal_order(op)
    assert result == op


def test_scalar_addition_in_hamiltonian() -> None:
    sympy = pytest.importorskip("sympy")
    E_0 = sympy.Symbol("E_0")
    H = cdag(0) * c(1) + E_0
    assert num_terms(H) == 2


class TestPauliIntegration:
    def test_pauli_product_rule(self) -> None:
        x, y, z = sigma_x(0), sigma_y(0), sigma_z(0)
        one = identity()
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
        for left, right, expected in cases:
            assert normal_order(left * right) == expected
        assert normal_order(x * y * z) == 1j * one

    def test_sigma_plus_minus_ladder_relations(self) -> None:
        z = sigma_z(0)
        assert normal_order(sp(0) * sp(0)) == zero()
        assert normal_order(sm(0) * sm(0)) == zero()
        assert normal_order(commutator(sp(0), sm(0))) == z
        assert normal_order(commutator(z, sp(0))) == 2 * sp(0)
        assert normal_order(commutator(z, sm(0))) == -2 * sm(0)
        assert hc(sp(0)) == sm(0)

    def test_heisenberg_bond_in_ladder_form(self) -> None:
        zz = sigma_z(0) * sigma_z(1)
        xyz = sigma_x(0) * sigma_x(1) + sigma_y(0) * sigma_y(1) + zz
        ladder = 2 * (sp(0) * sm(1) + sm(0) * sp(1)) + zz
        assert normal_order(xyz) == normal_order(ladder)

    def test_heisenberg_conserves_total_sz(self) -> None:
        H = sum(
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
            for i in range(2)
        )
        Sz = sum(sigma_z(i) for i in range(3))
        assert normal_order(commutator(H, Sz)) == zero()

    def test_heisenberg_ring_conserves_total_spin_up_to_round_off(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sites, J = 4, 0.1
        H = J * sum(
            sigma(i) * sigma((i + 1) % sites)
            for i in range(sites)
            for sigma in (sigma_x, sigma_y, sigma_z)
        )
        S2 = sum(
            sum(0.5 * sigma(i) for i in range(sites)) ** 2
            for sigma in (sigma_x, sigma_y, sigma_z)
        )
        with caplog.at_level(logging.INFO):
            assert drop_zeros(normal_order(commutator(H, S2))) == zero()
        assert "Dropped 22 of 22 terms" in caplog.text

    def test_anticommutator_same_gives_two_identity(self) -> None:
        for sigma in (sigma_x, sigma_y, sigma_z):
            result = normal_order(anticommutator(sigma(0), sigma(0)))
            assert result == 2 * identity()

    def test_anticommutator_different_gives_zero(self) -> None:
        pairs = [(sigma_x, sigma_y), (sigma_x, sigma_z), (sigma_y, sigma_z)]
        for sa, sb in pairs:
            result = normal_order(anticommutator(sa(0), sb(0)))
            assert result == zero()

    def test_different_sites_commute(self) -> None:
        result = normal_order(commutator(sigma_x(0), sigma_y(1)))
        assert result == zero()

    def test_hc_pauli_is_self(self) -> None:
        for sigma in (sigma_x, sigma_y, sigma_z):
            assert hc(sigma(0)) == sigma(0)

    def test_jordan_wigner_strings_obey_fermion_anticommutation(self) -> None:
        def jordan_wigner(j: int) -> OpSum:
            string = identity()
            for k in range(j):
                string = string * sigma_z(k)
            return string * sm(j)

        last = 59
        for i, j in [(0, 0), (0, last), (17, 42), (last, last)]:
            ci, cj = jordan_wigner(i), jordan_wigner(j)
            expected = identity() if i == j else zero()
            assert normal_order(anticommutator(ci, hc(cj))) == expected
            assert normal_order(anticommutator(ci, cj)) == zero()

    def test_long_pauli_strings_square_to_one_and_commute_by_parity(self) -> None:
        sites = 400
        kinds = [(sigma_x, sigma_y, sigma_z)[i % 3] for i in range(sites)]

        def string(kinds: Sequence[Callable[[int], OpSum]]) -> OpSum:
            out = identity()
            for i, sigma in enumerate(kinds):
                out = out * sigma(i)
            return out

        A = string(kinds)
        one_differs = string([sigma_z, *kinds[1:]])
        two_differ = string([sigma_z, *kinds[1:-1], sigma_y])
        assert normal_order(A * A) == identity()
        assert normal_order(anticommutator(A, one_differs)) == zero()
        assert normal_order(commutator(A, two_differ)) == zero()


class TestMixedAlgebras:
    def test_different_algebras_commute(self) -> None:
        for f in (c(0), cdag(0)):
            for other in (b(0), bdag(0), sigma_x(0), sigma_z(0)):
                assert normal_order(commutator(f, other)) == zero()

    def test_fermion_anticommutation_with_boson_in_between(self) -> None:
        result = normal_order(c(0) * b(1) * cdag(0) + cdag(0) * b(1) * c(0))
        assert result == b(1)

    def test_jaynes_cummings_conserves_excitations(self) -> None:
        omega, delta, g = 1.0, 0.7, 0.3
        H = (
            omega * bdag(0) * b(0)
            + 0.5 * delta * sigma_z(0)
            + g * (sp(0) * b(0) + sm(0) * bdag(0))
        )
        N = bdag(0) * b(0) + sp(0) * sm(0)
        assert drop_zeros(normal_order(commutator(H, N))) == zero()
