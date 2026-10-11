"""End-to-end tests against analytic / cross-checked Hubbard spectra.

Includes RISB embedding problems from https://github.com/thenoursehorse/risb.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import scipy.linalg

from operator_algebra import c, cdag, hc, n, normal_order
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates, groundstate
from operator_algebra.solve.matrix import expectations, to_matrix
from operator_algebra.solve.spaces import FermionSpace


def hubbard_chain(sites: int, t: float, u: float):
    modes = [(s, i) for i in range(sites) for s in ("up", "dn")]
    H = 0 * cdag("up", 0) * c("up", 0)
    for i in range(sites - 1):
        for s in ("up", "dn"):
            H = H - t * (cdag(s, i) * c(s, i + 1) + cdag(s, i + 1) * c(s, i))
    for i in range(sites):
        H = H + u * n("up", i) * n("dn", i)
    return modes, H


# The RISB embedding tests below are ported from `tests/test_atomdiag.py` in
# https://github.com/thenoursehorse/risb, including the expected values.


def risb_one_band():
    u = 1
    mu = u / 2  # half-filling
    h_int = u * n("up", 0) * n("dn", 0)
    lambda_c = np.array([[-mu]])
    d = np.array([[-0.3333]])
    h0_loc = np.array([[0]])
    expected = {
        "e0": -0.9619378905494498,
        "rho_f": np.array([[0.5]]),
        "rho_cf": np.array([[0.4681588161332029]]),
        "rho_c": np.array([[0.5]]),
        "num_c": 1.0,
        "s2": 0.5066828353209953,
    }
    return h_int, lambda_c, d, h0_loc, expected


def risb_bilayer():
    u = 1
    v = 0.25
    j = 0
    mu = u / 2  # half-filling
    spins = ("up", "dn")
    h_int = 0 * n("up", 0)
    for o in (0, 1):
        h_int += u * n("up", o) * n("dn", o)
    for s1, s2 in product(spins, spins):
        h_int += 0.5 * j * cdag(s1, 0) * c(s2, 0) * cdag(s2, 1) * c(s1, 1)
    lambda_c = np.array([[-mu, -0.00460398], [-0.00460398, -mu]])
    d = np.array([[-2.59694448e-01, 0], [0, -2.59694448e-01]])
    h0_loc = np.array([[0, v], [v, 0]])
    expected = {
        "e0": -1.7429249197415944,
        "rho_f": np.array([[0.5, -0.1999913941210893], [-0.1999913941210893, 0.5]]),
        "rho_cf": np.array([[0.42326519677453511, 0], [0, 0.42326519677453511]]),
        "rho_c": np.array([[0.5, -0.1836332097072352], [-0.1836332097072352, 0.5]]),
        "num_c": 2.0,
        "s2": 0.8247577338845973,
    }
    return h_int, lambda_c, d, h0_loc, expected


def risb_dh_trimer():
    # two-thirds filling, Hubbard interaction in the momentum basis
    u = 1
    tk = 1
    n_orb = 3
    h_int = 0 * n("up", 0)
    for m, mm, mmm in product(range(n_orb), repeat=3):
        h_int += (
            (u / n_orb)
            * cdag("up", m)
            * c("up", mm)
            * cdag("dn", mmm)
            * c("dn", (m + mmm - mm) % n_orb)
        )
    lambda_c = np.diag([-1.91730088, -1.69005946, -1.69005946])
    d = np.diag([-0.26504931, -0.39631238, -0.39631238])
    h0_loc = np.diag([-2 * tk, tk, tk])
    expected = {
        "e0": -9.555511743344764,
        "rho_f": np.diag([0.9932309740187902, 0.5033842231804342, 0.5033842231804342]),
        "rho_cf": np.diag([0.0811187181751014, 0.4910360103357626, 0.4910360103357626]),
        "rho_c": np.diag([0.9909259681893234, 0.5045367260951683, 0.5045367260951683]),
        "num_c": 3.99999884075932,
        "s2": 0.9171025003755656,
    }
    return h_int, lambda_c, d, h0_loc, expected


def risb_h_emb(h_int, lambda_c, d, h0_loc):
    """Embedding Hamiltonian as built by risb's ``EmbeddingAtomDiag.set_h_emb``.

    c-electrons are orbitals ``0..n_orb-1`` and bath f-electrons are orbitals
    ``n_orb..2*n_orb-1``, with the same matrices for both spins. The bath term
    is ``f Lambda_c f^dag`` (annihilator first), as in risb.
    """
    n_orb = len(lambda_c)
    h = h_int
    for s in ("up", "dn"):
        for a, b in product(range(n_orb), repeat=2):
            f_a, f_b = n_orb + a, n_orb + b
            h += h0_loc[a, b] * cdag(s, a) * c(s, b)
            h += lambda_c[a, b] * c(s, f_a) * cdag(s, f_b)
            h += d[a, b] * (cdag(s, a) * c(s, f_b) + cdag(s, f_b) * c(s, a))
    return h


@pytest.mark.parametrize("model", [risb_one_band, risb_bilayer, risb_dh_trimer])
def test_risb_embedding_matches_risb(model):
    h_int, lambda_c, d, h0_loc, expected = model()
    n_orb = len(lambda_c)
    h_emb = risb_h_emb(h_int, lambda_c, d, h0_loc)
    modes = [(s, i) for i in range(2 * n_orb) for s in ("up", "dn")]
    c_modes = [(s, i) for i in range(n_orb) for s in ("up", "dn")]
    # risb solves in the half-filled sector of the embedding space
    basis = enumerate_basis(
        {FermionSpace(): modes}, sectors={total_n(modes): n_orb * 2}
    )
    e0, v0 = groundstate(to_matrix(h_emb, basis))
    orbs = range(n_orb)
    rho_f = np.array(
        [
            [
                expectations(
                    to_matrix(c("up", n_orb + b) * cdag("up", n_orb + a), basis), v0
                )
                for b in orbs
            ]
            for a in orbs
        ]
    )
    rho_cf = np.array(
        [
            [
                expectations(to_matrix(cdag("up", a) * c("up", n_orb + b), basis), v0)
                for b in orbs
            ]
            for a in orbs
        ]
    )
    rho_c = np.array(
        [
            [
                expectations(to_matrix(cdag("up", a) * c("up", b), basis), v0)
                for b in orbs
            ]
            for a in orbs
        ]
    )
    assert e0 == pytest.approx(expected["e0"], abs=1e-10)
    assert rho_f == pytest.approx(expected["rho_f"], abs=1e-10)
    assert rho_cf == pytest.approx(expected["rho_cf"], abs=1e-10)
    assert rho_c == pytest.approx(expected["rho_c"], abs=1e-10)
    num_c = expectations(to_matrix(total_n(c_modes), basis), v0)
    assert num_c == pytest.approx(expected["num_c"], abs=1e-10)
    s2 = expectations(to_matrix(total_s2(c_modes), basis), v0)
    assert s2 == pytest.approx(expected["s2"], abs=1e-10)


def total_n(modes):
    op = 0 * cdag(*modes[0]) * c(*modes[0])
    for mode in modes:
        op = op + n(*mode)
    return op


def total_sz(modes):
    op = 0 * cdag(*modes[0]) * c(*modes[0])
    for s, i in modes:
        op = op + (0.5 if s == "up" else -0.5) * n(s, i)
    return op


def total_s2(modes):
    sz = total_sz(modes)
    s_plus = 0 * cdag(*modes[0]) * c(*modes[0])
    for i in {i for _, i in modes}:
        s_plus = s_plus + cdag("up", i) * c("dn", i)
    s_minus = hc(s_plus)
    return sz * sz + 0.5 * (s_plus * s_minus + s_minus * s_plus)


def test_two_site_half_filling_matches_analytic():
    t, u = 1.0, 4.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis(
        {FermionSpace(): modes}, sectors={total_n(modes): 2, total_sz(modes): 0}
    )
    e0, _ = groundstate(to_matrix(H, basis))
    expected = (u - np.sqrt(u**2 + 16 * t**2)) / 2
    assert e0 == pytest.approx(expected, abs=1e-10)


def test_two_site_quarter_filling_is_pure_hopping():
    t, u = 1.3, 7.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis(
        {FermionSpace(): modes},
        sectors={total_n(modes): 1, total_sz(modes): 0.5},
    )
    assert basis.dim == 2
    e0, _ = groundstate(to_matrix(H, basis))
    assert e0 == pytest.approx(-t, abs=1e-10)


def test_three_site_half_filling_sparse_matches_dense():
    t, u = 1.0, 2.5
    modes, H = hubbard_chain(3, t, u)
    basis = enumerate_basis(
        {FermionSpace(): modes},
        sectors={total_n(modes): 3, total_sz(modes): 0.5},
    )
    mat = to_matrix(H, basis).toarray()
    mat = 0.5 * (mat + mat.conj().T)
    dense_evals = np.sort(scipy.linalg.eigvalsh(mat))

    energies, _ = eigenstates(to_matrix(H, basis), k=3)
    assert np.allclose(energies, dense_evals[:3], atol=1e-9)


def test_full_basis_groundstate_is_one_electron_sector():
    t, u = 1.0, 4.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis({FermionSpace(): modes})
    assert basis.dim == 16
    e0_full, _ = groundstate(to_matrix(H, basis))
    assert e0_full == pytest.approx(-t, abs=1e-10)


def test_normal_ordered_h_squared_matches_matrix_square():
    modes, H = hubbard_chain(2, t=1.0, u=4.0)
    basis = enumerate_basis({FermionSpace(): modes})
    h = to_matrix(H, basis).toarray()
    h2 = to_matrix(normal_order(H * H), basis).toarray()
    assert h2 == pytest.approx(h @ h)
