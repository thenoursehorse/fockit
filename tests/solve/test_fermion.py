"""End-to-end tests against analytic / cross-checked Hubbard spectra."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import scipy.linalg

from operator_algebra import c, cdag, n
from operator_algebra.algebra import FERMION
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates, groundstate
from operator_algebra.solve.matrix import expectations, to_matrix


def hubbard_chain(sites: int, t: float, u: float):
    modes = [(s, i) for i in range(sites) for s in ("up", "dn")]
    H = 0 * cdag("up", 0) * c("up", 0)
    for i in range(sites - 1):
        for s in ("up", "dn"):
            H = H - t * (cdag(s, i) * c(s, i + 1) + cdag(s, i + 1) * c(s, i))
    for i in range(sites):
        H = H + u * n("up", i) * n("dn", i)
    return modes, H


def risb_bilayer_impurity():
    U = 1
    V = 0.25
    J = 0
    bath_offdiag = -0.00460398
    hybr_diag = -2.59694448e-01
    mu = U / 2.0  # half-filling
    imp_orbs = (0, 1)
    spin_names = ("up", "dn")
    h_int = 0 * n("up", 0)
    for o in imp_orbs:
        h_int += U * n("up", o) * n("dn", o)
    for s1, s2 in product(spin_names, spin_names):
        h_int += 0.5 * J * cdag(s1, 0) * c(s2, 0) * cdag(s2, 1) * c(s1, 1)
    h0 = 0 * n("up", 0)
    for spin in spin_names:
        h0 += V * cdag(spin, 0) * c(spin, 1)
        h0 += V * cdag(spin, 1) * c(spin, 0)
        # bath is f Lambda f^dag (annihilator first), as in risb
        h0 -= mu * c(spin, 2) * cdag(spin, 2)
        h0 -= mu * c(spin, 3) * cdag(spin, 3)
        h0 += bath_offdiag * c(spin, 2) * cdag(spin, 3)
        h0 += bath_offdiag * c(spin, 3) * cdag(spin, 2)
        h0 += hybr_diag * cdag(spin, 0) * c(spin, 2)
        h0 += hybr_diag * cdag(spin, 2) * c(spin, 0)
        h0 += hybr_diag * cdag(spin, 1) * c(spin, 3)
        h0 += hybr_diag * cdag(spin, 3) * c(spin, 1)
    modes = [(s, i) for i in range(2 * len(imp_orbs)) for s in spin_names]
    return modes, h0 + h_int


def test_risb_bilayer_impurity():
    modes, h_emb = risb_bilayer_impurity()
    num_op = total_n(modes)
    sz_op = total_sz(modes)
    num_elec_emb = int(len(modes) / 2)
    sz = 0
    basis = enumerate_basis(modes, FERMION, sectors={num_op: num_elec_emb, sz_op: sz})
    e0, v0 = groundstate(h_emb, basis)
    n0 = expectations(num_op, v0, basis)
    sz0 = expectations(sz_op, v0, basis)
    # bath f-electrons are orbitals 2, 3
    rho_f = np.array(
        [
            [
                expectations(c("up", 2 + b) * cdag("up", 2 + a), v0, basis)
                for b in (0, 1)
            ]
            for a in (0, 1)
        ]
    )
    rho_cf = np.array(
        [
            [expectations(cdag("up", a) * c("up", 2 + b), v0, basis) for b in (0, 1)]
            for a in (0, 1)
        ]
    )
    rho_c = np.array(
        [
            [expectations(cdag("up", a) * c("up", b), v0, basis) for b in (0, 1)]
            for a in (0, 1)
        ]
    )
    rho_f_expected = np.array([[0.5, -0.1999913941210893], [-0.1999913941210893, 0.5]])
    rho_cf_expected = np.array([[0.42326519677453511, 0], [0, 0.42326519677453511]])
    rho_c_expected = np.array([[0.5, -0.1836332097072352], [-0.1836332097072352, 0.5]])
    e0_expected = -1.7429249197415944
    assert n0 == pytest.approx(num_elec_emb)
    assert sz0 == pytest.approx(sz)
    assert e0 == pytest.approx(e0_expected, abs=1e-10)
    assert rho_f == pytest.approx(rho_f_expected, abs=1e-10)
    assert rho_cf == pytest.approx(rho_cf_expected, abs=1e-10)
    assert rho_c == pytest.approx(rho_c_expected, abs=1e-10)


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


def test_two_site_half_filling_matches_analytic():
    t, u = 1.0, 4.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis(
        modes, FERMION, sectors={total_n(modes): 2, total_sz(modes): 0}
    )
    e0, _ = groundstate(H, basis)
    expected = (u - np.sqrt(u**2 + 16 * t**2)) / 2
    assert e0 == pytest.approx(expected, abs=1e-10)


def test_two_site_quarter_filling_is_pure_hopping():
    t, u = 1.3, 7.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis(
        modes,
        FERMION,
        sectors={total_n(modes): 1, total_sz(modes): 0.5},
    )
    assert basis.dim == 2
    e0, _ = groundstate(H, basis)
    assert e0 == pytest.approx(-t, abs=1e-10)


def test_three_site_half_filling_sparse_matches_dense():
    t, u = 1.0, 2.5
    modes, H = hubbard_chain(3, t, u)
    basis = enumerate_basis(
        modes,
        FERMION,
        sectors={total_n(modes): 3, total_sz(modes): 0.5},
    )
    mat = to_matrix(H, basis).toarray()
    mat = 0.5 * (mat + mat.conj().T)
    dense_evals = np.sort(scipy.linalg.eigvalsh(mat))

    energies, _ = eigenstates(H, basis, k=3)
    assert np.allclose(energies, dense_evals[:3], atol=1e-9)


def test_full_basis_groundstate_is_one_electron_sector():
    t, u = 1.0, 4.0
    modes, H = hubbard_chain(2, t, u)
    basis = enumerate_basis(modes, FERMION)
    assert basis.dim == 16
    e0_full, _ = groundstate(H, basis)
    assert e0_full == pytest.approx(-t, abs=1e-10)
