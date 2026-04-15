"""Finite-temperature expectation values: analytic + hardcoded ED references."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import b, bdag, c, cdag, n, sigma_x, sigma_y, sigma_z
from operator_algebra.algebra import BOSON, FERMION, PAULI
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigh_full
from operator_algebra.solve.thermal import (
    kpm_thermal_expectation,
    lanczos_thermal_expectation,
    thermal_expectation,
    tpq_thermal_expectation,
)

HUBBARD_BETA = 1.0
HUBBARD_E = -0.4854088814823613
HUBBARD_DOUBLE_OCC = 0.018814277735803006
HUBBARD_N0 = 0.6802583075436314

BOSE_HUBBARD_BETA = 1.0
BOSE_HUBBARD_E = -1.0831275685097335
BOSE_HUBBARD_N0 = 1.1971085627436382

HEIS_BETA = 1.0
HEIS_E = -0.6330214400416613
HEIS_ZZ = -0.28635875583809756


def hubbard_2site():
    t, u = 1.0, 4.0
    modes = [(i, s) for i in range(2) for s in ("up", "dn")]
    H = -t * sum(
        cdag(i, s) * c(i + 1, s) + cdag(i + 1, s) * c(i, s)
        for i in range(1)
        for s in ("up", "dn")
    ) + u * sum(n(i, "up") * n(i, "dn") for i in range(2))
    return modes, H


def bose_hubbard_2site():
    t, u = 1.0, 1.0
    H = -t * (bdag(0) * b(1) + bdag(1) * b(0)) + 0.5 * u * sum(
        bdag(i) * bdag(i) * b(i) * b(i) for i in range(2)
    )
    return [0, 1], H


def heisenberg_4site():
    H = sum(
        0.25
        * (
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
        )
        for i in range(3)
    )
    return [0, 1, 2, 3], H


def test_two_level_matches_tanh():
    delta = 1.7
    H = 0.5 * delta * sigma_z(0)
    basis = enumerate_basis([0], PAULI)
    E, V = eigh_full(H, basis)
    for beta in (0.01, 0.5, 2.0, 50.0):
        expected = -np.tanh(0.5 * beta * delta)
        got = thermal_expectation(sigma_z(0), E, V, basis, beta).real
        assert got == pytest.approx(expected, abs=1e-10)


def test_free_fermions_match_fermi_dirac():
    t = 1.3
    H = -t * (cdag(0) * c(1) + cdag(1) * c(0))
    basis = enumerate_basis([0, 1], FERMION)
    E, V = eigh_full(H, basis)
    beta = 0.7
    fd_plus = 1.0 / (1.0 + np.exp(+beta * t))
    fd_minus = 1.0 / (1.0 + np.exp(-beta * t))
    expected = 0.5 * (fd_plus + fd_minus)
    got = thermal_expectation(n(0), E, V, basis, beta).real
    assert got == pytest.approx(expected, abs=1e-10)


def test_harmonic_oscillator_matches_bose_einstein():
    basis = enumerate_basis([0], BOSON, n_max=20)
    H = bdag(0) * b(0)
    E, V = eigh_full(H, basis)
    beta = 1.0
    expected = 1.0 / (np.expm1(beta))
    got = thermal_expectation(bdag(0) * b(0), E, V, basis, beta).real
    assert got == pytest.approx(expected, abs=1e-7)


MODELS = [
    (
        hubbard_2site,
        FERMION,
        {},
        HUBBARD_BETA,
        HUBBARD_E,
        lambda: n(0, "up") * n(0, "dn"),
        HUBBARD_DOUBLE_OCC,
    ),
    (
        bose_hubbard_2site,
        BOSON,
        {"n_max": 3},
        BOSE_HUBBARD_BETA,
        BOSE_HUBBARD_E,
        lambda: bdag(0) * b(0),
        BOSE_HUBBARD_N0,
    ),
    (
        heisenberg_4site,
        PAULI,
        {},
        HEIS_BETA,
        HEIS_E,
        lambda: sigma_z(0) * sigma_z(1),
        HEIS_ZZ,
    ),
]

MODEL_IDS = ["hubbard", "bose_hubbard", "heisenberg"]


@pytest.mark.parametrize(
    ("model", "stat", "basis_kwargs", "beta", "ref_e", "obs_factory", "ref_obs"),
    MODELS,
    ids=MODEL_IDS,
)
@pytest.mark.parametrize("method", ["ftlm", "ltlm"])
def test_lanczos_matches_ed_reference(
    model, stat, basis_kwargs, beta, ref_e, obs_factory, ref_obs, method
):
    labels, H = model()
    basis = enumerate_basis(labels, stat, **basis_kwargs)
    got_e = lanczos_thermal_expectation(
        H, H, basis, beta, n_samples=200, n_lanczos=basis.dim, method=method, seed=0
    ).real
    assert got_e == pytest.approx(ref_e, abs=4e-2)
    got_obs = lanczos_thermal_expectation(
        obs_factory(),
        H,
        basis,
        beta,
        n_samples=200,
        n_lanczos=basis.dim,
        method=method,
        seed=0,
    ).real
    assert got_obs == pytest.approx(ref_obs, abs=4e-2)


@pytest.mark.parametrize(
    ("model", "stat", "basis_kwargs", "beta", "ref_e", "obs_factory", "ref_obs"),
    MODELS,
    ids=MODEL_IDS,
)
def test_kpm_matches_ed_reference(
    model, stat, basis_kwargs, beta, ref_e, obs_factory, ref_obs
):
    labels, H = model()
    basis = enumerate_basis(labels, stat, **basis_kwargs)
    got_e = kpm_thermal_expectation(
        H, H, basis, beta, n_moments=100, n_samples=1000, seed=0
    ).real
    assert got_e == pytest.approx(ref_e, abs=5e-2)
    got_obs = kpm_thermal_expectation(
        obs_factory(), H, basis, beta, n_moments=100, n_samples=1000, seed=0
    ).real
    assert got_obs == pytest.approx(ref_obs, abs=5e-2)


@pytest.mark.parametrize(
    ("model", "stat", "basis_kwargs", "beta", "ref_e", "obs_factory", "ref_obs"),
    MODELS,
    ids=MODEL_IDS,
)
def test_tpq_matches_ed_reference(
    model, stat, basis_kwargs, beta, ref_e, obs_factory, ref_obs
):
    labels, H = model()
    basis = enumerate_basis(labels, stat, **basis_kwargs)
    got_e = tpq_thermal_expectation(H, H, basis, beta, n_samples=200, seed=0).real
    assert got_e == pytest.approx(ref_e, abs=5e-2)
    got_obs = tpq_thermal_expectation(
        obs_factory(), H, basis, beta, n_samples=200, seed=0
    ).real
    assert got_obs == pytest.approx(ref_obs, abs=5e-2)
