"""Finite-temperature expectation values with ED references."""

from __future__ import annotations

import numpy as np
import pytest

from fockit.algebra import b, bdag, c, cdag, nc, sigma_x, sigma_y, sigma_z
from fockit.hilbert import (
    BosonSpace,
    FermionSpace,
    PauliSpace,
    build_basis,
    to_matrix,
)
from fockit.solve import (
    eigh_full,
    thermal_expectation,
    tpq_state,
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
    ) + u * sum(nc(i, "up") * nc(i, "dn") for i in range(2))
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
    basis = build_basis({PauliSpace(): [0]})
    E, V = eigh_full(to_matrix(H, basis))
    for beta in (0.01, 0.5, 2.0, 50.0):
        expected = -np.tanh(0.5 * beta * delta)
        got = thermal_expectation(to_matrix(sigma_z(0), basis), E, V, beta).real
        assert got == pytest.approx(expected, abs=1e-10)


def test_free_fermions_match_fermi_dirac():
    t = 1.3
    H = -t * (cdag(0) * c(1) + cdag(1) * c(0))
    basis = build_basis({FermionSpace(): [0, 1]})
    E, V = eigh_full(to_matrix(H, basis))
    beta = 0.7
    fd_plus = 1.0 / (1.0 + np.exp(+beta * t))
    fd_minus = 1.0 / (1.0 + np.exp(-beta * t))
    expected = 0.5 * (fd_plus + fd_minus)
    got = thermal_expectation(to_matrix(nc(0), basis), E, V, beta).real
    assert got == pytest.approx(expected, abs=1e-10)


def test_harmonic_oscillator_matches_bose_einstein():
    basis = build_basis({BosonSpace(n_max=20): [0]})
    H = bdag(0) * b(0)
    E, V = eigh_full(to_matrix(H, basis))
    beta = 1.0
    expected = 1.0 / (np.expm1(beta))
    got = thermal_expectation(to_matrix(bdag(0) * b(0), basis), E, V, beta).real
    assert got == pytest.approx(expected, abs=1e-7)


MODELS = [
    (
        hubbard_2site,
        FermionSpace(),
        HUBBARD_BETA,
        HUBBARD_E,
        lambda: nc(0, "up") * nc(0, "dn"),
        HUBBARD_DOUBLE_OCC,
    ),
    (
        bose_hubbard_2site,
        BosonSpace(n_max=3),
        BOSE_HUBBARD_BETA,
        BOSE_HUBBARD_E,
        lambda: bdag(0) * b(0),
        BOSE_HUBBARD_N0,
    ),
    (
        heisenberg_4site,
        PauliSpace(),
        HEIS_BETA,
        HEIS_E,
        lambda: sigma_z(0) * sigma_z(1),
        HEIS_ZZ,
    ),
]

MODEL_IDS = ["hubbard", "bose_hubbard", "heisenberg"]


@pytest.mark.parametrize(
    ("model", "space", "beta", "ref_e", "obs_factory", "ref_obs"),
    MODELS,
    ids=MODEL_IDS,
)
def test_tpq_matches_ed_reference(model, space, beta, ref_e, obs_factory, ref_obs):
    labels, H = model()
    basis = build_basis({space: labels})
    H_mat = to_matrix(H, basis)
    got_e = tpq_thermal_expectation(H_mat, H_mat, beta, n_samples=200, seed=0).real
    assert got_e == pytest.approx(ref_e, abs=5e-2)
    obs = to_matrix(obs_factory(), basis)
    got_obs = tpq_thermal_expectation(obs, H_mat, beta, n_samples=200, seed=0).real
    assert got_obs == pytest.approx(ref_obs, abs=5e-2)


def test_tpq_states_average_to_the_thermal_expectation():
    labels, H = heisenberg_4site()
    basis = build_basis({PauliSpace(): labels})
    H_mat = to_matrix(H, basis)
    E, V = eigh_full(H_mat)
    op_mat = to_matrix(sigma_z(0) * sigma_z(1), basis)
    beta = 0.7

    rng = np.random.default_rng(0)
    num, denom = 0j, 0.0
    for _ in range(400):
        phi = tpq_state(H_mat, beta, rng=rng, e0=E[0])
        num += complex(phi.conj() @ (op_mat @ phi))
        denom += float((phi.conj() @ phi).real)

    ref = thermal_expectation(op_mat, E, V, beta).real
    assert (num / denom).real == pytest.approx(ref, abs=5e-2)
