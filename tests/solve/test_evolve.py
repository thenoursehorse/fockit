"""Real-time evolvers and finite-T dynamics primitives."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse.linalg

from fockit import c, cdag, sigma_x, sigma_y, sigma_z
from fockit.solve.basis import enumerate_basis, state_index
from fockit.solve.diagonalize import eigh_full
from fockit.solve.evolve import (
    boltzmann_weights,
    chebyshev_evolve,
    expm_multiply_evolve,
    krylov_evolve,
    lanczos_eigenpairs,
    suzuki_trotter_sequence,
    tpq_state,
    trotter_evolve,
)
from fockit.solve.matrix import expectations, matrix_elements, to_matrix
from fockit.solve.spaces import FermionSpace, PauliSpace
from fockit.solve.thermal import thermal_expectation


def free_hopping_chain(L: int, t: float = 1.0):
    H = -t * sum(cdag(i) * c(i + 1) + cdag(i + 1) * c(i) for i in range(L - 1))
    return list(range(L)), H


def heisenberg_chain(L: int):
    H = sum(
        0.25
        * (
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
        )
        for i in range(L - 1)
    )
    return list(range(L)), H


def test_free_fermion_single_particle_matches_analytic():
    L, t_hop = 4, 1.3
    labels, H = free_hopping_chain(L, t_hop)
    basis = enumerate_basis({FermionSpace(): labels})
    h = -t_hop * (np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1))

    psi0 = np.zeros(basis.dim, dtype=complex)
    psi0[state_index(basis, (1, 0, 0, 0))] = 1.0  # particle on site 0

    times = np.linspace(0.0, 2.0, 5)
    traj = expm_multiply_evolve(to_matrix(H, basis), psi0, times)

    evals, evecs = np.linalg.eigh(h)
    for k, t in enumerate(times):
        amp = (evecs * np.exp(-1j * t * evals)) @ evecs[0].conj()
        ref = np.abs(amp) ** 2
        for j in range(L):
            obs = expectations(to_matrix(cdag(j) * c(j), basis), traj[k : k + 1])[
                0
            ].real
            assert obs == pytest.approx(ref[j], abs=1e-10)


def test_evolvers_agree_on_heisenberg():
    labels, H = heisenberg_chain(4)
    basis = enumerate_basis({PauliSpace(): labels})
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    psi /= np.linalg.norm(psi)

    times = np.linspace(0.0, 1.5, 8)
    H_mat = to_matrix(H, basis)
    traj_em = expm_multiply_evolve(H_mat, psi, times)
    traj_kr = krylov_evolve(H_mat, psi, times, n_krylov=20)
    traj_ch = chebyshev_evolve(H_mat, psi, times)

    assert np.allclose(traj_em, traj_kr, atol=1e-9)
    assert np.allclose(traj_em, traj_ch, atol=1e-9)


@pytest.mark.parametrize(
    "evolver", [expm_multiply_evolve, krylov_evolve, chebyshev_evolve]
)
def test_evolver_unitary(evolver):
    labels, H = heisenberg_chain(3)
    basis = enumerate_basis({PauliSpace(): labels})
    rng = np.random.default_rng(1)
    psi = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    psi /= np.linalg.norm(psi)
    times = np.linspace(0.0, 2.0, 6)
    traj = evolver(to_matrix(H, basis), psi, times)
    norms = np.linalg.norm(traj, axis=1)
    assert norms == pytest.approx(np.ones_like(norms), abs=1e-9)


def heisenberg_fragments(L: int):
    def bond(i):
        return 0.25 * (
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
        )

    even = sum(bond(i) for i in range(0, L - 1, 2))
    odd = sum(bond(i) for i in range(1, L - 1, 2))
    return [even, odd]


@pytest.mark.parametrize("order", [1, 2, 4])
def test_suzuki_trotter_sequence_sums_to_dt(order):
    dt = 0.37 - 0.2j
    seq = suzuki_trotter_sequence(5, dt, order=order)
    totals: dict[int, complex] = {}
    for coeff, i in seq:
        totals[i] = totals.get(i, 0j) + coeff
    for i in range(5):
        assert totals[i] == pytest.approx(dt, abs=1e-12)


def test_suzuki_trotter_sequence_merges_adjacent():
    seq = suzuki_trotter_sequence(3, 1.0, order=2)
    for (_, i), (_, j) in pairwise(seq):
        assert i != j


@pytest.mark.parametrize(("order", "tol"), [(2, 5e-3), (4, 5e-6)])
def test_trotter_evolve_matches_exact(order, tol):
    labels, H = heisenberg_chain(4)
    frags = heisenberg_fragments(4)
    basis = enumerate_basis({PauliSpace(): labels})
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    psi /= np.linalg.norm(psi)

    n_steps, dt = 20, 0.05
    schedule = suzuki_trotter_sequence(len(frags), -1j * dt, order=order)
    frag_mats = [to_matrix(f, basis) for f in frags]
    traj = trotter_evolve(frag_mats, psi, schedule, n_steps)

    times = np.arange(n_steps + 1) * dt
    ref = expm_multiply_evolve(to_matrix(H, basis), psi, times)
    assert np.allclose(traj, ref, atol=tol)


def test_trotter_evolve_imaginary_time_matches_tpq():
    labels, H = heisenberg_chain(3)
    frags = heisenberg_fragments(3)
    basis = enumerate_basis({PauliSpace(): labels})
    H_mat = to_matrix(H, basis)

    rng = np.random.default_rng(0)
    r = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    r /= np.linalg.norm(r)

    beta = 0.4
    n_steps = 200
    dtau = beta / (2 * n_steps)
    schedule = suzuki_trotter_sequence(len(frags), -dtau, order=4)
    traj = trotter_evolve([to_matrix(f, basis) for f in frags], r, schedule, n_steps)

    ref = scipy.sparse.linalg.expm_multiply(-0.5 * beta * H_mat, r)
    assert np.allclose(traj[-1], ref, atol=1e-6)


def test_matrix_elements_and_boltzmann_reproduce_thermal_expectation():
    labels, H = heisenberg_chain(3)
    basis = enumerate_basis({PauliSpace(): labels})
    E, V = eigh_full(to_matrix(H, basis))
    op_mat = to_matrix(sigma_z(0) * sigma_z(1), basis)
    beta = 0.8

    op_eb = matrix_elements(op_mat, V)
    p = boltzmann_weights(E, beta)
    got = float((p * np.diag(op_eb).real).sum())

    ref = thermal_expectation(op_mat, E, V, beta).real
    assert got == pytest.approx(ref, abs=1e-12)


def test_tpq_state_averages_to_thermal_expectation():
    labels, H = heisenberg_chain(2)
    basis = enumerate_basis({PauliSpace(): labels})
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
    got = (num / denom).real

    ref = thermal_expectation(op_mat, E, V, beta).real
    assert got == pytest.approx(ref, abs=5e-2)


def test_lanczos_eigenpairs_compose_to_ftlm():
    labels, H = heisenberg_chain(3)
    basis = enumerate_basis({PauliSpace(): labels})
    H_mat = to_matrix(H, basis)
    E, V = eigh_full(H_mat)
    op_mat = to_matrix(sigma_z(0) * sigma_z(1), basis)
    beta = 0.6

    rng = np.random.default_rng(0)
    num, denom = 0j, 0.0
    for _ in range(200):
        eps, psi, r = lanczos_eigenpairs(H_mat, n_lanczos=basis.dim, rng=rng)
        c_k = psi.conj().T @ r
        op_kr = psi.conj().T @ (op_mat @ r)
        w = np.exp(-beta * (eps - eps.min()))
        num += complex(np.sum(w * c_k.conj() * op_kr))
        denom += float(np.sum(w * np.abs(c_k) ** 2).real)
    got = (num / denom).real

    ref = thermal_expectation(op_mat, E, V, beta).real
    assert got == pytest.approx(ref, abs=5e-2)


def test_krylov_warns_when_step_is_too_long_for_its_krylov_space():
    labels, H = heisenberg_chain(8)
    basis = enumerate_basis({PauliSpace(): labels})
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    psi /= np.linalg.norm(psi)
    with pytest.warns(RuntimeWarning, match="not converged"):
        krylov_evolve(to_matrix(H, basis), psi, [0.0, 5.0], n_krylov=5)
