"""Time evolution."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse.linalg

from fockit.algebra import c, cdag, sigma_x, sigma_y, sigma_z
from fockit.hilbert import (
    FermionSpace,
    PauliSpace,
    build_basis,
    expectations,
    matrix_elements,
    state_index,
    to_matrix,
)
from fockit.solve import (
    chebyshev_evolve,
    eigh_full,
    expm_multiply_evolve,
    suzuki_trotter_sequence,
    thermal_expectation,
    trotter_evolve,
)


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
    basis = build_basis({FermionSpace(): labels})
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
    basis = build_basis({PauliSpace(): labels})
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(basis.dim) + 1j * rng.standard_normal(basis.dim)
    psi /= np.linalg.norm(psi)

    times = np.linspace(0.0, 1.5, 8)
    H_mat = to_matrix(H, basis)
    traj_em = expm_multiply_evolve(H_mat, psi, times)
    traj_ch = chebyshev_evolve(H_mat, psi, times)

    assert np.allclose(traj_em, traj_ch, atol=1e-9)


@pytest.mark.parametrize("evolver", [expm_multiply_evolve, chebyshev_evolve])
def test_evolver_unitary(evolver):
    labels, H = heisenberg_chain(3)
    basis = build_basis({PauliSpace(): labels})
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
    basis = build_basis({PauliSpace(): labels})
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
    basis = build_basis({PauliSpace(): labels})
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


def test_matrix_elements_reproduce_thermal_expectation():
    labels, H = heisenberg_chain(3)
    basis = build_basis({PauliSpace(): labels})
    E, V = eigh_full(to_matrix(H, basis))
    op_mat = to_matrix(sigma_z(0) * sigma_z(1), basis)
    beta = 0.8

    op_eb = matrix_elements(op_mat, V)
    p = np.exp(-beta * (E - E.min()))
    p /= p.sum()
    got = float((p * np.diag(op_eb).real).sum())

    ref = thermal_expectation(op_mat, E, V, beta).real
    assert got == pytest.approx(ref, abs=1e-12)
