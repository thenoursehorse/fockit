"""Steady states, decay rates and evolution of Lindblad generators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fockit.algebra import (
    OpSum,
    c,
    cdag,
    left,
    liouvillian,
    nc,
    right,
    sigma_minus,
    sigma_plus,
    sigma_x,
    sigma_z,
)
from fockit.hilbert import (
    Basis,
    FermionSpace,
    PauliSpace,
    build_basis,
    density_matrix,
    doubled,
    expand_state,
    to_matrix,
    trace_vector,
)
from fockit.solve import expm_multiply_evolve, groundstate, slowest_modes, steady_state

SITES, LOSS, PUMP = 6, 0.6, 0.3
FILLING = PUMP / (LOSS + PUMP)


def lossy_ring() -> OpSum:
    # uniform loss and pump; H keeps N, so N relaxes at LOSS + PUMP for any H
    H: OpSum | complex = -sum(
        cdag((i + 1) % SITES) * c(i) + cdag(i) * c((i + 1) % SITES)
        for i in range(SITES)
    )
    H += 0.7 * sum(nc(i) * nc((i + 1) % SITES) for i in range(SITES))
    jumps = [LOSS**0.5 * c(i) for i in range(SITES)]
    jumps += [PUMP**0.5 * cdag(i) for i in range(SITES)]
    return liouvillian(H, jumps)


def number() -> OpSum | complex:
    return sum(nc(i) for i in range(SITES))


def sector(**symmetries: Any) -> Basis:
    N = number()
    return build_basis(
        doubled({FermionSpace(): range(SITES)}),
        diagonal_symmetries={left(N) - right(N): 0},
        **symmetries,
    )


def test_steady_state_of_loss_and_pump_is_the_product_state() -> None:
    shift = {i: (i + 1) % SITES for i in range(SITES)}
    basis = sector(permutation_symmetries=[(shift, 1)])
    rho = steady_state(to_matrix(lossy_ring(), basis), trace_vector(basis))

    plain = sector()
    closed = build_basis({FermionSpace(): range(SITES)})
    matrix = density_matrix(expand_state(rho, basis, plain), plain, closed)
    particles = closed.levels.sum(axis=1)
    product = np.diag(FILLING**particles * (1 - FILLING) ** (SITES - particles))
    assert matrix == pytest.approx(product, abs=1e-10)


def test_driven_spin_coherence_agrees_with_its_density_matrix() -> None:
    # driving and decay leave a complex coherence <sigma+> in the steady state
    H = 0.8 * sigma_x(0) + 0.3 * sigma_z(0)
    basis = build_basis(doubled({PauliSpace(): [0]}))
    generator = to_matrix(liouvillian(H, [0.5 * sigma_minus(0)]), basis)
    trace = trace_vector(basis)
    rho = steady_state(generator, trace)
    closed = build_basis({PauliSpace(): [0]})
    matrix = density_matrix(rho, basis, closed)

    coherence = trace @ (to_matrix(left(sigma_plus(0)), basis) @ rho)
    assert abs(coherence.imag) > 0.01
    assert np.trace(to_matrix(sigma_plus(0), closed) @ matrix) == pytest.approx(
        coherence
    )
    assert matrix == pytest.approx(matrix.conj().T)
    assert np.trace(matrix) == pytest.approx(1)
    assert np.linalg.eigvalsh(matrix).min() > -1e-12


def test_particle_number_relaxes_at_the_sum_of_the_rates() -> None:
    basis = sector()
    generator = to_matrix(lossy_ring(), basis)
    trace = trace_vector(basis)
    N = to_matrix(left(number()), basis)
    # start from the empty state
    empty = np.zeros(basis.dim, dtype=complex)
    empty[np.all(basis.levels == 0, axis=1)] = 1
    times = np.linspace(0, 3, 7)
    states = expm_multiply_evolve(1j * generator, empty, times)
    assert states @ trace == pytest.approx(np.ones(len(times)))
    exact = SITES * FILLING * (1 - np.exp(-(LOSS + PUMP) * times))
    assert (N @ states.T).T @ trace == pytest.approx(exact)


def test_single_mode_with_loss_has_the_known_decay_rates() -> None:
    omega, gamma = 1.3, 0.4
    generator = liouvillian(omega * nc(0), [np.sqrt(gamma) * c(0)])
    basis = build_basis(doubled({FermionSpace(): [0]}))
    values, _ = slowest_modes(to_matrix(generator, basis), k=4)
    # steady state, the two coherences, then the population
    assert values[[0, 3]] == pytest.approx([0, -gamma])
    assert values[1:3].real == pytest.approx([-gamma / 2, -gamma / 2])
    assert sorted(values[1:3].imag) == pytest.approx([-omega, omega])


def test_hamiltonian_solvers_reject_a_lindblad_generator() -> None:
    basis = sector()
    with pytest.raises(ValueError, match="not Hermitian"):
        groundstate(to_matrix(lossy_ring(), basis))
