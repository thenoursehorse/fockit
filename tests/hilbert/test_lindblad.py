"""Lindblad master equation as a matrix on doubled modes, with symmetry sectors."""

from __future__ import annotations

import cmath
import runpy
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from fockit.algebra import (
    OpSum,
    b,
    bdag,
    c,
    cdag,
    hc,
    left,
    liouvillian,
    nc,
    normal_order,
    right,
    sigma_minus,
    sigma_plus,
    sigma_x,
    sigma_y,
    sigma_z,
)
from fockit.hilbert import (
    BosonSpace,
    FermionSpace,
    LocalSpace,
    PauliSpace,
    build_basis,
    doubled,
    to_matrix,
)

type ComplexArray = NDArray[np.complex128]

EXAMPLES = Path(__file__).parents[2] / "examples"


def master_equation[S: LocalSpace](
    H: OpSum, jumps: Sequence[OpSum], spaces: Mapping[S, Sequence[Hashable]]
) -> ComplexArray:
    """Dense generator of ``vec(rho)``, row-major, from the closed-system matrices."""
    basis = build_basis(spaces)
    h = to_matrix(H, basis).toarray()
    one = np.eye(basis.dim)
    out = -1j * (np.kron(h, one) - np.kron(one, h.T))
    for jump in jumps:
        m = to_matrix(jump, basis).toarray()
        decay = m.conj().T @ m
        out += np.kron(m, m.conj()) - 0.5 * (
            np.kron(decay, one) + np.kron(one, decay.T)
        )
    return out


def sector_spectrum[S: LocalSpace](
    generator: OpSum,
    spaces: Mapping[S, Sequence[Hashable]],
    sectors: Iterable[Mapping[str, Any]],
) -> ComplexArray:
    parts = [np.empty(0, dtype=complex)]
    for kwargs in sectors:
        basis = build_basis(doubled(spaces), **kwargs)
        if basis.dim:
            parts.append(scipy.linalg.eigvals(to_matrix(generator, basis).toarray()))
    return np.concatenate(parts)


def assert_same_spectrum(found: ComplexArray, expected: ComplexArray) -> None:
    assert len(found) == len(expected)
    rows, cols = linear_sum_assignment(np.abs(found[:, None] - expected[None, :]))
    assert np.abs(found[rows] - expected[cols]).max() < 1e-8


def interacting_ring(sites: int) -> OpSum:
    hop = sum(-cmath.exp(0.3j) * cdag(i) * c((i + 1) % sites) for i in range(sites))
    interaction = sum(nc(i) * nc((i + 1) % sites) for i in range(sites))
    return hop + hc(hop) + 0.7 * interaction


def shift(sites: int) -> dict[int, int]:
    return {i: (i + 1) % sites for i in range(sites)}


def momentum(k: int, sites: int) -> complex:
    return cmath.exp(2j * cmath.pi * k / sites)


def test_liouvillian_is_the_master_equation_for_fermions_with_loss_and_pump() -> None:
    sites = 4
    jumps = [0.6 * c(i) for i in range(sites)] + [0.4 * cdag(i) for i in range(sites)]
    spaces = {FermionSpace(): range(sites)}
    generator = to_matrix(
        liouvillian(interacting_ring(sites), jumps), build_basis(doubled(spaces))
    )
    assert generator.toarray() == pytest.approx(
        master_equation(interacting_ring(sites), jumps, spaces)
    )


def test_liouvillian_is_the_master_equation_for_mixed_algebras() -> None:
    hop = -cmath.exp(0.4j) * cdag(0) * c(1)
    H = hop + hc(hop) + bdag(0) * b(0) + 0.5 * nc(0) * (b(0) + bdag(0))
    H += 0.3 * nc(1) * sigma_y(0) + 0.2 * sigma_x(0)
    jumps = [
        0.5 * c(0),
        0.3 * (b(0) + 0.5j * cdag(1) * c(0)),
        0.4 * sigma_minus(0),
        0.2 * bdag(0),
    ]
    spaces: dict[LocalSpace, Sequence[Hashable]] = {
        FermionSpace(): [0, 1],
        BosonSpace(n_max=2): [0],
        PauliSpace(): [0],
    }
    generator = to_matrix(liouvillian(H, jumps), build_basis(doubled(spaces)))
    assert generator.toarray() == pytest.approx(master_equation(H, jumps, spaces))


def test_liouvillian_is_the_master_equation_for_a_custom_algebra() -> None:
    ex = runpy.run_path(str(EXAMPLES / "anyons.py"))
    a, adag, AnyonSpace = ex["a"], ex["adag"], ex["AnyonSpace"]
    theta, sites = np.pi / 3, 3
    H = -sum(
        adag(j + 1, theta) * a(j, theta) + adag(j, theta) * a(j + 1, theta)
        for j in range(sites - 1)
    )
    jumps = [0.4 * a(j, theta) for j in range(sites)]
    spaces = {AnyonSpace(theta, n_max=2): range(sites)}
    generator = to_matrix(liouvillian(H, jumps), build_basis(doubled(spaces)))
    assert generator.toarray() == pytest.approx(master_equation(H, jumps, spaces))


def test_normal_ordering_keeps_the_liouvillian_for_fermions_and_spins() -> None:
    hop = -cmath.exp(0.4j) * cdag(0) * c(1)
    H = hop + hc(hop) + 0.3 * nc(1) * sigma_y(0) + 0.2 * sigma_x(1) * sigma_z(0)
    jumps = [
        0.5 * c(0),
        0.4 * (sigma_x(0) + 0.5j * sigma_y(0)),
        0.3 * nc(1) * sigma_y(1),
    ]
    basis = build_basis(doubled({FermionSpace(): [0, 1], PauliSpace(): [0, 1]}))
    generator = liouvillian(H, jumps)
    assert to_matrix(normal_order(generator), basis).toarray() == pytest.approx(
        to_matrix(generator, basis).toarray()
    )


def test_pumped_truncated_boson_conserves_the_trace() -> None:
    # dTr(rho)/dt = <<1| L |rho>> must vanish for every rho, even at n_max
    H = bdag(0) * b(0) + 0.3 * (b(0) + bdag(0))
    basis = build_basis(doubled({BosonSpace(n_max=3): [0]}))
    identity = (basis.levels[:, 0] == basis.levels[:, 1]).astype(float)
    generator = to_matrix(liouvillian(H, [0.5 * bdag(0), 0.7 * b(0)]), basis)
    assert identity @ generator.toarray() == pytest.approx(np.zeros(basis.dim))


def test_weak_number_and_momentum_sectors_rebuild_the_spectrum() -> None:
    # loss and pump change N, but keep N_left - N_right and translation
    sites = 4
    jumps = [0.6 * c(i) for i in range(sites)] + [0.4 * cdag(i) for i in range(sites)]
    spaces = {FermionSpace(): range(sites)}
    N = sum(nc(i) for i in range(sites))
    sectors = [
        {
            "diagonal_symmetries": {left(N) - right(N): q},
            "permutation_symmetries": [(shift(sites), momentum(k, sites))],
        }
        for q in range(-sites, sites + 1)
        for k in range(sites)
    ]
    found = sector_spectrum(
        liouvillian(interacting_ring(sites), jumps), spaces, sectors
    )
    expected = scipy.linalg.eigvals(
        master_equation(interacting_ring(sites), jumps, spaces)
    )
    assert_same_spectrum(found, expected)


def test_dephasing_keeps_one_steady_state_per_particle_number() -> None:
    # dephasing commutes with N, so N_left and N_right are each conserved
    sites = 4
    jumps = [0.5 * nc(i) for i in range(sites)]
    spaces = {FermionSpace(): range(sites)}
    N = sum(nc(i) for i in range(sites))
    sectors = [
        {"diagonal_symmetries": {left(N): n, right(N): m}}
        for n in range(sites + 1)
        for m in range(sites + 1)
    ]
    found = sector_spectrum(
        liouvillian(interacting_ring(sites), jumps), spaces, sectors
    )
    expected = scipy.linalg.eigvals(
        master_equation(interacting_ring(sites), jumps, spaces)
    )
    assert_same_spectrum(found, expected)
    assert np.sum(np.abs(found) < 1e-9) == sites + 1


def test_momentum_and_spin_flip_sectors_rebuild_the_spectrum() -> None:
    # rho -> U rho U† with U the product of sigma_x, and translation
    sites = 4
    H = sum(
        sigma_x(i) * sigma_x((i + 1) % sites)
        + 0.6 * sigma_z(i) * sigma_z((i + 1) % sites)
        for i in range(sites)
    )
    jumps = [0.4 * sigma_z(i) for i in range(sites)]
    jumps += [0.3 * sigma_minus(i) for i in range(sites)]
    jumps += [0.3 * sigma_plus(i) for i in range(sites)]
    spaces = {PauliSpace(): range(sites)}
    flip = [left(sigma_x(i)) for i in range(sites)]
    flip += [right(hc(sigma_x(i))) for i in range(sites)]
    sectors = [
        {
            "permutation_symmetries": [(shift(sites), momentum(k, sites))],
            "onsite_symmetries": [(flip, parity)],
        }
        for k in range(sites)
        for parity in (1, -1)
    ]
    assert isinstance(H, OpSum)
    found = sector_spectrum(liouvillian(H, jumps), spaces, sectors)
    expected = scipy.linalg.eigvals(master_equation(H, jumps, spaces))
    assert_same_spectrum(found, expected)
