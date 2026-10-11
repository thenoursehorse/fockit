"""Symmetry-adapted bases: momentum and spin-flip sectors."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import (
    b,
    bdag,
    c,
    cdag,
    n,
    nb,
    sigma_x,
    sigma_y,
    sigma_z,
)
from operator_algebra.solve.basis import enumerate_basis, expand_state
from operator_algebra.solve.diagonalize import eigenstates, eigh_full, groundstate
from operator_algebra.solve.matrix import expectations, to_matrix
from operator_algebra.solve.spaces import BosonSpace, FermionSpace, PauliSpace

SPINS = ("up", "dn")


def shift(sites, spins=None):
    if spins is None:
        return {i: (i + 1) % sites for i in range(sites)}
    return {(i, s): ((i + 1) % sites, s) for i in range(sites) for s in spins}


def momentum(k, sites):
    return np.exp(2j * np.pi * k / sites)


def spectrum(H, spaces, **kwargs):
    energies, _ = eigh_full(to_matrix(H, enumerate_basis(spaces, **kwargs)))
    return energies


def assert_momentum_sectors_rebuild_spectrum(H, spaces, sites, label_map, **kwargs):
    full = spectrum(H, spaces, **kwargs)
    parts = [
        spectrum(
            H,
            spaces,
            permutation_symmetries=[(label_map, momentum(k, sites))],
            **kwargs,
        )
        for k in range(sites)
    ]
    assert np.sort(np.concatenate(parts)) == pytest.approx(full)


def hopping_ring(sites, spins=None):
    if spins is None:
        return -sum(
            cdag((i + 1) % sites) * c(i) + cdag(i) * c((i + 1) % sites)
            for i in range(sites)
        )
    return -sum(
        cdag((i + 1) % sites, s) * c(i, s) + cdag(i, s) * c((i + 1) % sites, s)
        for i in range(sites)
        for s in spins
    )


def hubbard_ring(sites, u=4.0):
    return hopping_ring(sites, SPINS) + u * sum(
        n(i, "up") * n(i, "dn") for i in range(sites)
    )


def heisenberg_ring(sites):
    return sum(
        sigma_x(i) * sigma_x((i + 1) % sites)
        + sigma_y(i) * sigma_y((i + 1) % sites)
        + sigma_z(i) * sigma_z((i + 1) % sites)
        for i in range(sites)
    )


def test_single_particle_on_ring_fills_the_cosine_band():
    sites = 6
    space = {FermionSpace(): range(sites)}
    one = {sum(n(i) for i in range(sites)): 1}
    for k in range(sites):
        symmetries = [(shift(sites), momentum(k, sites))]
        energies = spectrum(
            hopping_ring(sites),
            space,
            diagonal_symmetries=one,
            permutation_symmetries=symmetries,
        )
        assert energies == pytest.approx([-2 * np.cos(2 * np.pi * k / sites)])


def test_hubbard_momentum_sectors_rebuild_spectrum():
    sites = 5
    modes = [(i, s) for i in range(sites) for s in SPINS]
    sectors = {
        sum(n(i, "up") for i in range(sites)): 2,
        sum(n(i, "dn") for i in range(sites)): 2,
    }
    assert_momentum_sectors_rebuild_spectrum(
        hubbard_ring(sites),
        {FermionSpace(): modes},
        sites,
        shift(sites, SPINS),
        diagonal_symmetries=sectors,
    )


def test_hubbard_momentum_and_spin_flip_sectors_rebuild_spectrum():
    sites = 4
    modes = [(i, s) for i in range(sites) for s in SPINS]
    H = hubbard_ring(sites)
    sz = 0.5 * sum(n(i, "up") - n(i, "dn") for i in range(sites))
    sectors = {sum(n(*m) for m in modes): 4, sz: 0}
    flip = {(i, "up"): (i, "dn") for i in range(sites)}
    flip |= {(i, "dn"): (i, "up") for i in range(sites)}
    space = {FermionSpace(): modes}

    full = spectrum(H, space, diagonal_symmetries=sectors)
    parts = [
        spectrum(
            H,
            space,
            diagonal_symmetries=sectors,
            permutation_symmetries=[
                (shift(sites, SPINS), momentum(k, sites)),
                (flip, parity),
            ],
        )
        for k in range(sites)
        for parity in (1, -1)
    ]
    assert np.sort(np.concatenate(parts)) == pytest.approx(full)


@pytest.mark.parametrize(("sites", "ground_k"), [(6, 3), (8, 0)])
def test_heisenberg_ring_ground_state_momentum(sites, ground_k):
    # Marshall sign rule: k = pi when sites/2 is odd, k = 0 when it is even
    H = heisenberg_ring(sites)
    space = {PauliSpace(): range(sites)}
    sz0 = {0.5 * sum(sigma_z(i) for i in range(sites)): 0}
    lowest = [
        spectrum(
            H,
            space,
            diagonal_symmetries=sz0,
            permutation_symmetries=[(shift(sites), momentum(k, sites))],
        )[0]
        for k in range(sites)
    ]
    assert int(np.argmin(lowest)) == ground_k
    assert min(lowest) == pytest.approx(spectrum(H, space, diagonal_symmetries=sz0)[0])


def test_holstein_ring_translates_electrons_and_phonons_together():
    sites, omega, g = 3, 1.0, 0.6
    H = hopping_ring(sites) + sum(
        omega * nb(i) + g * n(i) * (b(i) + bdag(i)) for i in range(sites)
    )
    spaces = {FermionSpace(): range(sites), BosonSpace(n_max=2): range(sites)}
    one = {sum(n(i) for i in range(sites)): 1}
    assert_momentum_sectors_rebuild_spectrum(
        H, spaces, sites, shift(sites), diagonal_symmetries=one
    )


def test_bose_hubbard_momentum_sectors_rebuild_spectrum():
    sites = 4
    H = -sum(
        bdag((i + 1) % sites) * b(i) + bdag(i) * b((i + 1) % sites)
        for i in range(sites)
    ) + 0.5 * sum(bdag(i) * bdag(i) * b(i) * b(i) for i in range(sites))
    sectors = {sum(nb(i) for i in range(sites)): 3}
    assert_momentum_sectors_rebuild_spectrum(
        H,
        {BosonSpace(n_max=3): range(sites)},
        sites,
        shift(sites),
        diagonal_symmetries=sectors,
    )


def test_non_symmetric_operator_is_rejected_in_a_momentum_sector():
    sites = 4
    basis = enumerate_basis(
        {FermionSpace(): range(sites)},
        permutation_symmetries=[(shift(sites), momentum(1, sites))],
    )
    with pytest.raises(ValueError, match="not symmetric"):
        to_matrix(n(0), basis)


def test_non_commuting_symmetries_are_rejected():
    sites = 4
    reflection = {i: (sites - 1 - i) % sites for i in range(sites)}
    with pytest.raises(ValueError, match="commute"):
        enumerate_basis(
            {FermionSpace(): range(sites)},
            permutation_symmetries=[(shift(sites), 1), (reflection, 1)],
        )


def hubbard_bases(sites, n_up, n_dn, k):
    modes = [(i, s) for i in range(sites) for s in SPINS]
    diagonal = {
        sum(n(i, "up") for i in range(sites)): n_up,
        sum(n(i, "dn") for i in range(sites)): n_dn,
    }
    full = enumerate_basis({FermionSpace(): modes}, diagonal_symmetries=diagonal)
    sym = enumerate_basis(
        {FermionSpace(): modes},
        diagonal_symmetries=diagonal,
        permutation_symmetries=[(shift(sites, SPINS), momentum(k, sites))],
    )
    return sym, full


def test_expanded_ground_state_is_eigenvector_of_full_hamiltonian():
    sites = 6
    H = hubbard_ring(sites)
    sym, full = hubbard_bases(sites, 2, 2, k=1)
    energy, psi = groundstate(to_matrix(H, sym))
    v = expand_state(psi, sym, full)
    assert np.linalg.norm(v) == pytest.approx(1)
    assert to_matrix(H, full) @ v == pytest.approx(energy * v)


def test_momentum_eigenstate_has_uniform_density():
    sites = 6
    sym, full = hubbard_bases(sites, 2, 3, k=2)
    _, psi = groundstate(to_matrix(hubbard_ring(sites), sym))
    v = expand_state(psi, sym, full)
    densities = [
        expectations(to_matrix(n(i, "up"), full), v).real for i in range(sites)
    ]
    assert densities == pytest.approx([2 / sites] * sites)


def test_expanded_states_from_different_momenta_are_orthogonal():
    sites = 5
    H = hubbard_ring(sites)
    columns = []
    for k in range(sites):
        sym, full = hubbard_bases(sites, 2, 1, k)
        _, states = eigenstates(to_matrix(H, sym), k=2)
        columns.append(expand_state(states, sym, full))
    V = np.hstack(columns)
    overlaps = V.conj().T @ V
    assert overlaps == pytest.approx(np.eye(V.shape[1]))


def test_heisenberg_correlations_from_symmetric_ground_state_match_full_basis():
    sites = 8
    H = heisenberg_ring(sites)
    space = {PauliSpace(): range(sites)}
    sz0 = {0.5 * sum(sigma_z(i) for i in range(sites)): 0}
    full = enumerate_basis(space, diagonal_symmetries=sz0)
    sym = enumerate_basis(
        space,
        diagonal_symmetries=sz0,
        permutation_symmetries=[(shift(sites), momentum(0, sites))],
    )
    _, psi_full = groundstate(to_matrix(H, full))
    _, psi_sym = groundstate(to_matrix(H, sym))
    v = expand_state(psi_sym, sym, full)
    for r in range(1, sites // 2 + 1):
        zz = to_matrix(sigma_z(0) * sigma_z(r), full)
        assert expectations(zz, v) == pytest.approx(expectations(zz, psi_full))


def test_expand_rejects_a_full_basis_with_other_states():
    sym, _ = hubbard_bases(4, 2, 2, k=0)
    _, other = hubbard_bases(4, 3, 2, k=0)
    with pytest.raises(ValueError, match="missing states"):
        expand_state(np.ones(sym.dim), sym, other)
