"""Symmetry-adapted bases: momentum, spin flip, spin inversion, particle-hole."""

from __future__ import annotations

import cmath
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from fockit.algebra import (
    OpSum,
    b,
    bdag,
    c,
    cdag,
    identity,
    nb,
    nc,
    sigma_x,
    sigma_y,
    sigma_z,
)
from fockit.hilbert import (
    Basis,
    BosonSpace,
    FermionSpace,
    LocalSpace,
    PauliSpace,
    build_basis,
    expand_state,
    expectation,
    to_matrix,
)
from fockit.solve import eigenstates, eigh_full, groundstate

SPINS = ("up", "dn")


def shift(sites: int, spins: Sequence[str] | None = None) -> dict[Any, Any]:
    if spins is None:
        return {i: (i + 1) % sites for i in range(sites)}
    return {(i, s): ((i + 1) % sites, s) for i in range(sites) for s in spins}


def momentum(k: int, sites: int) -> complex:
    return cmath.exp(2j * cmath.pi * k / sites)


def spectrum[S: LocalSpace](
    H: OpSum | complex, spaces: Mapping[S, Sequence[Hashable]], **kwargs: Any
) -> NDArray[np.float64]:
    energies, _ = eigh_full(to_matrix(H, build_basis(spaces, **kwargs)))
    return energies


def assert_momentum_sectors_rebuild_spectrum[S: LocalSpace](
    H: OpSum | complex,
    spaces: Mapping[S, Sequence[Hashable]],
    sites: int,
    label_map: Mapping[Any, Hashable],
    **kwargs: Any,
) -> None:
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


def hopping_ring(sites: int, spins: Sequence[str] | None = None) -> OpSum | complex:
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


def hubbard_ring(sites: int, u: float = 4.0) -> OpSum | complex:
    return hopping_ring(sites, SPINS) + u * sum(
        nc(i, "up") * nc(i, "dn") for i in range(sites)
    )


def heisenberg_ring(sites: int) -> OpSum | complex:
    return sum(
        sigma_x(i) * sigma_x((i + 1) % sites)
        + sigma_y(i) * sigma_y((i + 1) % sites)
        + sigma_z(i) * sigma_z((i + 1) % sites)
        for i in range(sites)
    )


def test_single_particle_on_ring_fills_the_cosine_band() -> None:
    sites = 6
    space = {FermionSpace(): range(sites)}
    one = {sum(nc(i) for i in range(sites)): 1}
    for k in range(sites):
        symmetries = [(shift(sites), momentum(k, sites))]
        energies = spectrum(
            hopping_ring(sites),
            space,
            diagonal_symmetries=one,
            permutation_symmetries=symmetries,
        )
        assert energies == pytest.approx([-2 * np.cos(2 * np.pi * k / sites)])


def test_hubbard_momentum_sectors_rebuild_spectrum() -> None:
    sites = 5
    modes = [(i, s) for i in range(sites) for s in SPINS]
    sectors = {
        sum(nc(i, "up") for i in range(sites)): 2,
        sum(nc(i, "dn") for i in range(sites)): 2,
    }
    assert_momentum_sectors_rebuild_spectrum(
        hubbard_ring(sites),
        {FermionSpace(): modes},
        sites,
        shift(sites, SPINS),
        diagonal_symmetries=sectors,
    )


def test_hubbard_momentum_and_spin_flip_sectors_rebuild_spectrum() -> None:
    sites = 4
    modes = [(i, s) for i in range(sites) for s in SPINS]
    H = hubbard_ring(sites)
    sz = 0.5 * sum(nc(i, "up") - nc(i, "dn") for i in range(sites))
    sectors = {sum(nc(*m) for m in modes): 4, sz: 0}
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
def test_heisenberg_ring_ground_state_momentum(sites: int, ground_k: int) -> None:
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


def test_holstein_ring_translates_electrons_and_phonons_together() -> None:
    sites, omega, g = 3, 1.0, 0.6
    H = hopping_ring(sites) + sum(
        omega * nb(i) + g * nc(i) * (b(i) + bdag(i)) for i in range(sites)
    )
    spaces = {FermionSpace(): range(sites), BosonSpace(n_max=2): range(sites)}
    one = {sum(nc(i) for i in range(sites)): 1}
    assert_momentum_sectors_rebuild_spectrum(
        H, spaces, sites, shift(sites), diagonal_symmetries=one
    )


def test_bose_hubbard_momentum_sectors_rebuild_spectrum() -> None:
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


def test_non_symmetric_operator_is_rejected_in_a_momentum_sector() -> None:
    sites = 4
    basis = build_basis(
        {FermionSpace(): range(sites)},
        permutation_symmetries=[(shift(sites), momentum(1, sites))],
    )
    with pytest.raises(ValueError, match="not symmetric"):
        to_matrix(nc(0), basis)


def reflection(sites: int) -> dict[int, int]:
    return {i: sites - 1 - i for i in range(sites)}


def test_reflection_parity_splits_momentum_zero_and_pi() -> None:
    sites = 8
    H = heisenberg_ring(sites)
    space = {PauliSpace(): range(sites)}
    sz0 = {0.5 * sum(sigma_z(i) for i in range(sites)): 0}
    for k in (0, sites // 2):
        translation = (shift(sites), momentum(k, sites))
        parts = [
            spectrum(
                H,
                space,
                diagonal_symmetries=sz0,
                permutation_symmetries=[translation, (reflection(sites), parity)],
            )
            for parity in (1, -1)
        ]
        whole = spectrum(
            H, space, diagonal_symmetries=sz0, permutation_symmetries=[translation]
        )
        assert np.sort(np.concatenate(parts)) == pytest.approx(whole)


def test_square_lattice_heisenberg_ground_state_has_full_lattice_symmetry() -> None:
    side = 4
    sites = [(x, y) for x in range(side) for y in range(side)]
    H = sum(
        f(x, y) * f((x + dx) % side, (y + dy) % side)
        for x, y in sites
        for dx, dy in ((1, 0), (0, 1))
        for f in (sigma_x, sigma_y, sigma_z)
    )
    space = {PauliSpace(): sites}
    sz0 = {0.5 * sum(sigma_z(x, y) for x, y in sites): 0}
    energy, _ = groundstate(to_matrix(H, build_basis(space, diagonal_symmetries=sz0)))

    shift_x = {(x, y): ((x + 1) % side, y) for x, y in sites}
    shift_y = {(x, y): (x, (y + 1) % side) for x, y in sites}
    rotation = {(x, y): (y, -x % side) for x, y in sites}
    mirror = {(x, y): (-x % side, y) for x, y in sites}
    fully_symmetric = [(shift_x, 1), (shift_y, 1), (rotation, 1), (mirror, 1)]
    lowest = spectrum(
        H, space, diagonal_symmetries=sz0, permutation_symmetries=fully_symmetric
    )[0]
    assert lowest == pytest.approx(energy)


def test_symmetry_eigenvalues_that_do_not_fit_are_rejected() -> None:
    sites = 4
    space = {FermionSpace(): range(sites)}
    with pytest.raises(ValueError, match="don't fit"):
        build_basis(
            space,
            permutation_symmetries=[
                (shift(sites), momentum(1, sites)),
                (reflection(sites), 1),
            ],
        )
    with pytest.raises(ValueError, match="don't fit"):
        build_basis(space, permutation_symmetries=[(shift(sites), np.exp(0.3j))])


def hubbard_bases(sites: int, n_up: int, n_dn: int, k: int) -> tuple[Basis, Basis]:
    modes = [(i, s) for i in range(sites) for s in SPINS]
    diagonal = {
        sum(nc(i, "up") for i in range(sites)): n_up,
        sum(nc(i, "dn") for i in range(sites)): n_dn,
    }
    full = build_basis({FermionSpace(): modes}, diagonal_symmetries=diagonal)
    sym = build_basis(
        {FermionSpace(): modes},
        diagonal_symmetries=diagonal,
        permutation_symmetries=[(shift(sites, SPINS), momentum(k, sites))],
    )
    return sym, full


def test_expanded_ground_state_is_eigenvector_of_full_hamiltonian() -> None:
    sites = 6
    H = hubbard_ring(sites)
    sym, full = hubbard_bases(sites, 2, 2, k=1)
    energy, psi = groundstate(to_matrix(H, sym))
    v = expand_state(psi, sym, full)
    assert np.linalg.norm(v) == pytest.approx(1)
    assert to_matrix(H, full) @ v == pytest.approx(energy * v)


def test_momentum_eigenstate_has_uniform_density() -> None:
    sites = 6
    sym, full = hubbard_bases(sites, 2, 3, k=2)
    _, psi = groundstate(to_matrix(hubbard_ring(sites), sym))
    v = expand_state(psi, sym, full)
    densities = [
        expectation(to_matrix(nc(i, "up"), full), v).real for i in range(sites)
    ]
    assert densities == pytest.approx([2 / sites] * sites)


def test_expanded_states_from_different_momenta_are_orthogonal() -> None:
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


def test_heisenberg_correlations_from_symmetric_ground_state_match_full_basis() -> None:
    sites = 8
    H = heisenberg_ring(sites)
    space = {PauliSpace(): range(sites)}
    sz0 = {0.5 * sum(sigma_z(i) for i in range(sites)): 0}
    full = build_basis(space, diagonal_symmetries=sz0)
    sym = build_basis(
        space,
        diagonal_symmetries=sz0,
        permutation_symmetries=[(shift(sites), momentum(0, sites))],
    )
    _, psi_full = groundstate(to_matrix(H, full))
    _, psi_sym = groundstate(to_matrix(H, sym))
    v = expand_state(psi_sym, sym, full)
    for r in range(1, sites // 2 + 1):
        zz = to_matrix(sigma_z(0) * sigma_z(r), full)
        assert expectation(zz, v) == pytest.approx(expectation(zz, psi_full))


def test_expand_rejects_a_full_basis_with_other_states() -> None:
    sym, _ = hubbard_bases(4, 2, 2, k=0)
    _, other = hubbard_bases(4, 3, 2, k=0)
    with pytest.raises(ValueError, match="missing states"):
        expand_state(np.ones(sym.dim), sym, other)


def test_symmetry_that_leaves_the_selected_states_is_rejected() -> None:
    sites = 4
    modes = [(i, s) for i in range(sites) for s in SPINS]
    flip = {(i, "up"): (i, "dn") for i in range(sites)}
    flip |= {(i, "dn"): (i, "up") for i in range(sites)}
    sz = 0.5 * sum(nc(i, "up") - nc(i, "dn") for i in range(sites))
    # spin flip turns Sz = 1 into Sz = -1
    with pytest.raises(ValueError, match="outside"):
        build_basis(
            {FermionSpace(): modes},
            diagonal_symmetries={sz: 1},
            permutation_symmetries=[(flip, 1)],
        )
    # translation moves the spin off site 0
    with pytest.raises(ValueError, match="outside"):
        build_basis(
            {PauliSpace(): range(sites)},
            permutation_symmetries=[(shift(sites), 1)],
            predicate_function=lambda lv: lv[:, 0] == 1,
        )
    # spin inversion turns Sz = 1 into Sz = -1
    with pytest.raises(ValueError, match="outside"):
        build_basis(
            {PauliSpace(): range(sites)},
            diagonal_symmetries={0.5 * sum(sigma_z(i) for i in range(sites)): 1},
            onsite_symmetries=[(spin_inversion(sites), 1)],
        )


def tfim_ring(sites: int, h: float) -> OpSum | complex:
    return -sum(sigma_z(i) * sigma_z((i + 1) % sites) for i in range(sites)) - h * sum(
        sigma_x(i) for i in range(sites)
    )


def spin_inversion(sites: int) -> list[OpSum]:
    return [sigma_x(i) for i in range(sites)]


def test_tfim_momentum_and_parity_sectors_rebuild_spectrum() -> None:
    sites = 6
    H = tfim_ring(sites, 0.7)
    space = {PauliSpace(): range(sites)}
    parts = [
        spectrum(
            H,
            space,
            permutation_symmetries=[(shift(sites), momentum(k, sites))],
            onsite_symmetries=[(spin_inversion(sites), parity)],
        )
        for k in range(sites)
        for parity in (1, -1)
    ]
    assert np.sort(np.concatenate(parts)) == pytest.approx(spectrum(H, space))


def test_tfim_ordered_phase_has_an_even_ground_state_and_a_closing_odd_partner() -> (
    None
):
    splittings = []
    for sites in (6, 8):
        H = tfim_ring(sites, 0.3)
        space = {PauliSpace(): range(sites)}
        even, odd = (
            spectrum(H, space, onsite_symmetries=[(spin_inversion(sites), parity)])[0]
            for parity in (1, -1)
        )
        assert even < odd
        splittings.append(odd - even)
    assert splittings[1] < splittings[0]


def test_total_spin_from_a_penalty_and_from_sz_sectors_agree() -> None:
    # singlets appear at Sz=0 but not at Sz=1, so they are the Sz=0 levels
    # left after removing the Sz=1 ones; adding 10 S^2 lifts every other S
    sites = 10
    H = heisenberg_ring(sites)
    total_spin = [
        0.5 * sum(f(i) for i in range(sites)) for f in (sigma_x, sigma_y, sigma_z)
    ]
    S2 = sum(s * s for s in total_spin)
    space = {PauliSpace(): range(sites)}
    sz0 = build_basis(space, diagonal_symmetries={total_spin[2]: 0})

    spin_one_or_more = spectrum(H, space, diagonal_symmetries={total_spin[2]: 1})
    singlets = list(eigh_full(to_matrix(H, sz0))[0])
    for level in spin_one_or_more:
        singlets.pop(int(np.argmin(np.abs(np.array(singlets) - level))))
    singlets.sort()

    penalized, _ = eigh_full(to_matrix(H + 10 * S2, sz0))
    assert penalized[:3] == pytest.approx(singlets[:3])

    energy, psi = groundstate(to_matrix(H, sz0))
    assert expectation(to_matrix(S2, sz0), psi).real == pytest.approx(0, abs=1e-9)
    assert singlets[0] == pytest.approx(energy)
    # the lowest triplet lies between the two lowest singlets
    assert singlets[0] < spin_one_or_more[0] < singlets[1]


def test_heisenberg_ground_state_lies_in_its_spin_inversion_sector() -> None:
    sites = 8
    H = heisenberg_ring(sites)
    space = {PauliSpace(): range(sites)}
    sz0 = {0.5 * sum(sigma_z(i) for i in range(sites)): 0}
    full = build_basis(space, diagonal_symmetries=sz0)
    energy, psi = groundstate(to_matrix(H, full))
    flip_all = identity()
    for i in range(sites):
        flip_all = flip_all * sigma_x(i)
    parity = expectation(to_matrix(flip_all, full), psi).real
    assert abs(parity) == pytest.approx(1)

    sym = build_basis(
        space,
        diagonal_symmetries=sz0,
        permutation_symmetries=[(shift(sites), momentum(0, sites))],
        onsite_symmetries=[(spin_inversion(sites), round(parity))],
    )
    sym_energy, sym_psi = groundstate(to_matrix(H, sym))
    assert sym_energy == pytest.approx(energy)
    v = expand_state(sym_psi, sym, full)
    assert to_matrix(H, full) @ v == pytest.approx(energy * v)


def test_hubbard_particle_hole_spin_flip_and_momentum_sectors_rebuild_spectrum() -> (
    None
):
    sites = 4
    modes = [(i, s) for i in range(sites) for s in SPINS]
    H = hopping_ring(sites, SPINS) + 4 * sum(
        (nc(i, "up") - 0.5) * (nc(i, "dn") - 0.5) for i in range(sites)
    )
    sz = 0.5 * sum(nc(i, "up") - nc(i, "dn") for i in range(sites))
    half_filling = {sum(nc(*m) for m in modes): sites, sz: 0}
    flip = {(i, "up"): (i, "dn") for i in range(sites)}
    flip |= {(i, "dn"): (i, "up") for i in range(sites)}
    particle_hole = [c(i, s) + (-1) ** i * cdag(i, s) for i, s in modes]
    space = {FermionSpace(): modes}

    parts = [
        spectrum(
            H,
            space,
            diagonal_symmetries=half_filling,
            permutation_symmetries=[
                (shift(sites, SPINS), momentum(k, sites)),
                (flip, spin_parity),
            ],
            onsite_symmetries=[(particle_hole, ph_parity)],
        )
        for k in range(sites)
        for spin_parity in (1, -1)
        for ph_parity in (1, -1)
    ]
    full = spectrum(H, space, diagonal_symmetries=half_filling)
    assert np.sort(np.concatenate(parts)) == pytest.approx(full)


def test_spinless_particle_hole_and_momentum_sectors_rebuild_spectrum() -> None:
    sites = 6
    H = hopping_ring(sites) + 2 * sum(
        (nc(i) - 0.5) * (nc((i + 1) % sites) - 0.5) for i in range(sites)
    )
    space = {FermionSpace(): range(sites)}
    half_filling = {sum(nc(i) for i in range(sites)): sites // 2}
    particle_hole = [c(i) + (-1) ** i * cdag(i) for i in range(sites)]
    parts = [
        spectrum(
            H,
            space,
            diagonal_symmetries=half_filling,
            permutation_symmetries=[(shift(sites), momentum(k, sites))],
            onsite_symmetries=[(particle_hole, parity)],
        )
        for k in range(sites)
        for parity in (1, -1)
    ]
    full = spectrum(H, space, diagonal_symmetries=half_filling)
    assert np.sort(np.concatenate(parts)) == pytest.approx(full)


def test_onsite_symmetry_that_is_not_a_symmetry_is_rejected() -> None:
    sites = 4
    space = {PauliSpace(): range(sites)}
    # flipping only even sites doesn't commute with translation
    with pytest.raises(ValueError, match="commute"):
        build_basis(
            space,
            permutation_symmetries=[(shift(sites), 1)],
            onsite_symmetries=[([sigma_x(i) for i in range(0, sites, 2)], 1)],
        )
    # a longitudinal field breaks spin inversion
    basis = build_basis(space, onsite_symmetries=[(spin_inversion(sites), 1)])
    field = 0.1 * sum(sigma_z(i) for i in range(sites))
    with pytest.raises(ValueError, match="not symmetric"):
        to_matrix(tfim_ring(sites, 0.5) + field, basis)
