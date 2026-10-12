"""Restricting a basis to symmetry sectors."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from fockit import (
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
from fockit.solve.basis import enumerate_basis
from fockit.solve.diagonalize import eigh_full
from fockit.solve.matrix import to_matrix
from fockit.solve.spaces import BosonSpace, FermionSpace, PauliSpace

SPINS = ("up", "dn")


def hubbard_chain(sites, t=1.0, u=4.0):
    modes = [(i, s) for i in range(sites) for s in SPINS]
    H = -t * sum(
        cdag(i, s) * c(i + 1, s) + cdag(i + 1, s) * c(i, s)
        for i in range(sites - 1)
        for s in SPINS
    ) + u * sum(n(i, "up") * n(i, "dn") for i in range(sites))
    return modes, H


def spectrum(H, spaces, **kwargs):
    energies, _ = eigh_full(to_matrix(H, enumerate_basis(spaces, **kwargs)))
    return energies


def assert_sectors_rebuild_spectrum(H, spaces, sector_kwargs):
    full = spectrum(H, spaces)
    parts = np.concatenate([spectrum(H, spaces, **kw) for kw in sector_kwargs])
    assert np.sort(parts) == pytest.approx(full)


class TestFermionSectors:
    sites = 4
    modes = [(i, s) for i in range(sites) for s in SPINS]
    n_up = sum(n(i, "up") for i in range(sites))
    n_dn = sum(n(i, "dn") for i in range(sites))
    n_tot = n_up + n_dn
    sz = 0.5 * (n_up - n_dn)
    double_occ = sum(n(i, "up") * n(i, "dn") for i in range(sites))

    def dim(self, **kwargs):
        return enumerate_basis({FermionSpace(): self.modes}, **kwargs).dim

    def test_particle_number_sector_size(self):
        for n_part in range(2 * self.sites + 1):
            expected = comb(2 * self.sites, n_part)
            assert self.dim(diagonal_symmetries={self.n_tot: n_part}) == expected

    def test_spin_resolved_sector_size(self):
        expected = comb(self.sites, 2) * comb(self.sites, 2)
        assert self.dim(diagonal_symmetries={self.n_tot: 4, self.sz: 0}) == expected
        assert self.dim(diagonal_symmetries={self.n_up: 2, self.n_dn: 2}) == expected

    def test_no_double_occupancy_sector_size(self):
        # choose the occupied sites, then a spin for each
        for n_part in range(self.sites + 1):
            expected = comb(self.sites, n_part) * 2**n_part
            sectors = {self.n_tot: n_part, self.double_occ: 0}
            assert self.dim(diagonal_symmetries=sectors) == expected

    def test_parity_sector_size(self):
        even = self.dim(predicate_function=lambda lv: lv.sum(axis=1) % 2 == 0)
        assert even == 2 ** (2 * self.sites - 1)

    def test_hubbard_spin_sectors_rebuild_full_spectrum(self):
        sites = 3
        modes, H = hubbard_chain(sites)
        n_up = sum(n(i, "up") for i in range(sites))
        n_dn = sum(n(i, "dn") for i in range(sites))
        sectors = [
            {"diagonal_symmetries": {n_up: k_up, n_dn: k_dn}}
            for k_up in range(sites + 1)
            for k_dn in range(sites + 1)
        ]
        assert_sectors_rebuild_spectrum(H, {FermionSpace(): modes}, sectors)

    def test_pairing_parity_sectors_rebuild_full_spectrum(self):
        # pairing breaks particle number but keeps fermion parity
        sites, delta = 3, 0.7
        modes, H = hubbard_chain(sites)
        H = H + delta * sum(
            cdag(i, "up") * cdag(i, "dn") + c(i, "dn") * c(i, "up")
            for i in range(sites)
        )
        sectors = [
            {"predicate_function": lambda lv, p=p: lv.sum(axis=1) % 2 == p}
            for p in (0, 1)
        ]
        assert_sectors_rebuild_spectrum(H, {FermionSpace(): modes}, sectors)


class TestBosonSectors:
    sites, n_max = 3, 3

    def test_particle_number_sector_size(self):
        # stars and bars, while the cutoff is not reached
        n_tot = sum(nb(i) for i in range(self.sites))
        space = {BosonSpace(n_max=self.n_max): range(self.sites)}
        for n_part in range(self.n_max + 1):
            basis = enumerate_basis(space, diagonal_symmetries={n_tot: n_part})
            assert basis.dim == comb(n_part + self.sites - 1, self.sites - 1)

    def test_bose_hubbard_number_sectors_rebuild_full_spectrum(self):
        sites = self.sites
        H = -sum(
            bdag(i) * b(i + 1) + bdag(i + 1) * b(i) for i in range(sites - 1)
        ) + 0.5 * sum(bdag(i) * bdag(i) * b(i) * b(i) for i in range(sites))
        n_tot = sum(nb(i) for i in range(sites))
        sectors = [
            {"diagonal_symmetries": {n_tot: k}} for k in range(sites * self.n_max + 1)
        ]
        space = {BosonSpace(n_max=self.n_max): range(sites)}
        assert_sectors_rebuild_spectrum(H, space, sectors)


class TestPauliSectors:
    sites = 6
    sz = 0.5 * sum(sigma_z(i) for i in range(sites))

    def test_magnetization_sector_size(self):
        space = {PauliSpace(): range(self.sites)}
        for n_up in range(self.sites + 1):
            m = n_up - self.sites / 2
            basis = enumerate_basis(space, diagonal_symmetries={self.sz: m})
            assert basis.dim == comb(self.sites, n_up)

    def test_xxz_magnetization_sectors_rebuild_full_spectrum(self):
        H = sum(
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + 0.5 * sigma_z(i) * sigma_z(i + 1)
            for i in range(self.sites - 1)
        )
        sectors = [
            {"diagonal_symmetries": {self.sz: n_up - self.sites / 2}}
            for n_up in range(self.sites + 1)
        ]
        assert_sectors_rebuild_spectrum(H, {PauliSpace(): range(self.sites)}, sectors)

    def test_transverse_ising_parity_sectors_rebuild_full_spectrum(self):
        # σxσx flips two spins and σz flips none, so the parity of up spins
        # is conserved
        H = -sum(sigma_x(i) * sigma_x(i + 1) for i in range(self.sites - 1))
        H = H - 0.8 * sum(sigma_z(i) for i in range(self.sites))
        sectors = [
            {"predicate_function": lambda lv, p=p: lv.sum(axis=1) % 2 == p}
            for p in (0, 1)
        ]
        assert_sectors_rebuild_spectrum(H, {PauliSpace(): range(self.sites)}, sectors)


def test_off_diagonal_sector_operators_are_rejected():
    for space, labels, op in [
        (PauliSpace(), range(2), sigma_x(0)),
        (FermionSpace(), range(2), cdag(0) * c(1) + cdag(1) * c(0)),
    ]:
        with pytest.raises(ValueError, match="not diagonal"):
            enumerate_basis({space: labels}, diagonal_symmetries={op: 1})
