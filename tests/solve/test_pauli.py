"""Tests against analytic Pauli/spin spectra."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import normal_order, sigma_x, sigma_y, sigma_z, sm, sp
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates, eigh_full, groundstate
from operator_algebra.solve.matrix import expectations
from operator_algebra.solve.spaces import PauliSpace


def heisenberg_chain(sites: int, j: float):
    labels = list(range(sites))
    H = 0 * sigma_z(0) * sigma_z(0)
    for i in range(sites - 1):
        H = H + (j / 4) * (
            sigma_x(i) * sigma_x(i + 1)
            + sigma_y(i) * sigma_y(i + 1)
            + sigma_z(i) * sigma_z(i + 1)
        )
    return labels, H


def test_heisenberg_dimer_matches_singlet_triplet():
    labels, H = heisenberg_chain(2, j=1.0)
    basis = enumerate_basis(labels, PauliSpace())
    energies, _ = eigenstates(H, basis, k=4)
    assert np.allclose(np.sort(energies), [-0.75, 0.25, 0.25, 0.25])


def test_zeeman_groundstate_aligns_with_field():
    h = 1.7
    basis = enumerate_basis(["A"], PauliSpace())
    e0, psi = groundstate(-h * sigma_z("A"), basis)
    assert e0 == pytest.approx(-h, abs=1e-12)
    assert expectations(sp("A") * sm("A"), psi, basis) == pytest.approx(1.0)


def test_two_spin_total_spin_squared():
    S2 = sum(
        (0.5 * (sigma(0) + sigma(1))) ** 2 for sigma in (sigma_x, sigma_y, sigma_z)
    )
    basis = enumerate_basis([0, 1], PauliSpace())
    energies, _ = eigh_full(normal_order(S2), basis)
    assert np.sort(energies) == pytest.approx([0.0, 2.0, 2.0, 2.0])


def test_xxz_chain_in_ladder_form_has_same_spectrum():
    sites, delta = 4, 0.5
    labels = list(range(sites))
    zz = sum(delta * sigma_z(i) * sigma_z(i + 1) for i in range(sites - 1))
    H_xy = zz + sum(
        sigma_x(i) * sigma_x(i + 1) + sigma_y(i) * sigma_y(i + 1)
        for i in range(sites - 1)
    )
    H_ladder = zz + sum(
        2 * (sp(i) * sm(i + 1) + sm(i) * sp(i + 1)) for i in range(sites - 1)
    )
    basis = enumerate_basis(labels, PauliSpace())
    e_xy, _ = eigh_full(H_xy, basis)
    e_ladder, _ = eigh_full(normal_order(H_ladder), basis)
    assert e_ladder == pytest.approx(e_xy)
