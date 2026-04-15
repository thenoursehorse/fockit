"""Tests against analytic Pauli/spin spectra."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import sigma_x, sigma_y, sigma_z
from operator_algebra.algebra import PAULI
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates, groundstate


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
    basis = enumerate_basis(labels, PAULI)
    energies, _ = eigenstates(H, basis, k=4)
    assert np.allclose(np.sort(energies), [-0.75, 0.25, 0.25, 0.25])


def test_zeeman_groundstate_aligns_with_field():
    h = 1.7
    basis = enumerate_basis(["A"], PAULI)
    e0, _ = groundstate(-h * sigma_z("A"), basis)
    assert e0 == pytest.approx(-h, abs=1e-12)
