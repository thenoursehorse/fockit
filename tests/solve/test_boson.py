"""Tests against analytic boson spectra."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import b, bdag
from operator_algebra.algebra import BOSON
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates, groundstate


def test_harmonic_oscillator_ladder():
    basis = enumerate_basis([0], BOSON, n_max=5)
    H = bdag(0) * b(0)
    energies, _ = eigenstates(H, basis, k=5)
    assert np.allclose(np.sort(energies), [0.0, 1.0, 2.0, 3.0, 4.0])


def test_displaced_oscillator_groundstate_energy():
    alpha = 0.7
    basis = enumerate_basis([0], BOSON, n_max=20)
    H = bdag(0) * b(0) + alpha * (bdag(0) + b(0))
    e0, _ = groundstate(H, basis)
    assert e0 == pytest.approx(-(alpha**2), abs=1e-6)


def test_two_mode_hopping_single_boson():
    t = 1.3
    basis = enumerate_basis([0, 1], BOSON, n_max=1)
    H = -t * (bdag(0) * b(1) + bdag(1) * b(0))
    energies, _ = eigenstates(H, basis, k=2)
    assert -t == pytest.approx(np.min(energies), abs=1e-10)
