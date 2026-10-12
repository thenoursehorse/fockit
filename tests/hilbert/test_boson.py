"""Tests against analytic boson spectra."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fockit.algebra import b, bdag, normal_order
from fockit.hilbert import BosonSpace, enumerate_basis, expectations, to_matrix
from fockit.solve import eigenstates, groundstate


def test_harmonic_oscillator_ladder():
    basis = enumerate_basis({BosonSpace(n_max=5): [0]})
    H = bdag(0) * b(0)
    energies, _ = eigenstates(to_matrix(H, basis), k=5)
    assert np.allclose(np.sort(energies), [0.0, 1.0, 2.0, 3.0, 4.0])


def test_displaced_oscillator_groundstate_energy():
    alpha = 0.7
    basis = enumerate_basis({BosonSpace(n_max=20): [0]})
    H = bdag(0) * b(0) + alpha * (bdag(0) + b(0))
    e0, _ = groundstate(to_matrix(H, basis))
    assert e0 == pytest.approx(-(alpha**2), abs=1e-6)


def test_two_mode_hopping_single_boson():
    t = 1.3
    basis = enumerate_basis({BosonSpace(n_max=1): [0, 1]})
    H = -t * (bdag(0) * b(1) + bdag(1) * b(0))
    energies, _ = eigenstates(to_matrix(H, basis), k=2)
    assert -t == pytest.approx(np.min(energies), abs=1e-10)


def test_oscillator_groundstate_position_moments():
    power = 12
    basis = enumerate_basis({BosonSpace(n_max=power): [0]})
    _, vacuum = groundstate(to_matrix(bdag(0) * b(0), basis))
    x_n = normal_order((b(0) + bdag(0)) ** power)
    expected = math.prod(range(power - 1, 0, -2))  # (n-1)!!
    assert expectations(to_matrix(x_n, basis), vacuum) == pytest.approx(expected)
