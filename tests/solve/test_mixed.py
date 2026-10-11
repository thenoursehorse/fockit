"""Tests for bases mixing fermions, bosons and spins."""

from __future__ import annotations

import numpy as np
import pytest

from operator_algebra import (
    b,
    bdag,
    c,
    cdag,
    n,
    normal_order,
    sigma_x,
    sigma_z,
    sm,
    sp,
)
from operator_algebra.solve.basis import enumerate_basis
from operator_algebra.solve.diagonalize import eigenstates
from operator_algebra.solve.matrix import to_matrix
from operator_algebra.solve.spaces import BosonSpace, FermionSpace, PauliSpace


def test_holstein_site_has_polaron_shift():
    eps, omega, g = 0.4, 1.0, 0.5
    H = eps * n(0) + omega * bdag(0) * b(0) + g * n(0) * (b(0) + bdag(0))
    basis = enumerate_basis(
        {FermionSpace(): [0], BosonSpace(n_max=30): [0]}, diagonal_symmetries={n(0): 1}
    )
    energies, _ = eigenstates(to_matrix(H, basis), k=3)
    expected = [eps - g**2 / omega + omega * m for m in range(3)]
    assert energies == pytest.approx(expected, abs=1e-10)


def test_jaynes_cummings_spectrum():
    omega, delta, g = 1.0, 1.3, 0.2
    H = (
        omega * bdag(0) * b(0)
        + 0.5 * delta * sigma_z(0)
        + g * (sp(0) * b(0) + sm(0) * bdag(0))
    )
    excitations = normal_order(bdag(0) * b(0) + sp(0) * sm(0))
    detuning = delta - omega
    for n_exc in range(1, 4):
        basis = enumerate_basis(
            {PauliSpace(): [0], BosonSpace(n_max=5): [0]},
            diagonal_symmetries={excitations: n_exc},
        )
        energies, _ = eigenstates(to_matrix(H, basis), k=2)
        split = 0.5 * np.sqrt(detuning**2 + 4 * g**2 * n_exc)
        mean = omega * (n_exc - 0.5)
        assert energies == pytest.approx([mean - split, mean + split])


def test_fermions_commute_with_other_algebras_as_matrices():
    # bosons and spins come first, so a sign leaking across them would show
    basis = enumerate_basis(
        {BosonSpace(n_max=2): [0], PauliSpace(): [0], FermionSpace(): [0, 1]}
    )
    f = to_matrix(c(1), basis).toarray()
    for other in (b(0), sigma_x(0)):
        o = to_matrix(other, basis).toarray()
        assert f @ o == pytest.approx(o @ f)
    f0 = to_matrix(cdag(0), basis).toarray()
    assert f @ f0 == pytest.approx(-f0 @ f)
