"""The scripts in examples/ run and give the right physics."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from fockit.algebra import b, bdag, c, cdag, commutator, normal_order, zero
from fockit.hilbert import BosonSpace, FermionSpace, enumerate_basis, to_matrix
from fockit.solve import eigh_full, groundstate

EXAMPLES = Path(__file__).parents[1] / "examples"


def run(name):
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))


def test_hubbard_symmetric_ground_state_is_the_true_one():
    ex = run("hubbard")
    plain_energy, _ = groundstate(to_matrix(ex["H"], ex["plain"]))
    assert ex["energy"] == pytest.approx(plain_energy)
    assert 0 < ex["double"] < 0.25
    assert ex["spin_flipped"] > ex["energy"]


def test_heisenberg_symmetric_ground_state_is_the_true_one():
    ex = run("heisenberg")
    plain_energy, _ = groundstate(to_matrix(ex["H"], ex["plain"]))
    assert ex["energy"] == pytest.approx(plain_energy)
    assert ex["triplet"] > ex["energy"]
    # total Sz is zero, so the correlations of site 0 with every site add to zero
    assert sum(ex["correlations"]) == pytest.approx(0, abs=1e-10)


def test_jaynes_cummings_hubbard_polaritons_match_the_exact_bands():
    ex = run("mixed")
    L, omega, delta, g, J = ex["L"], ex["omega"], ex["delta"], ex["g"], ex["J"]
    for k, (lower, upper) in enumerate(ex["bands"]):
        photon = omega - 2 * J * np.cos(2 * np.pi * k / L)
        middle = -L * delta / 2 + (photon + delta) / 2
        split = np.hypot((photon - delta) / 2, g)
        assert [lower, upper] == pytest.approx([middle - split, middle + split])


def test_kitaev_chain_has_a_majorana_zero_mode():
    energies = run("majorana")["energies"]
    assert energies[:2] == pytest.approx([-2.5, -2.5])
    assert energies[2] > energies[0] + 0.5


def test_anyons_are_bosons_at_zero_and_fermions_at_pi():
    ex = run("anyons")
    a, adag, AnyonSpace = ex["a"], ex["adag"], ex["AnyonSpace"]
    sites = 4

    def chain(op, opdag, U):
        hop = -sum(
            opdag(j + 1) * op(j) + opdag(j) * op(j + 1) for j in range(sites - 1)
        )
        return hop + 0.5 * U * sum(
            opdag(j) * opdag(j) * op(j) * op(j) for j in range(sites)
        )

    def spectrum(H, space):
        return eigh_full(to_matrix(H, enumerate_basis({space: range(sites)})))[0]

    anyons = spectrum(
        chain(lambda j: a(j, 0.0), lambda j: adag(j, 0.0), U=1.0), AnyonSpace(0.0, 2)
    )
    bosons = spectrum(chain(b, bdag, U=1.0), BosonSpace(2))
    assert anyons == pytest.approx(bosons)

    anyons = spectrum(
        chain(lambda j: a(j, np.pi), lambda j: adag(j, np.pi), U=0.0),
        AnyonSpace(np.pi, 1),
    )
    fermions = spectrum(chain(c, cdag, U=0.0), FermionSpace())
    assert anyons == pytest.approx(fermions)


def test_anyon_algebra_matches_its_hilbert_space():
    # normal ordering with the swap rules must not change the matrix; check on
    # states far from the n_max cutoff, where truncation doesn't matter
    ex = run("anyons")
    a, adag, AnyonSpace = ex["a"], ex["adag"], ex["AnyonSpace"]
    theta, n_max = 0.7, 4
    basis = enumerate_basis({AnyonSpace(theta, n_max): range(3)})
    far = basis.levels.max(axis=1) <= n_max - 3
    products = [
        a(0, theta) * adag(2, theta) * a(1, theta),
        adag(1, theta) * a(0, theta) * adag(0, theta),
        a(2, theta) * a(0, theta) * adag(1, theta),
        adag(2, theta) * adag(0, theta) * a(2, theta),
    ]
    for product in products:
        ordered = to_matrix(normal_order(product), basis).toarray()[:, far]
        direct = to_matrix(product, basis).toarray()[:, far]
        assert ordered == pytest.approx(direct)


def test_pauli_strings_reproduce_the_fermion_ground_energy():
    ex = run("pauli_strings")
    assert ex["energy"] == pytest.approx(ex["fermion_energy"])
    for group in ex["groups"]:
        for first in group:
            for second in group:
                assert normal_order(commutator(first, second)) == zero()
