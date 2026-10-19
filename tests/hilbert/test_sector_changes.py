"""Operators that take states from one symmetry sector to another."""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fockit.algebra import OpSum, c, cdag, nc, sigma_x, sigma_z, zero
from fockit.hilbert import (
    Basis,
    FermionSpace,
    PauliSpace,
    build_basis,
    expand_state,
    to_matrix,
)
from fockit.solve import groundstate

SITES = 6


def momentum(k: int) -> complex:
    return cmath.exp(2j * cmath.pi * k / SITES)


def shift(spins: tuple[str, ...] = ()) -> dict[Any, Any]:
    if not spins:
        return {i: (i + 1) % SITES for i in range(SITES)}
    return {(i, s): ((i + 1) % SITES, s) for i in range(SITES) for s in spins}


def fourier(op: Callable[[int], OpSum], k: int) -> OpSum:
    """``Σ_j e^(-2πi kj/L) op(j) / √L``: carries momentum ``k``."""
    out = zero()
    for j in range(SITES):
        out += cmath.exp(-2j * cmath.pi * k * j / SITES) / math.sqrt(SITES) * op(j)
    return out


def spinless(particles: int, **symmetries: Any) -> Basis:
    N = sum(nc(i) for i in range(SITES))
    return build_basis(
        {FermionSpace(): range(SITES)}, diagonal_symmetries={N: particles}, **symmetries
    )


def test_c_k_between_momentum_sectors_matches_the_plain_basis() -> None:
    plain3, plain2 = spinless(3), spinless(2)
    plain = to_matrix(fourier(c, 1), plain2, source=plain3).toarray()
    for K in range(SITES):
        # removing momentum 1 moves the translation eigenvalue by e^(2πi/L)
        source = spinless(3, permutation_symmetries=[(shift(), momentum(K))])
        target = spinless(2, permutation_symmetries=[(shift(), momentum(K + 1))])
        reduced = to_matrix(fourier(c, 1), target, source=source).toarray()
        into = expand_state(np.eye(source.dim), source, plain3)
        out_of = expand_state(np.eye(target.dim), target, plain2)
        assert reduced == pytest.approx(out_of.conj().T @ plain @ into)


def test_products_of_sector_changes_give_the_operator_in_one_sector() -> None:
    # c†(0) c(0) through N - 1, and c_k† c_k through momentum K + k
    three, two = spinless(3), spinless(2)
    through = to_matrix(cdag(0), three, source=two) @ to_matrix(c(0), two, source=three)
    assert through.toarray() == pytest.approx(to_matrix(nc(0), three).toarray())

    source = spinless(3, permutation_symmetries=[(shift(), momentum(0))])
    target = spinless(2, permutation_symmetries=[(shift(), momentum(2))])
    ck, ck_dag = fourier(c, 2), fourier(cdag, -2)
    through = to_matrix(ck_dag, source, source=target) @ to_matrix(
        ck, target, source=source
    )
    assert through.toarray() == pytest.approx(to_matrix(ck_dag * ck, source).toarray())


def test_momentum_distribution_of_the_half_filled_hubbard_ring() -> None:
    spins = ("up", "dn")
    modes = [(i, s) for i in range(SITES) for s in spins]
    H: OpSum | complex = -sum(
        cdag((i + 1) % SITES, s) * c(i, s) + cdag(i, s) * c((i + 1) % SITES, s)
        for i in range(SITES)
        for s in spins
    )
    H += 4 * sum(nc(i, "up") * nc(i, "dn") for i in range(SITES))

    def sector(n_up: int, K: int) -> Basis:
        numbers = {
            sum(nc(i, "up") for i in range(SITES)): n_up,
            sum(nc(i, "dn") for i in range(SITES)): SITES // 2,
        }
        return build_basis(
            {FermionSpace(): modes},
            diagonal_symmetries=numbers,
            permutation_symmetries=[(shift(spins), momentum(K))],
        )

    ground = sector(SITES // 2, 0)
    _, psi = groundstate(to_matrix(H, ground))
    occupations = []
    for k in range(SITES):
        ck = fourier(lambda j: c(j, "up"), k)
        removed = to_matrix(ck, sector(SITES // 2 - 1, k), source=ground) @ psi
        occupations.append(np.vdot(removed, removed).real)
    assert sum(occupations) == pytest.approx(SITES // 2)
    # particle-hole symmetry at half filling: n_k + n_(k+π) = 1
    half = SITES // 2
    for k in range(half):
        assert occupations[k] + occupations[k + half] == pytest.approx(1)


def test_spin_density_wave_flips_spin_inversion_parity() -> None:
    Sz = sum(sigma_z(i) for i in range(SITES))
    flip = [sigma_x(i) for i in range(SITES)]
    wave = fourier(sigma_z, 2)

    def sector(K: int, parity: int) -> Basis:
        return build_basis(
            {PauliSpace(): range(SITES)},
            diagonal_symmetries={Sz: 0},
            permutation_symmetries=[(shift(), momentum(K))],
            onsite_symmetries=[(flip, parity)],
        )

    assert to_matrix(wave, sector(2, -1), source=sector(0, 1)).nnz > 0
    with pytest.raises(ValueError, match="but basis has"):
        to_matrix(wave, sector(2, 1), source=sector(0, 1))


def test_operators_that_cannot_reach_the_basis_are_rejected() -> None:
    source = spinless(3, permutation_symmetries=[(shift(), momentum(0))])
    # one site is a mix of every momentum
    with pytest.raises(ValueError, match="phase"):
        to_matrix(
            c(0),
            spinless(2, permutation_symmetries=[(shift(), momentum(0))]),
            source=source,
        )
    # c removes a particle, so the target can't have the same number
    same_number = spinless(3, permutation_symmetries=[(shift(), momentum(1))])
    with pytest.raises(ValueError, match="outside the basis"):
        to_matrix(fourier(c, 1), same_number, source=source)
