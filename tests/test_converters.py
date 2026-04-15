"""Tests for converters to external formats."""

from itertools import product

import pytest


def test_kanamori_3orbital():
    import numpy as np

    from operator_algebra import c, cdag, n, normal_order
    from operator_algebra.converters import to_triqs

    pytest.importorskip("triqs")
    from triqs.operators.util.hamiltonians import h_int_kanamori

    n_orb = 3
    U_val = 4.0
    J_val = 0.5
    Uprime_val = U_val - 2 * J_val
    spins = ["up", "down"]
    orbs = range(n_orb)

    U = np.full((n_orb, n_orb), U_val)
    np.fill_diagonal(U, 0)
    Uprime = np.full((n_orb, n_orb), Uprime_val)
    np.fill_diagonal(Uprime, 0)

    # density-density; diagonal=0 naturally excludes same-mode
    H = sum(
        0.5
        * (float(U[a1, a2]) if s1 == s2 else float(Uprime[a1, a2]))
        * n(s1, a1)
        * n(s2, a2)
        for s1, s2 in product(spins, spins)
        for a1, a2 in product(orbs, orbs)
    )
    # spin-flip
    H = H + sum(
        -0.5 * J_val * cdag(s1, a1) * c(s2, a1) * cdag(s2, a2) * c(s1, a2)
        for s1, s2 in product(spins, spins)
        if s1 != s2
        for a1, a2 in product(orbs, orbs)
        if a1 != a2
    )
    # pair-hopping
    H = H + sum(
        0.5 * J_val * cdag(s1, a1) * cdag(s2, a1) * c(s2, a2) * c(s1, a2)
        for s1, s2 in product(spins, spins)
        if s1 != s2
        for a1, a2 in product(orbs, orbs)
        if a1 != a2
    )
    H = normal_order(H)

    expected = h_int_kanamori(spins, n_orb, U, Uprime, J_val, off_diag=True)
    assert to_triqs(H) == expected
