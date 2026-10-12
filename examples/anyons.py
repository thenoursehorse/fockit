"""Abelian anyons on a chain: define the algebra and the Hilbert space, then solve.

Anyons on a lattice are bosons on each site that pick up a phase e^{iθ} when
they pass each other. θ = 0 gives bosons, and θ = π with at most one per site
gives fermions.
"""

import cmath
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fockit.algebra import Monomial, Op, OpSum, SwapResult, mode_sort_key
from fockit.hilbert import build_basis, to_matrix
from fockit.solve import groundstate


class Anyon(StrEnum):
    CREATE = "a†"
    ANNIHILATE = "a"


@dataclass(frozen=True)
class AnyonRules:
    """Bosons on one site; between sites, for j < k.

    a_j a_k = e^{iθ} a_k a_j and a_j a_k† = e^{-iθ} a_k† a_j.
    """

    theta: float

    def swap(self, left, right):
        if left.mode == right.mode:
            return SwapResult(terms=((1, ()), (1, (right, left))))  # a a† = 1 + a† a
        phase = cmath.exp(1j * self.theta)
        if left.kind is not right.kind:
            phase = phase.conjugate()
        if mode_sort_key(left.mode) > mode_sort_key(right.mode):
            phase = phase.conjugate()
        return SwapResult(terms=((phase, (right, left)),))

    def is_normal_ordered_pair(self, left, right):
        if left.kind is not right.kind:
            return left.kind is Anyon.CREATE
        return mode_sort_key(left.mode) <= mode_sort_key(right.mode)

    def dagger(self, op):
        flip = Anyon.CREATE if op.kind is Anyon.ANNIHILATE else Anyon.ANNIHILATE
        return Op(op.mode, flip, self)


@dataclass(frozen=True)
class AnyonSpace:
    """Occupations 0 to n_max; a_j picks up e^{iθ n_k} from each earlier site k."""

    theta: float
    n_max: int

    @property
    def rules(self):
        return AnyonRules(self.theta)

    @property
    def dim(self):
        return self.n_max + 1

    def action(self, op):  # new level and amplitude for each level
        level = np.arange(self.dim)
        if op.kind is Anyon.CREATE:
            amp = np.sqrt(level + 1).astype(complex)
            amp[-1] = 0
            return np.minimum(level + 1, self.n_max), amp
        return np.maximum(level - 1, 0), np.sqrt(level).astype(complex)

    def string_factor(self, op):  # phase from each earlier site, by its level
        sign = 1 if op.kind is Anyon.ANNIHILATE else -1
        return np.exp(sign * 1j * self.theta * np.arange(self.dim))

    def exchange_factor(self):  # only for permutation symmetries; not worked out
        return np.ones((self.dim, self.dim), dtype=complex)


def a(j, theta):
    return OpSum({Monomial((Op(j, Anyon.ANNIHILATE, AnyonRules(theta)),)): 1})


def adag(j, theta):
    return OpSum({Monomial((Op(j, Anyon.CREATE, AnyonRules(theta)),)): 1})


# anyon-Hubbard chain with two anyons, for a few statistics angles
L, U = 6, 1.0
for theta in (0.0, np.pi / 2, np.pi):
    H = -sum(
        adag(j + 1, theta) * a(j, theta) + adag(j, theta) * a(j + 1, theta)
        for j in range(L - 1)
    )
    for j in range(L):
        H += 0.5 * U * adag(j, theta) * adag(j, theta) * a(j, theta) * a(j, theta)
    basis = build_basis(
        {AnyonSpace(theta, n_max=2): range(L)},
        diagonal_symmetries={sum(adag(j, theta) * a(j, theta) for j in range(L)): 2},
    )
    energy, _ = groundstate(to_matrix(H, basis))
    print(f"theta = {theta:.3f}: ground energy {energy:.6f}")
