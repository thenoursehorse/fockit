"""Majorana fermions: define the algebra, then solve the Kitaev chain.

A single Majorana has no states of its own; two of them make one fermion mode.
So to solve a Majorana model, substitute fermions for each pair.
"""

from dataclasses import dataclass

from fockit import c, cdag, normal_order, substitute
from fockit.algebra import SwapResult, mode_sort_key
from fockit.operators import Monomial, Op, OpSum
from fockit.solve import FermionSpace, eigh_full, enumerate_basis, to_matrix


@dataclass(frozen=True)
class MajoranaRules:
    """γ_i γ_j + γ_j γ_i = 2 δ_ij."""

    def swap(self, left, right):
        if left == right:
            return SwapResult(terms=((1, ()),))  # γ γ = 1
        return SwapResult(terms=((-1, (right, left)),))  # anticommute

    def is_normal_ordered_pair(self, left, right):
        return mode_sort_key(left.mode) < mode_sort_key(right.mode)

    def dagger(self, op):
        return op


def gamma(mode):
    return OpSum({Monomial((Op(mode, "γ", MajoranaRules()),)): 1})


print(normal_order(gamma(1) * gamma(0) * gamma(1)))  # -γ(0)

# Kitaev chain at its sweet spot: the end Majoranas gamma(0) and gamma(2L - 1)
# are left out, so the ground state is twofold degenerate
L = 6
H = sum(0.5j * gamma(2 * j + 1) * gamma(2 * j + 2) for j in range(L - 1))

pairs = {}
for j in range(L):
    pairs[gamma(2 * j)] = c(j) + cdag(j)
    pairs[gamma(2 * j + 1)] = 1j * (cdag(j) - c(j))

basis = enumerate_basis({FermionSpace(): range(L)})
energies, _ = eigh_full(to_matrix(substitute(H, pairs), basis))
print(energies[:4])  # -2.5, -2.5, -1.5, -1.5
