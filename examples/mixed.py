"""Jaynes-Cummings-Hubbard ring: photons hop between cavities, each with an atom.

Bosons (photons) and Pauli operators (atoms) in one model. The excitation number
is conserved, and with one excitation the polariton bands are known exactly.
"""

import numpy as np

from fockit import b, bdag, commutator, drop_zeros, normal_order, sigma_z, sm, sp, zero
from fockit.solve import BosonSpace, PauliSpace, eigh_full, enumerate_basis, to_matrix

L, omega, delta, g, J = 6, 1.0, 1.0, 0.3, 0.2

H = sum(
    omega * bdag(i) * b(i)
    + 0.5 * delta * sigma_z(i)
    + g * (sp(i) * b(i) + sm(i) * bdag(i))
    for i in range(L)
)
H -= J * sum(bdag((i + 1) % L) * b(i) + bdag(i) * b((i + 1) % L) for i in range(L))

excitations = sum(bdag(i) * b(i) + sp(i) * sm(i) for i in range(L))
# drop the float round-off left where terms cancel
change = drop_zeros(normal_order(commutator(H, excitations)), tol=1e-12)
print("excitations conserved:", change == zero())

# cavities and atoms share the labels 0..L-1, so one translation moves both
shift = {i: (i + 1) % L for i in range(L)}
bands = []
for k in range(L):
    basis = enumerate_basis(
        {BosonSpace(n_max=1): range(L), PauliSpace(): range(L)},
        diagonal_symmetries={excitations: 1},
        permutation_symmetries=[(shift, np.exp(2j * np.pi * k / L))],
    )
    energies, _ = eigh_full(to_matrix(H, basis))
    bands.append(energies)

print(" k   lower polariton   upper polariton")
for k, (lower, upper) in enumerate(bands):
    print(f"{k:2d}   {lower:15.6f}   {upper:15.6f}")
