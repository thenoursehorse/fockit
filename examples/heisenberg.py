"""Heisenberg ring: ground energy, spin gap and spin-spin correlations."""

import numpy as np

from fockit import sigma_x, sigma_y, sigma_z
from fockit.solve import (
    PauliSpace,
    enumerate_basis,
    expand_state,
    expectations,
    groundstate,
    to_matrix,
)

L = 16
space = {PauliSpace(): range(L)}

# H = sum_i S_i . S_{i+1}, with S = σ/2
H = 0.25 * sum(
    f(i) * f((i + 1) % L) for i in range(L) for f in (sigma_x, sigma_y, sigma_z)
)
Sz = 0.5 * sum(sigma_z(i) for i in range(L))

# the ground state has Sz = 0, zero momentum, and is even under reflection and
# under flipping every spin
shift = {i: (i + 1) % L for i in range(L)}
reflection = {i: L - 1 - i for i in range(L)}
basis = enumerate_basis(
    space,
    diagonal_symmetries={Sz: 0},
    permutation_symmetries=[(shift, 1), (reflection, 1)],
    onsite_symmetries=[([sigma_x(i) for i in range(L)], 1)],
)
energy, psi = groundstate(to_matrix(H, basis))

# spin gap: the lowest state with Sz = 1
triplet, _ = groundstate(
    to_matrix(H, enumerate_basis(space, diagonal_symmetries={Sz: 1}))
)

# correlations <S^z_0 S^z_r> aren't symmetric, so measure them in the plain basis
plain = enumerate_basis(space, diagonal_symmetries={Sz: 0})
psi_plain = expand_state(psi, basis, plain)
correlations = [
    expectations(to_matrix(0.25 * sigma_z(0) * sigma_z(r), plain), psi_plain).real
    for r in range(L)
]

print(f"{basis.dim} symmetric states instead of {plain.dim}")
print(
    f"ground energy per site {energy / L:.6f} (infinite chain {0.25 - np.log(2):.6f})"
)
print(f"spin gap {triplet - energy:.6f}")
print("<S^z_0 S^z_r>:", np.round(correlations, 4))
