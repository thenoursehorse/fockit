"""Hubbard ring at half filling: ground energy, double occupancy and spin gap."""

from fockit import c, cdag, n
from fockit.solve import (
    FermionSpace,
    enumerate_basis,
    expand_state,
    expectations,
    groundstate,
    to_matrix,
)

L, t, U = 6, 1.0, 4.0
spins = ("up", "dn")
modes = [(i, s) for i in range(L) for s in spins]

H = -t * sum(
    cdag((i + 1) % L, s) * c(i, s) + cdag(i, s) * c((i + 1) % L, s)
    for i in range(L)
    for s in spins
)
H += U * sum(n(i, "up") * n(i, "dn") for i in range(L))


def sector(n_up, n_dn, **symmetries):
    """States with ``n_up`` up and ``n_dn`` down electrons."""
    numbers = {
        sum(n(i, "up") for i in range(L)): n_up,
        sum(n(i, "dn") for i in range(L)): n_dn,
    }
    return enumerate_basis(
        {FermionSpace(): modes}, diagonal_symmetries=numbers, **symmetries
    )


# the ground state has zero momentum and is odd under swapping up and down
shift = {(i, s): ((i + 1) % L, s) for i, s in modes}
flip = {(i, s): (i, "dn" if s == "up" else "up") for i, s in modes}
basis = sector(L // 2, L // 2, permutation_symmetries=[(shift, 1), (flip, -1)])
energy, psi = groundstate(to_matrix(H, basis))

# n_up n_dn on one site isn't symmetric, so measure it in the plain basis
plain = sector(L // 2, L // 2)
psi_plain = expand_state(psi, basis, plain)
double = expectations(to_matrix(n(0, "up") * n(0, "dn"), plain), psi_plain).real

# spin gap: the lowest state with one spin flipped
spin_flipped, _ = groundstate(to_matrix(H, sector(L // 2 + 1, L // 2 - 1)))

print(f"{basis.dim} symmetric states instead of {plain.dim}")
print(f"ground energy per site {energy / L:.6f}")
print(f"double occupancy {double:.6f} (0.25 without interactions)")
print(f"spin gap {spin_flipped - energy:.6f}")
