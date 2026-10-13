"""Pauli strings for quantum computing.

Map a fermion model to qubits with Jordan-Wigner, list its Pauli strings, group
the ones that commute (they can be measured together), and get the exact ground
energy as a reference.
"""

from fockit.algebra import (
    OpSum,
    c,
    cdag,
    commutator,
    identity,
    nc,
    normal_order,
    sigma_z,
    sm,
    sp,
    substitute,
    zero,
)
from fockit.hilbert import (
    FermionSpace,
    PauliSpace,
    build_basis,
    expectation,
    to_matrix,
)
from fockit.solve import groundstate

# spinless fermions on an open chain with nearest-neighbour repulsion
L, V = 4, 1.0
hopping = -sum(cdag(j + 1) * c(j) + cdag(j) * c(j + 1) for j in range(L - 1))
repulsion = V * sum((nc(j) - 0.5) * (nc(j + 1) - 0.5) for j in range(L - 1))
H_fermion = hopping + repulsion

# Jordan-Wigner: c_j = Z_0 ... Z_{j-1} σ-_j
jordan_wigner = {}
for j in range(L):
    string = identity()
    for k in range(j):
        string = string * sigma_z(k)
    jordan_wigner[c(j)] = string * sm(j)
    jordan_wigner[cdag(j)] = string * sp(j)
H = normal_order(substitute(H_fermion, jordan_wigner))
print("qubit Hamiltonian:", H)

# group Pauli strings that commute with every other string in their group
strings = [OpSum({mono: 1}) for mono in H.terms]
groups: list[list[OpSum]] = []
for string in strings:
    for group in groups:
        if all(normal_order(commutator(string, other)) == zero() for other in group):
            group.append(string)
            break
    else:
        groups.append([string])
print(f"{len(strings)} strings in {len(groups)} commuting groups")

# exact ground energy, and the expectation of each string in the ground state
basis = build_basis({PauliSpace(): range(L)})
energy, psi = groundstate(to_matrix(H, basis))
fermion_energy, _ = groundstate(
    to_matrix(H_fermion, build_basis({FermionSpace(): range(L)}))
)
print(f"ground energy {energy:.6f} (fermions: {fermion_energy:.6f})")
for mono, coeff in H.terms.items():
    weight = complex(coeff).real
    value = expectation(to_matrix(OpSum({mono: 1}), basis), psi).real
    print(f"  {weight:+.2f} <{OpSum({mono: 1})}> = {weight * value:+.4f}")
