"""Open fermion ring: weak and strong symmetries of a Lindblad master equation.

``left(A)`` is rho -> A rho and ``right(B)`` is rho -> rho B. rho is vectorized
(written as a vector of its elements rho_mn, in Liouville space): ``doubled``
adds a copy of every mode to hold the column index n, so each basis state
stands for the matrix |m><n|.
Symmetries work as they do for states.

Weak symmetry: the jumps change the particle number N on both sides of rho
(c rho c†), so only ``left(N) - right(N)`` is conserved. Strong symmetry: the
jumps commute with N, so ``left(N)`` and ``right(N)`` are each conserved.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fockit.algebra import OpSum, c, cdag, left, liouvillian, nc, right
from fockit.hilbert import (
    FermionSpace,
    build_basis,
    doubled,
    state_index,
    to_matrix,
    trace_vector,
)
from fockit.solve import expm_multiply_evolve, slowest_modes, steady_state

L, t, V = 5, 1.0, 0.7
loss, pump, dephasing = 0.6, 0.3, 0.4

H = -t * sum(cdag((i + 1) % L) * c(i) + cdag(i) * c((i + 1) % L) for i in range(L))
H += V * sum(nc(i) * nc((i + 1) % L) for i in range(L))
N = sum(nc(i) for i in range(L))
spaces = doubled({FermionSpace(): range(L)})
shift = {i: (i + 1) % L for i in range(L)}


def slowest(generator: OpSum, **symmetries: Any) -> NDArray[np.complex128]:
    """The two eigenvalues with the largest real part in one sector."""
    basis = build_basis(spaces, **symmetries)
    if not basis.dim:
        return np.empty(0, dtype=complex)
    values, _ = slowest_modes(to_matrix(generator, basis), k=2)
    return values


def steady_states(values: NDArray[np.complex128]) -> int:
    return int(np.sum(np.abs(values) < 1e-9))


# weak: particles leak out and are pumped in on every site, the same everywhere
jumps = [np.sqrt(loss) * c(i) for i in range(L)]
jumps += [np.sqrt(pump) * cdag(i) for i in range(L)]
weak = liouvillian(H, jumps)
sectors = {
    (q, k): slowest(
        weak,
        diagonal_symmetries={left(N) - right(N): q},
        permutation_symmetries=[(shift, np.exp(2j * np.pi * k / L))],
    )
    for q in range(-L, L + 1)
    for k in range(L)
}
weak_steady = [sector for sector, values in sectors.items() if steady_states(values)]
decays = {
    sector: -values.real[np.abs(values) > 1e-9].max()
    for sector, values in sectors.items()
    if np.any(np.abs(values) > 1e-9)
}
gap = min(decays.values())
# sectors q and -q hold rho and its Hermitian conjugate, so they come in pairs
slowest_sectors = [sector for sector, rate in decays.items() if rate - gap < 1e-9]

# the steady state, and the evolution from the empty ring towards it
basis = build_basis(
    spaces,
    diagonal_symmetries={left(N) - right(N): 0},
    permutation_symmetries=[(shift, 1)],
)
generator = to_matrix(weak, basis)
# the identity matrix as a vector: trace @ rho is Tr(rho)
trace = trace_vector(basis)
number = to_matrix(left(N), basis)
rho = steady_state(generator, trace)
filling = (trace @ (number @ rho)).real / L
empty = np.zeros(basis.dim, dtype=complex)
empty[state_index(basis, np.zeros(2 * L, dtype=int))] = 1
times = np.linspace(0, 4, 5)
# expm_multiply_evolve applies e^{-iHt}, so H = i L gives e^{Lt}
history = expm_multiply_evolve(1j * generator, empty, times)
fillings = (history @ number.T @ trace).real / L

print("Weak symmetry: loss and pump, sectors (q, K) = (left(N) - right(N), momentum)")
print(f"  {4**L} states in all, the steady state's sector has {basis.dim}")
print(f"  steady state in sector {weak_steady}")
print(f"  particles per site {filling:.6f}, exactly pump/(loss + pump)")
print(f"  slowest decay rate {gap:.6f}, in sectors {slowest_sectors}")
print("  particles per site from an empty ring:")
for time, value in zip(times, fillings, strict=True):
    print(f"    t = {time:.1f}: {value:.6f}")

# strong: dephasing measures each site's occupation and never changes N
strong = liouvillian(H, [np.sqrt(dephasing) * nc(i) for i in range(L)])
strong_steady = np.array(
    [
        [
            steady_states(
                slowest(strong, diagonal_symmetries={left(N): n, right(N): m})
            )
            for m in range(L + 1)
        ]
        for n in range(L + 1)
    ]
)

print("\nStrong symmetry: dephasing, steady states per sector (left(N), right(N))")
print("  left(N) \\ right(N) " + " ".join(f"{m}" for m in range(L + 1)))
for n, row in enumerate(strong_steady):
    print(f"  {n:7d}            " + " ".join(f"{count}" for count in row))
print("  one steady state for each particle number")
