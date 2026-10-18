"""Open fermion ring: weak and strong symmetries of a Lindblad master equation.

The density matrix rho is a state on twice the modes: ``left(A)`` multiplies it
from the left (rho -> A rho) and ``right(B)`` from the right (rho -> rho B).
``doubled`` adds the modes ``right`` acts on, so ``build_basis`` and
``to_matrix`` work on rho as on any state, symmetries included.

Weak symmetry: each jump changes the particle number N, but on both sides of
rho at once (c rho c†), so ``left(N) - right(N)`` is conserved. Strong
symmetry: the jumps commute with N, so ``left(N)`` and ``right(N)`` are each
conserved. Using a strong sector when the symmetry is only weak silently drops
the jumps that leave it.
"""

from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from fockit.algebra import OpSum, c, cdag, left, liouvillian, nc, right
from fockit.hilbert import FermionSpace, build_basis, doubled, to_matrix

L, t, V = 5, 1.0, 0.7
loss, pump, dephasing = 0.6, 0.3, 0.4

H = -t * sum(cdag((i + 1) % L) * c(i) + cdag(i) * c((i + 1) % L) for i in range(L))
H += V * sum(nc(i) * nc((i + 1) % L) for i in range(L))
N = sum(nc(i) for i in range(L))
spaces = doubled({FermionSpace(): range(L)})
shift = {i: (i + 1) % L for i in range(L)}


def eigenvalues(generator: OpSum, **symmetries: Any) -> NDArray[np.complex128]:
    """Eigenvalues of the generator in one sector.

    fockit's solvers assume a Hermitian matrix, which the generator isn't.
    """
    basis = build_basis(spaces, **symmetries)
    return scipy.linalg.eigvals(to_matrix(generator, basis).toarray())


def steady_states(values: NDArray[np.complex128]) -> int:
    return int(np.sum(np.abs(values) < 1e-9))


# weak: particles leak out and are pumped in on every site, the same everywhere
jumps = [np.sqrt(loss) * c(i) for i in range(L)]
jumps += [np.sqrt(pump) * cdag(i) for i in range(L)]
weak = liouvillian(H, jumps)
sectors = {
    (q, k): eigenvalues(
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
slowest = [sector for sector, rate in decays.items() if rate - gap < 1e-9]

print("Weak symmetry: loss and pump, sectors (q, K) = (N_left - N_right, momentum)")
largest = max(len(values) for values in sectors.values())
print(f"  {4**L} states in all, the largest sector has {largest}")
print(f"  steady state in sector {weak_steady}")
print(f"  slowest decay rate {gap:.6f}, in sectors {slowest}")

# strong: dephasing measures each site's occupation and never changes N
strong = liouvillian(H, [np.sqrt(dephasing) * nc(i) for i in range(L)])
strong_steady = np.array(
    [
        [
            steady_states(
                eigenvalues(strong, diagonal_symmetries={left(N): n, right(N): m})
            )
            for m in range(L + 1)
        ]
        for n in range(L + 1)
    ]
)

print("\nStrong symmetry: dephasing, steady states in each (N_left, N_right) sector")
print("  N_left \\ N_right " + " ".join(f"{m}" for m in range(L + 1)))
for n, row in enumerate(strong_steady):
    print(f"  {n:6d}           " + " ".join(f"{count}" for count in row))
print("  one steady state for each particle number")
