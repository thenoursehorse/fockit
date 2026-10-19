# fockit - Quantum many-body operators and solvers

Copyright (C) 2026 H. L. Nourse

## What is fockit?

Build many-body models, write your own operator algebras, and test algorithms
on them. Quick and simple.

Comes with fermions, bosons and Pauli operators, and you can add your own.
Simple exact solvers check your ideas on small systems.

Particles must have abelian statistics.

## Where to start?

The example below builds a model and solves it; the sections after it add your
own algebra and open systems. More examples are in `examples/` and in the tests:

- `tests/algebra/` for the operator algebra.
- `tests/hilbert/` for Hilbert spaces and matrices; `test_symmetries.py` there
  for symmetry sectors (momentum, reflection, spin flip, spin inversion,
  particle-hole, point groups).
- `tests/solve/` for the solvers.
- `examples/lindblad.py` for open systems, with weak and strong symmetries.

## Example

An electron hopping between two sites. Each site has a phonon and a spin.

```python
from fockit.algebra import b, bdag, c, cdag, nc, sigma_x, sigma_z
from fockit.hilbert import (
    BosonSpace, FermionSpace, PauliSpace, build_basis, to_matrix,
)
from fockit.solve import groundstate

# The operator. Modes are labelled by int, float, str, bool, None, an Enum,
# a frozen dataclass, or a tuple of these, e.g. cdag(0, "up")
H = -(cdag(0) * c(1) + cdag(1) * c(0)) + 0.2 * sigma_z(0) * sigma_z(1)
for i in (0, 1):
    H += bdag(i) * b(i) + 0.5 * nc(i) * (b(i) + bdag(i)) + 0.3 * nc(i) * sigma_x(i)

# A symmetry reduced basis H acts on
basis = build_basis(
    {FermionSpace(): [0, 1], BosonSpace(n_max=6): [0, 1], PauliSpace(): [0, 1]},
    diagonal_symmetries={nc(0) + nc(1): 1},               # one electron
    permutation_symmetries=[({0: 1, 1: 0}, 1)],           # even under swapping the sites
    onsite_symmetries=[([sigma_x(0), sigma_x(1)], 1)],    # even under flipping both spins
)

# The matrix representation of H in the above basis
H_mat = to_matrix(H, basis)

# The ground state
energy, state = groundstate(H_mat)
```

## Your own algebra

Give the rules for swapping two neighbouring operators, which order is normal,
and the adjoint. For example Majorana fermions:

```python
from dataclasses import dataclass

from fockit.algebra import Monomial, Op, OpSum, SwapResult, mode_sort_key, normal_order


@dataclass(frozen=True)
class MajoranaRules:
    """γ_i γ_j + γ_j γ_i = 2 δ_ij."""

    def swap(self, left: Op, right: Op) -> SwapResult:
        if left == right:
            return SwapResult(terms=((1, ()),))  # γ γ = 1
        return SwapResult(terms=((-1, (right, left)),))  # anticommute

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        return mode_sort_key(left.mode) < mode_sort_key(right.mode)

    def dagger(self, op: Op) -> Op:
        return op


def gamma(mode: int) -> OpSum:
    return OpSum({Monomial((Op(mode, "γ", MajoranaRules()),)): 1})


normal_order(gamma(1) * gamma(0) * gamma(1))  # -γ(0)
normal_order((gamma(0) + gamma(1)) ** 2)  # 2
```

`examples/majorana.py` solves a Majorana model with it, and `examples/anyons.py`
also gives anyons their own Hilbert space.

## Open systems

Lindblad master equations work the same way. `left(A)` multiplies the density
matrix rho from the left (rho -> A rho) and `right(B)` from the right
(rho -> rho B), so the generator is an ordinary operator expression.

To get a matrix, rho is vectorized: written as a vector of its elements rho_mn,
in Liouville space. `doubled` adds a copy of every mode to hold the column index
n, so each basis state is a pair (m, n) and stands for the matrix |m><n|. The
generator, a superoperator, then becomes a sparse matrix acting on that
vector. Symmetries and your own algebras work as
before.

```python
import numpy as np

from fockit.algebra import c, cdag, left, liouvillian, nc, right
from fockit.hilbert import FermionSpace, build_basis, doubled, to_matrix, trace_vector
from fockit.solve import slowest_modes, steady_state

sites = range(6)
H = -sum(cdag((i + 1) % 6) * c(i) + cdag(i) * c((i + 1) % 6) for i in sites)
N = sum(nc(i) for i in sites)

# particles leak out of every site and are pumped back in
jumps = [np.sqrt(0.6) * c(i) for i in sites] + [np.sqrt(0.3) * cdag(i) for i in sites]
generator = liouvillian(H, jumps)

# a basis for the density matrix rho, with symmetries as before
basis = build_basis(
    doubled({FermionSpace(): sites}),
    diagonal_symmetries={left(N) - right(N): 0},
    permutation_symmetries=[({i: (i + 1) % 6 for i in sites}, 1)],
)
L = to_matrix(generator, basis)

# the identity matrix as a vector: trace @ rho is Tr(rho)
trace = trace_vector(basis)
# the steady state, normalized to Tr(rho) = 1
rho = steady_state(L, trace)
# Tr(N rho), which is 2
filling = trace @ (to_matrix(left(N), basis) @ rho)
# the 3 slowest-decaying modes; their decay rates are -rates.real
rates, _ = slowest_modes(L, k=3)
```

The jumps here change N, but on both sides of rho at once, so only
`left(N) - right(N)` is conserved (a weak symmetry). Jumps that commute with N,
like dephasing, also conserve `left(N)` and `right(N)` separately (a strong
symmetry). `expm_multiply_evolve(1j * L, rho, times)` evolves rho in time, and
`density_matrix` turns the vector back into a matrix.

## Dependencies

- [python](https://www.python.org/) version `>= 3.14`
- [numpy](https://numpy.org/) and [scipy](https://scipy.org/)
- [sympy](https://www.sympy.org/) for symbolic coefficients (optional)

## Installation

Clone source

```shell
git clone https://github.com/thenoursehorse/fockit
cd fockit
```

Install with [uv](https://docs.astral.sh/uv)

```shell
uv sync
```

Add `--extra sympy` for symbolic coefficients.

## Tests

```shell
uv sync --extra sympy
uv run pytest
```

## Development

After installing as for the tests, set up the pre-commit hooks:

```shell
uv run pre-commit install
```

## AI use

This code was written with AI coding assistants (Claude Code and Codex). All of
it is reviewed and tested.
