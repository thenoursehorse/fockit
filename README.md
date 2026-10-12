# fockit - Quantum many-body operators and solvers

Copyright (C) 2026 H. L. Nourse

## What is fockit?

Build many-body models, write your own operator algebras, and test algorithms
on them. Quick and simple.

Comes with fermions, bosons and Pauli operators, and you can add your own.
Particles must have abelian statistics. Simple exact solvers check your ideas
on small systems.

## Where to start?

The example below builds a model and solves it; the section after it adds your
own algebra. More examples are in `examples/` and in the tests:

- `tests/algebra/` for the operator algebra.
- `tests/hilbert/` for Hilbert spaces and matrices; `test_symmetries.py` there
  for symmetry sectors (momentum, reflection, spin flip, spin inversion,
  particle-hole, point groups).
- `tests/solve/` for the solvers.

## Example

An electron hopping between two sites. Each site has a phonon and a spin.

```python
from fockit.algebra import b, bdag, c, cdag, nc, sigma_x, sigma_z
from fockit.hilbert import (
    BosonSpace, FermionSpace, PauliSpace, build_basis, to_matrix,
)
from fockit.solve import groundstate

# The operator
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
mat = to_matrix(H, basis)

# The ground state
energy, state = groundstate(mat)
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


normal_order(gamma(1) * gamma(0) * gamma(1))  # -γ(0)
normal_order((gamma(0) + gamma(1)) ** 2)  # 2
```

`examples/majorana.py` solves a Majorana model with it, and `examples/anyons.py`
also gives anyons their own Hilbert space.

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
