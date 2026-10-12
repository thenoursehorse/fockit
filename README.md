# fockit - Quantum many-body operators and solvers

Copyright (C) 2026 H. L. Nourse

## What is fockit?

Build many-body models, write your own operator algebras, and test algorithms
on them. Quick and simple.

Comes with fermions, bosons and Pauli operators, and you can add your own.
Simple exact solvers check your ideas on small systems.

## Where to start?

The tests are examples:

- `tests/test_integration.py` for the operator algebra.
- `tests/solve/` for matrices and solvers.
- `tests/solve/test_symmetries.py` for symmetry sectors (momentum, reflection,
  spin flip, spin inversion, particle-hole, point groups).

## Dependencies

- [python](https://www.python.org/) version `>= 3.12`
- [numpy](https://numpy.org/) and [scipy](https://scipy.org/) for the solvers
- [sympy](https://www.sympy.org/) for symbolic coefficients (optional)

## Installation

Clone source

```shell
git clone https://github.com/thenoursehorse/fockit
cd fockit
```

Install with [uv](https://docs.astral.sh/uv)

```shell
uv sync --extra solve
```

Add `--extra sympy` for symbolic coefficients.

## Tests

```shell
uv sync --group test --extra sympy --extra solve
uv run pytest
```

## Development

```shell
uv sync --group all --extra sympy --extra solve
```
