# operator-algebra

## What is operator-algebra?

Performs operator algebra on fermions, bosons, Pauli operators, and more.

See [uv](https://docs.astral.sh/uv) for install instructions on `uv`.

## Install
```bash
uv sync
```

## Symbolic algebra
```bash
uv sync --extra sympy
```

## Solvers
Diagonalize, time-evolve, find ground-states, and find thermal states of operators.
```bash
uv sync --extra solve
```

## Tests
Remove any `--extra` not interested in testing.
```bash
uv sync --group test --extra sympy --extra solve
uv run pytest
```

## Dev
```bash
uv sync --group all --extra sympy --extra solve
```
