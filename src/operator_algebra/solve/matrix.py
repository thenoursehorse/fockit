"""Build sparse matrix representations of operators in a Fock basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from operator_algebra._types import as_complex
from operator_algebra.operators import OpSum
from operator_algebra.solve.basis import Basis, check_modes
from operator_algebra.solve.spaces import apply_op

type SparseMatrix = sp.spmatrix
type CSRMatrix = sp.csr_matrix
type ComplexArray = NDArray[np.complex128]


def to_matrix(op: OpSum, basis: Basis) -> CSRMatrix:
    """Build the sparse matrix representation of ``op`` in ``basis``.

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    check_modes(op, basis.mode_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    lookup = basis.lookup

    # resolve each operator's mode position once, not once per state
    terms = []
    for mono, coeff in op.terms.items():
        steps = []
        for op_single in reversed(mono.ops):
            k = basis.mode_index[(op_single.rules, op_single.mode)]
            steps.append((basis.spaces[k], op_single, k, basis.starts[k]))
        terms.append((as_complex(coeff), steps))

    for j, state in enumerate(basis.states):
        for coeff, steps in terms:
            amp: complex = coeff
            cur = state
            for space, op_single, k, start in steps:
                res = apply_op(space, op_single, k, cur, start)
                if res is None:
                    amp = 0j
                    break
                a, cur = res
                amp *= a
            if amp == 0:
                continue
            i = lookup.get(cur)
            if i is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(amp)

    mat = sp.coo_matrix(
        (np.asarray(data, dtype=complex), (rows, cols)),
        shape=(basis.dim, basis.dim),
    ).tocsr()
    mat.sum_duplicates()
    return mat


def expectations(
    op: OpSum,
    states: ComplexArray,
    basis: Basis,
) -> ComplexArray | complex:
    """Compute ``⟨ψ|op|ψ⟩`` for each row of ``states``.

    ``states`` has shape ``(dim,)`` for a single state (returns a scalar)
    or ``(n, dim)`` for a batch (returns shape ``(n,)``).
    """
    op_mat = to_matrix(op, basis)
    if states.ndim == 1:
        return complex(states.conj() @ (op_mat @ states))
    op_states = (op_mat @ states.T).T
    return np.asarray(np.sum(states.conj() * op_states, axis=1))


def matrix_elements(
    op: OpSum,
    states: ComplexArray,
    basis: Basis,
) -> ComplexArray:
    """Return ``O_{mn} = ⟨m|op|n⟩`` for the column states in ``states``.

    ``states`` has shape ``(dim, k)`` (columns are the bra/ket states).
    """
    op_mat = to_matrix(op, basis)
    return np.asarray(states.conj().T @ (op_mat @ states))
