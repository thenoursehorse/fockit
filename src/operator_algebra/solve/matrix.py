"""Build sparse matrix representations of operators in a Fock basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from operator_algebra._types import as_complex
from operator_algebra.inspection import modes as op_modes
from operator_algebra.operators import OpSum
from operator_algebra.solve.basis import Basis
from operator_algebra.solve.states import apply_op

type SparseMatrix = sp.spmatrix
type CSRMatrix = sp.csr_matrix
type ComplexArray = NDArray[np.complex128]


def to_matrix(op: OpSum, basis: Basis) -> CSRMatrix:
    """Build the sparse matrix representation of ``op`` in ``basis``.

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    if type(op.rules) is not type(basis.rules):
        msg = f"Operator rules {op.rules!r} do not match basis rules {basis.rules!r}."
        raise ValueError(msg)
    for mode in op_modes(op):
        if mode not in basis.mode_index:
            msg = f"Operator references mode {mode!r} which is not in the basis."
            raise ValueError(msg)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    rules = basis.rules
    mode_index = basis.mode_index
    n_max = basis.n_max
    lookup = basis.lookup

    for j, state in enumerate(basis.states):
        for mono, coeff in op.terms.items():
            amp: complex = as_complex(coeff)
            cur = state
            for op_single in reversed(mono.ops):
                res = apply_op(rules, op_single, mode_index[op_single.mode], cur, n_max)
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
) -> ComplexArray:
    """Compute ``⟨ψ|op|ψ⟩`` for each row of ``states``.

    ``states`` has shape ``(n, dim)``. Returns shape ``(n,)``.
    """
    op_mat = to_matrix(op, basis)
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
