"""Build sparse matrix representations of operators in a product basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from operator_algebra._types import as_complex
from operator_algebra.operators import OpSum
from operator_algebra.solve.basis import Basis, apply_monomial, check_modes
from operator_algebra.solve.spaces import IntArray

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

    rows: list[IntArray] = [np.empty(0, dtype=np.int64)]
    cols: list[IntArray] = [np.empty(0, dtype=np.int64)]
    data: list[ComplexArray] = [np.empty(0, dtype=complex)]
    for mono, coeff in op.terms.items():
        source, new, amp = apply_monomial(
            mono, basis.levels, basis.mode_index, basis.spaces, basis.starts
        )
        codes = new @ basis.strides
        pos = np.minimum(np.searchsorted(basis.codes, codes), basis.dim - 1)
        keep = basis.codes[pos] == codes
        rows.append(pos[keep])
        cols.append(source[keep])
        data.append(as_complex(coeff) * amp[keep])

    mat = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(basis.dim, basis.dim),
    ).tocsr()
    mat.sum_duplicates()
    return mat


def expectations(op: SparseMatrix, states: ComplexArray) -> ComplexArray | complex:
    """Compute ``⟨ψ|op|ψ⟩`` for each row of ``states``.

    ``states`` has shape ``(dim,)`` for a single state (returns a scalar)
    or ``(n, dim)`` for a batch (returns shape ``(n,)``).
    """
    if states.ndim == 1:
        return complex(states.conj() @ (op @ states))
    op_states = (op @ states.T).T
    return np.asarray(np.sum(states.conj() * op_states, axis=1))


def matrix_elements(op: SparseMatrix, states: ComplexArray) -> ComplexArray:
    """Return ``O_{mn} = ⟨m|op|n⟩`` for the column states in ``states``.

    ``states`` has shape ``(dim, k)`` (columns are the bra/ket states).
    """
    return np.asarray(states.conj().T @ (op @ states))
