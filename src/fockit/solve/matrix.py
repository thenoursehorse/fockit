"""Build sparse matrix representations of operators in a product basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from fockit._types import as_complex
from fockit.operators import OpSum
from fockit.solve.basis import (
    Basis,
    apply_monomial,
    check_modes,
    check_symmetric,
    representatives,
)
from fockit.solve.spaces import IntArray

type SparseMatrix = sp.spmatrix
type CSRMatrix = sp.csr_matrix
type ComplexArray = NDArray[np.complex128]


def to_matrix(op: OpSum, basis: Basis) -> CSRMatrix:
    """Build the sparse matrix representation of ``op`` in ``basis``.

    With permutation symmetries, ``op`` must commute with them.

    Returns
    -------
    scipy.sparse.csr_matrix
        Real if every matrix element is real, complex otherwise.

    """
    check_modes(op, basis.mode_index)
    check_symmetric(op, basis.permutation_symmetries, basis.onsite_symmetries)
    symmetric = len(basis.group) > 1

    rows: list[IntArray] = [np.empty(0, dtype=np.int64)]
    cols: list[IntArray] = [np.empty(0, dtype=np.int64)]
    data: list[ComplexArray] = [np.empty(0, dtype=complex)]
    for mono, coeff in op.terms.items():
        source, new, amp = apply_monomial(
            mono, basis.levels, basis.mode_index, basis.spaces, basis.starts
        )
        if symmetric:
            # land on the symmetric state of the result's orbit
            codes, phase, element = representatives(basis, new)
            amp = amp * phase * np.conj(basis.characters[element])
        else:
            codes = new @ basis.strides
        pos = np.minimum(np.searchsorted(basis.codes, codes), basis.dim - 1)
        keep = basis.codes[pos] == codes
        if symmetric:
            amp = amp * np.sqrt(basis.norms[pos] / basis.norms[source])
        rows.append(pos[keep])
        cols.append(source[keep])
        data.append(as_complex(coeff) * amp[keep])

    mat = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(basis.dim, basis.dim),
    ).tocsr()
    mat.sum_duplicates()
    if not np.any(mat.data.imag):
        return mat.real
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
