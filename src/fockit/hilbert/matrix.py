"""Build sparse matrix representations of operators in a product basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from fockit._types import as_complex
from fockit.algebra.operators import OpSum, as_opsum
from fockit.hilbert.basis import Basis, apply_monomial, check_modes
from fockit.hilbert.spaces import IntArray
from fockit.hilbert.symmetries import check_symmetric, representatives

type SparseMatrix = sp.csr_matrix
type ComplexArray = NDArray[np.complex128]

_HERMITIAN_TOL = 1e-10


def to_matrix(op: OpSum | complex, basis: Basis) -> SparseMatrix:
    """Build the sparse matrix representation of ``op`` in ``basis``.

    With permutation symmetries, ``op`` must commute with them.

    Returns
    -------
    scipy.sparse.csr_matrix
        Real if every matrix element is real, complex otherwise.

    """
    op = as_opsum(op)
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


def check_hermitian(H: SparseMatrix) -> None:
    """Raise ValueError if ``H`` isn't Hermitian, allowing for round-off."""
    scale = max(abs(H).max(), 1.0)
    if H.shape[0] and abs(H - H.conj().T).max() > _HERMITIAN_TOL * scale:
        msg = (
            "Matrix is not Hermitian. For a Lindblad generator use steady_state "
            "or slowest_modes."
        )
        raise ValueError(msg)


def expectation(op: SparseMatrix, state: ComplexArray) -> complex:
    """Compute ``⟨ψ|op|ψ⟩`` for one state of shape ``(dim,)``."""
    return complex(state.conj() @ (op @ state))


def matrix_elements(op: SparseMatrix, states: ComplexArray) -> ComplexArray:
    """Return ``O_{mn} = ⟨m|op|n⟩`` for the column states in ``states``.

    ``states`` has shape ``(dim, k)`` (columns are the bra/ket states).
    """
    return np.asarray(states.conj().T @ (op @ states))
