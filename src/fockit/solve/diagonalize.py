"""Diagonalization of Hamiltonian matrices."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from fockit.hilbert.matrix import SparseMatrix

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def eigh_full(H: SparseMatrix) -> tuple[RealArray, ComplexArray]:
    """Full dense diagonalization of ``H``.

    Returns
    -------
    energies : np.ndarray, shape (dim,)
        Real eigenvalues in ascending order.
    states : np.ndarray, shape (dim, dim)
        Column ``i`` is the i-th eigenvector.

    """
    dense = H.toarray()
    dense = 0.5 * (dense + dense.conj().T)
    evals, evecs = scipy.linalg.eigh(dense)
    return np.asarray(evals.real), np.asarray(evecs)


def eigenstates(H: SparseMatrix, k: int = 1) -> tuple[RealArray, ComplexArray]:
    """Compute the ``k`` lowest eigenpairs of ``H``.

    Returns
    -------
    eigenenergies : np.ndarray, shape (k,)
    eigenstates : np.ndarray, shape (dim, k)
        Column ``i`` is the i-th eigenvector.

    """
    if k >= H.shape[0] - 1:
        dense = H.toarray()
        dense = 0.5 * (dense + dense.conj().T)
        evals, evecs = scipy.linalg.eigh(dense)
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(H, k=k, which="SA")
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    return np.asarray(evals[:k].real), np.asarray(evecs[:, :k])


def groundstate(H: SparseMatrix) -> tuple[float, ComplexArray]:
    """Return ``(E0, psi0)``, the lowest eigenvalue and eigenvector."""
    evals, evecs = eigenstates(H, k=1)
    return float(evals[0]), evecs[:, 0]
