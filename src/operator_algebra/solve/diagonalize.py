"""Diagonalization of operators in a Fock basis."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from operator_algebra.operators import OpSum
from operator_algebra.solve.basis import Basis
from operator_algebra.solve.matrix import CSRMatrix, to_matrix

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def eigh_full(H: OpSum, basis: Basis) -> tuple[RealArray, ComplexArray]:
    """Full dense diagonalization of ``H`` in ``basis``.

    Returns
    -------
    energies : np.ndarray, shape (dim,)
        Real eigenvalues in ascending order.
    states : np.ndarray, shape (dim, dim)
        Column ``i`` is the i-th eigenvector.

    """
    dense = to_matrix(H, basis).toarray()
    dense = 0.5 * (dense + dense.conj().T)
    evals, evecs = scipy.linalg.eigh(dense)
    return np.asarray(evals.real), np.asarray(evecs)


def eigenstates(H: OpSum, basis: Basis, k: int = 1) -> tuple[RealArray, ComplexArray]:
    """Compute the ``k`` lowest eigenpairs of ``H`` in ``basis``.

    Returns
    -------
    eigenenergies : np.ndarray, shape (k,)
    eigenstates : np.ndarray, shape (dim, k)
        Column ``i`` is the i-th eigenvector.

    """
    mat = to_matrix(H, basis)
    dim = basis.dim

    if k >= dim - 1:
        dense = mat.toarray()
        dense = 0.5 * (dense + dense.conj().T)
        evals, evecs = scipy.linalg.eigh(dense)
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(mat, k=k, which="SA")
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    return np.asarray(evals[:k].real), np.asarray(evecs[:, :k])


def groundstate(H: OpSum, basis: Basis) -> tuple[float, ComplexArray]:
    """Return ``(E0, psi0)`` — the lowest eigenvalue and its eigenvector."""
    evals, evecs = eigenstates(H, basis, k=1)
    return float(evals[0]), evecs[:, 0]


def lanczos(
    H: CSRMatrix,
    v0: ComplexArray,
    m: int,
) -> tuple[RealArray, RealArray, ComplexArray]:
    """Plain m-step Lanczos from a unit vector ``v0``.

    Returns ``(alphas, betas, V)`` with ``alphas`` length ``j``,
    ``betas`` length ``j-1``, ``V`` shape ``(dim, j)``, where ``j <= m``
    (early exit on breakdown).
    """
    dim = H.shape[0]
    V = np.zeros((dim, m), dtype=complex)
    alphas = np.zeros(m)
    betas = np.zeros(m - 1)
    v = v0
    V[:, 0] = v
    w = H @ v
    a = float((v.conj() @ w).real)
    alphas[0] = a
    w = w - a * v
    for j in range(1, m):
        b = float(np.linalg.norm(w))
        if b < 1e-12:
            return alphas[:j], betas[: j - 1], V[:, :j]
        v_new = w / b
        V[:, j] = v_new
        betas[j - 1] = b
        w = H @ v_new - b * v
        a = float((v_new.conj() @ w).real)
        alphas[j] = a
        w = w - a * v_new
        v = v_new
    return alphas, betas, V
