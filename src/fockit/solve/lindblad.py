"""Steady states and decay rates of Lindblad generators."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray

from fockit.hilbert.matrix import SparseMatrix

type ComplexArray = NDArray[np.complex128]

_TOL = 1e-12


def slowest_modes(
    generator: SparseMatrix, k: int = 1
) -> tuple[ComplexArray, ComplexArray]:
    """Find the ``k`` slowest-decaying eigenpairs of ``generator``.

    These have the largest real part. The decay rates are
    ``-eigenvalues.real``, and a steady state has rate 0.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
        Largest real part first.
    modes : np.ndarray, shape (dim, k)
        Column ``i`` is the i-th eigenvector.

    """
    if k >= generator.shape[0] - 1:
        values, vectors = scipy.linalg.eig(generator.toarray())
    else:
        values, vectors = scipy.sparse.linalg.eigs(generator, k=k, which="LR")
    order = np.argsort(-values.real)[:k]
    return np.asarray(values[order]), np.asarray(vectors[:, order])


def steady_state(generator: SparseMatrix, trace: ArrayLike) -> ComplexArray:
    """Find the steady state of ``generator``, normalized to ``Tr(rho) = 1``.

    Pass ``trace_vector(basis)`` as ``trace``. If there are several steady
    states, you get one of them.
    """
    _, modes = slowest_modes(generator)
    rho = modes[:, 0]
    norm = np.asarray(trace) @ rho
    if abs(norm) < _TOL:
        msg = "No steady state with a nonzero trace in this sector."
        raise ValueError(msg)
    return np.asarray(rho / norm)
