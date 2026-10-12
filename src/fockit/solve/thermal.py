"""Thermal averages: exact from the full spectrum, or with TPQ states."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from fockit.hilbert.matrix import SparseMatrix

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def thermal_expectation(
    op: SparseMatrix,
    energies: RealArray,
    states: ComplexArray,
    beta: float,
) -> complex:
    """Thermal expectation ``Tr[op e^{-beta H}] / Z``.

    Evaluated in the eigenbasis of ``H``. ``energies`` and ``states``
    are the full spectrum from :func:`eigh_full`.
    """
    diag = np.sum(states.conj() * (op @ states), axis=0)
    w = np.exp(-beta * (energies - energies.min()))
    return complex((w @ diag) / w.sum())


def tpq_thermal_expectation(
    op: SparseMatrix,
    H: SparseMatrix,
    beta: float,
    *,
    n_samples: int,
    seed: int | None = None,
) -> complex:
    """Thermal expectation averaged over ``n_samples`` TPQ states.

    For systems too large for :func:`eigh_full`. Each sample is a state from
    :func:`tpq_state`; the result is ``sum ⟨φ|op|φ⟩ / sum ⟨φ|φ⟩``.
    """
    e0 = _lowest_energy(H)
    rng = np.random.default_rng(seed)
    num: complex = 0j
    denom: float = 0.0
    for _ in range(n_samples):
        phi = tpq_state(H, beta, rng=rng, e0=e0)
        num += complex(phi.conj() @ (op @ phi))
        denom += float((phi.conj() @ phi).real)
    return num / denom


def tpq_state(
    H: SparseMatrix,
    beta: float,
    *,
    rng: np.random.Generator,
    e0: float | None = None,
) -> ComplexArray:
    """Sample one TPQ state ``|φ⟩ = e^{-β H/2}|r⟩``.

    The random input ``|r⟩`` is a unit vector. ``H`` is shifted by its
    ground-state energy ``e0`` so the matrix exponential does not overflow;
    pass ``e0`` when sampling many states so it is only computed once.
    Caller handles sample averaging and the state-dependent norm.
    """
    dim = H.shape[0]
    if e0 is None:
        e0 = _lowest_energy(H)
    H_shift = H - e0 * scipy.sparse.eye(dim, dtype=complex, format="csr")
    r = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    r /= np.linalg.norm(r)
    return np.asarray(scipy.sparse.linalg.expm_multiply(-0.5 * beta * H_shift, r))


def _lowest_energy(H: SparseMatrix) -> float:
    return float(
        scipy.sparse.linalg.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    )
