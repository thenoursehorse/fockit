"""Thermal expectation values via exact and stochastic methods."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from operator_algebra.solve.diagonalize import lanczos
from operator_algebra.solve.matrix import SparseMatrix

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


def lanczos_thermal_expectation(
    op: SparseMatrix,
    H: SparseMatrix,
    beta: float,
    *,
    n_samples: int,
    n_lanczos: int,
    method: str = "ftlm",
    seed: int | None = None,
) -> complex:
    """Finite-temperature expectation via FTLM or LTLM.

    Parameters
    ----------
    method : {"ftlm", "ltlm"}
        Standard finite-T Lanczos (cheap, biased at low T) or its
        symmetrized low-T variant (one extra ``n_lanczos`` matvecs of
        ``op`` per sample, but well-behaved at low T).

    """
    if method not in ("ftlm", "ltlm"):
        msg = f"method must be 'ftlm' or 'ltlm', got {method!r}"
        raise ValueError(msg)

    dim = H.shape[0]
    rng = np.random.default_rng(seed)

    samples: list[tuple[RealArray, ComplexArray, ComplexArray]] = []
    for _ in range(n_samples):
        v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v0 /= np.linalg.norm(v0)
        alphas, betas, V = lanczos(H, v0, n_lanczos)
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        eps, U = np.linalg.eigh(T)
        c = U[0, :]
        if method == "ftlm":
            w = V.conj().T @ (op @ v0)
            payload = U.conj().T @ w
        else:
            AV = op @ V
            M_op = V.conj().T @ AV
            payload = U.conj().T @ M_op @ U
        samples.append((eps, c, payload))

    shift = min(eps.min() for eps, _, _ in samples)
    num: complex = 0j
    denom: float = 0.0
    for eps, c, payload in samples:
        w_e = np.exp(-beta * (eps - shift))
        denom += float(np.sum(w_e * c * c))
        if method == "ftlm":
            num += complex(np.sum(w_e * c * payload))
        else:
            wc = np.exp(-0.5 * beta * (eps - shift)) * c
            num += complex(wc @ payload @ wc)
    return num / denom


def tpq_thermal_expectation(
    op: SparseMatrix,
    H: SparseMatrix,
    beta: float,
    *,
    n_samples: int,
    seed: int | None = None,
) -> complex:
    """Finite-T expectation via Sugiura-Shimizu TPQ states.

    Builds ``|ψ⟩ = e^{-β H / 2} |r⟩`` for random unit vectors ``|r⟩`` via
    ``scipy.sparse.linalg.expm_multiply`` and averages
    ``⟨ψ|op|ψ⟩ / ⟨ψ|ψ⟩``.
    """
    dim = H.shape[0]
    e0 = float(
        scipy.sparse.linalg.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    )
    H_shift = H - e0 * scipy.sparse.eye(dim, dtype=complex, format="csr")
    rng = np.random.default_rng(seed)

    num: complex = 0j
    denom: float = 0.0
    for _ in range(n_samples):
        r = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        r /= np.linalg.norm(r)
        psi = scipy.sparse.linalg.expm_multiply(-0.5 * beta * H_shift, r)
        norm2 = float((psi.conj() @ psi).real)
        num += complex(psi.conj() @ (op @ psi))
        denom += norm2
    return num / denom


def kpm_thermal_expectation(
    op: SparseMatrix,
    H: SparseMatrix,
    beta: float,
    *,
    n_moments: int,
    n_samples: int,
    e_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> complex:
    """Finite-T expectation via Chebyshev/KPM expansion of ``e^{-β H}``.

    Stochastically estimates the moments ``μ_n = Tr[T_n(H̃)]`` and
    ``nu_n = Tr[op T_n(H̃)]`` with ``n_samples`` random vectors, then
    contracts them with the Jackson-damped Chebyshev coefficients of
    ``e^{-β H}``. ``H̃`` is ``H`` rescaled to ``[-1, 1]``; ``e_bounds``
    overrides the spectral bounds (otherwise estimated via ``eigsh``).
    """
    dim = H.shape[0]
    if e_bounds is None:
        e_min = float(
            scipy.sparse.linalg.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        )
        e_max = float(
            scipy.sparse.linalg.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
        )
    else:
        e_min, e_max = e_bounds
    pad = 0.01 * (e_max - e_min)
    a = (e_max - e_min) / 2 + pad
    b = (e_max + e_min) / 2

    def h_tilde(v: ComplexArray) -> ComplexArray:
        # H rescaled so its spectrum lies in [-1, 1]
        return np.asarray((H @ v - b * v) / a)

    rng = np.random.default_rng(seed)
    mu = np.zeros(n_moments)
    nu = np.zeros(n_moments, dtype=complex)
    for _ in range(n_samples):
        r = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        r /= np.linalg.norm(r)
        v0 = r
        v1 = h_tilde(r)
        mu[0] += float((r.conj() @ v0).real)
        nu[0] += complex(r.conj() @ (op @ v0))
        if n_moments > 1:
            mu[1] += float((r.conj() @ v1).real)
            nu[1] += complex(r.conj() @ (op @ v1))
        for n_idx in range(2, n_moments):
            v2 = 2 * h_tilde(v1) - v0
            mu[n_idx] += float((r.conj() @ v2).real)
            nu[n_idx] += complex(r.conj() @ (op @ v2))
            v0, v1 = v1, v2
    mu /= n_samples
    nu /= n_samples

    n_idx_arr = np.arange(n_moments)
    M = max(2 * n_moments, 64)
    k = np.arange(M)
    x = np.cos(np.pi * (k + 0.5) / M)
    f = np.exp(-beta * a * (x + 1.0))
    alpha = (2.0 / M) * (
        f[None, :] * np.cos(np.pi * n_idx_arr[:, None] * (k + 0.5) / M)
    ).sum(axis=1)
    alpha[0] *= 0.5
    g = (
        (n_moments - n_idx_arr + 1) * np.cos(np.pi * n_idx_arr / (n_moments + 1))
        + np.sin(np.pi * n_idx_arr / (n_moments + 1)) / np.tan(np.pi / (n_moments + 1))
    ) / (n_moments + 1)
    coeffs = g * alpha
    return complex((coeffs @ nu) / (coeffs @ mu))
