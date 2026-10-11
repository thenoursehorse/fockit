"""Real-time evolution and finite-T dynamics primitives.

Provides three T=0 evolvers (``expm_multiply_evolve``, ``krylov_evolve``,
``chebyshev_evolve``) and lower-level helpers for composing finite-T
dynamical correlators (``tpq_state``, ``lanczos_eigenpairs``,
``boltzmann_weights``).

The finite-T helpers expose building blocks rather than fixed n-point
correlators so users can assemble arbitrary objects (2-point, OTOCs,
regulated OTOCs, ...) by composition.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.special import jv

from operator_algebra.solve.diagonalize import lanczos
from operator_algebra.solve.matrix import SparseMatrix

if TYPE_CHECKING:
    from collections.abc import Sequence

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def expm_multiply_evolve(
    H: SparseMatrix,
    psi: ComplexArray,
    times: Sequence[float],
) -> ComplexArray:
    """Evolve ``|ψ⟩`` under ``H`` to each ``t``.

    Uses ``scipy.expm_multiply``. ``times`` must be uniformly spaced and
    start at ``0`` for the time-stepping path; otherwise this falls back
    to one independent call per time.

    Returns
    -------
    np.ndarray, shape ``(len(times), dim)``
        Row ``i`` is ``e^{-i H t_i} ψ``.

    """
    t = np.asarray(times, dtype=float)
    if t.size >= 2 and t[0] == 0.0 and np.allclose(np.diff(t), t[1] - t[0]):
        return np.asarray(
            scipy.sparse.linalg.expm_multiply(
                -1j * H, psi, start=t[0], stop=t[-1], num=t.size, endpoint=True
            )
        )
    out = np.empty((t.size, H.shape[0]), dtype=complex)
    for i, ti in enumerate(t):
        out[i] = scipy.sparse.linalg.expm_multiply(-1j * ti * H, psi)
    return out


def krylov_evolve(
    H: SparseMatrix,
    psi: ComplexArray,
    times: Sequence[float],
    *,
    n_krylov: int = 30,
    tol: float = 1e-10,
) -> ComplexArray:
    """Evolve ``|ψ⟩`` via a Lanczos/Krylov approximation of ``e^{-i H dt}``.

    Between consecutive ``times[i]`` and ``times[i+1]`` a
    ``n_krylov``-step Lanczos run starts from the current state. The
    tridiagonal is diagonalized and the propagator is applied in the
    Krylov basis. A ``RuntimeWarning`` is raised when a step has not
    converged (trailing Krylov coefficient above ``tol``); then pass closer
    ``times`` or a larger ``n_krylov``.

    Returns
    -------
    np.ndarray, shape ``(len(times), dim)``

    """
    t = np.asarray(times, dtype=float)
    out = np.empty((t.size, H.shape[0]), dtype=complex)
    psi_cur = np.asarray(psi, dtype=complex)
    norm0 = float(np.linalg.norm(psi_cur))
    out[0] = psi_cur if t[0] == 0.0 else _krylov_step(H, psi_cur, t[0], n_krylov, tol)
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        psi_cur = _krylov_step(H, out[i - 1], dt, n_krylov, tol)
        out[i] = psi_cur
    if abs(norm0 - 1.0) < 1e-12:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def chebyshev_evolve(
    H: SparseMatrix,
    psi: ComplexArray,
    times: Sequence[float],
    *,
    n_moments: int | None = None,
    e_bounds: tuple[float, float] | None = None,
) -> ComplexArray:
    """Evolve ``|ψ⟩`` via a Chebyshev expansion of ``e^{-i H t}``.

    Builds the Chebyshev vectors ``T_k(H̃)|ψ⟩`` once and contracts them
    against the time-dependent Bessel coefficients for each entry of
    ``times``. Best when ``len(times)`` is large and ``H`` is fixed.

    Parameters
    ----------
    n_moments
        Truncation of the Chebyshev expansion. Defaults to
        ``int(a*max|t|) + 30`` where ``a`` is the half-spectral-width.
    e_bounds
        Optional ``(e_min, e_max)`` overriding the spectral bounds
        (otherwise estimated via ``eigsh``).

    """
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

    t = np.asarray(times, dtype=float)
    if n_moments is None:
        n_moments = int(a * np.max(np.abs(t))) + 30

    phi = np.empty((n_moments, H.shape[0]), dtype=complex)
    phi[0] = psi
    if n_moments > 1:
        phi[1] = h_tilde(psi)
    for k in range(2, n_moments):
        phi[k] = 2 * h_tilde(phi[k - 1]) - phi[k - 2]

    k_idx = np.arange(n_moments)
    coeffs = jv(k_idx[None, :], a * t[:, None]) * (-1j) ** k_idx[None, :]
    coeffs[:, 1:] *= 2.0
    phase = np.exp(-1j * b * t)
    return np.asarray(phase[:, None] * (coeffs @ phi))


def suzuki_trotter_sequence(
    n_fragments: int, dt: complex, order: int = 2
) -> list[tuple[complex, int]]:
    """Return a Suzuki-Trotter sequence for one step of size ``dt``.

    Parameters
    ----------
    n_fragments
        Number of Hamiltonian fragments ``H = Σ_i H_i``.
    dt
        Step size. May be complex (e.g. ``-1j*dt`` for real-time evolution,
        ``-dtau/2`` for imaginary-time / TPQ).
    order
        ``1`` for first-order forward sweep, or any even integer
        ``≥ 2`` for symmetric Suzuki-Trotter
        (recursive for ``order ≥ 4``).

    Returns
    -------
    list[tuple[complex, int]]
        Sequence of ``(coefficient, fragment_index)`` pairs. Applying
        ``exp(c * H_i)`` in order approximates ``exp(dt * H)``. Adjacent
        entries with the same fragment index are merged.

    """
    if order == 1:
        return [(dt, i) for i in range(n_fragments)]
    if order == 2:
        seq: list[tuple[complex, int]] = [(0.5 * dt, i) for i in range(n_fragments - 1)]
        seq.append((dt, n_fragments - 1))
        seq.extend((0.5 * dt, i) for i in reversed(range(n_fragments - 1)))
        return _merge_adjacent(seq)
    if order % 2 != 0 or order < 1:
        msg = "order must be 1 or an even integer ≥ 2"
        raise ValueError(msg)
    u_k = 1 / (4 - 4 ** (1 / (order - 1)))
    edge = suzuki_trotter_sequence(n_fragments, u_k * dt, order - 2)
    middle = suzuki_trotter_sequence(n_fragments, (1 - 4 * u_k) * dt, order - 2)
    return _merge_adjacent([*edge, *edge, *middle, *edge, *edge])


def trotter_evolve(
    H_fragments: Sequence[SparseMatrix],
    psi: ComplexArray,
    schedule: Sequence[tuple[complex, int]],
    n_steps: int,
) -> ComplexArray:
    """Apply a Trotter ``schedule`` ``n_steps`` times to ``ψ``.

    Each entry ``(c, i)`` of ``schedule`` applies ``exp(c * H_i)`` via
    ``scipy.expm_multiply``. No fused unitary is constructed, so memory
    stays sparse-friendly.

    Returns
    -------
    np.ndarray, shape ``(n_steps + 1, dim)``
        Row ``0`` is ``ψ``; row ``k`` is the state after ``k`` applications
        of the schedule.

    """
    out = np.empty((n_steps + 1, len(psi)), dtype=complex)
    out[0] = psi
    psi_cur = np.asarray(psi, dtype=complex)
    for k in range(1, n_steps + 1):
        for c, i in schedule:
            psi_cur = scipy.sparse.linalg.expm_multiply(c * H_fragments[i], psi_cur)
        out[k] = psi_cur
    return out


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
        e0 = float(
            scipy.sparse.linalg.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        )
    H_shift = H - e0 * scipy.sparse.eye(dim, dtype=complex, format="csr")
    r = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    r /= np.linalg.norm(r)
    return np.asarray(scipy.sparse.linalg.expm_multiply(-0.5 * beta * H_shift, r))


def lanczos_eigenpairs(
    H: SparseMatrix,
    *,
    n_lanczos: int,
    rng: np.random.Generator,
) -> tuple[RealArray, ComplexArray, ComplexArray]:
    """Approximate eigenpairs from a single random-vector Lanczos run.

    Returns
    -------
    eps : np.ndarray, shape ``(m,)``
        Ritz values from the ``m``-step tridiagonal (``m <= n_lanczos``).
    psi : np.ndarray, shape ``(dim, m)``
        Ritz vectors ``V U`` in the original basis.
    r : np.ndarray, shape ``(dim,)``
        Starting random unit vector. ``psi.conj().T @ r`` gives
        ``⟨ψ_k|r⟩``, useful for FTLM-style stochastic traces.

    """
    dim = H.shape[0]
    r = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    r /= np.linalg.norm(r)
    alphas, betas, V = lanczos(H, r, n_lanczos)
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    eps, U = np.linalg.eigh(T)
    psi = V @ U
    return eps, psi, r


def boltzmann_weights(energies: RealArray, beta: float) -> RealArray:
    """Return normalized Boltzmann weights ``e^{-β(E - E_min)} / Z``.

    Subtracting ``E_min`` keeps the exponent finite at low temperature.
    """
    w = np.exp(-beta * (energies - energies.min()))
    return np.asarray(w / w.sum())


def _merge_adjacent(
    seq: Sequence[tuple[complex, int]],
) -> list[tuple[complex, int]]:
    out: list[tuple[complex, int]] = []
    for c, i in seq:
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0] + c, i)
        else:
            out.append((c, i))
    return out


def _krylov_step(
    H: SparseMatrix,
    psi: ComplexArray,
    dt: float,
    m: int,
    tol: float,
) -> ComplexArray:
    norm = float(np.linalg.norm(psi))
    if norm == 0.0:
        return psi.copy()
    v0 = psi / norm
    alphas, betas, V = lanczos(H, v0, m)
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    eps, U = np.linalg.eigh(T)
    e1 = U[0, :]
    coeffs = U @ (np.exp(-1j * dt * eps) * e1)
    if len(coeffs) == m and abs(coeffs[-1]) > tol:
        msg = (
            f"Krylov step not converged (last coefficient {abs(coeffs[-1]):.1e} > "
            f"tol={tol:.0e}); use smaller time steps or a larger n_krylov."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return np.asarray(norm * (V @ coeffs))
