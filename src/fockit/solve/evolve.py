"""Time evolution: scipy's expm_multiply, a Chebyshev expansion, or Trotter steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.special import jv

from fockit.hilbert.matrix import SparseMatrix, check_hermitian

if TYPE_CHECKING:
    from collections.abc import Sequence

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def expm_multiply_evolve(
    H: SparseMatrix,
    psi: ComplexArray,
    times: ArrayLike,
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


def chebyshev_evolve(
    H: SparseMatrix,
    psi: ComplexArray,
    times: ArrayLike,
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
    check_hermitian(H)
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
