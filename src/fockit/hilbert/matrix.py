"""Build sparse matrix representations of operators in a product basis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from fockit._types import as_complex
from fockit.algebra.operators import OpSum, as_opsum
from fockit.hilbert.basis import Basis, apply_monomial, check_modes
from fockit.hilbert.spaces import IntArray
from fockit.hilbert.symmetries import representatives, symmetry_phases

type SparseMatrix = sp.csr_matrix
type ComplexArray = NDArray[np.complex128]

_HERMITIAN_TOL = 1e-10


def to_matrix(
    op: OpSum | complex, basis: Basis, source: Basis | None = None
) -> SparseMatrix:
    """Build the sparse matrix representation of ``op`` in ``basis``.

    With ``source``, build the matrix that takes states of ``source`` to
    states of ``basis``: rows in ``basis``, columns in ``source``. Use this
    for operators that change the symmetry sector, like ``c`` lowering the
    particle number. Both bases need the same permutation and on-site
    symmetries. Each symmetry must multiply ``op`` by a phase, and ``basis``
    must have ``source``'s eigenvalue times that phase.

    Returns
    -------
    scipy.sparse.csr_matrix
        Real if every matrix element is real, complex otherwise.

    """
    op = as_opsum(op)
    source = basis if source is None else source
    check_modes(op, basis.mode_index)
    if not basis.dim or not source.dim:
        return sp.csr_matrix((basis.dim, source.dim))
    _check_sectors(op, basis, source)
    symmetric = len(basis.group) > 1

    rows: list[IntArray] = [np.empty(0, dtype=np.int64)]
    cols: list[IntArray] = [np.empty(0, dtype=np.int64)]
    data: list[ComplexArray] = [np.empty(0, dtype=complex)]
    moved = False
    for mono, coeff in op.terms.items():
        start, new, amp = apply_monomial(
            mono, source.levels, source.mode_index, source.spaces, source.starts
        )
        moved = moved or len(start) > 0
        if symmetric:
            # land on the symmetric state of the result's orbit in basis
            codes, phase, element = representatives(basis, new)
            amp = amp * phase * np.conj(basis.characters[element])
        else:
            codes = new @ basis.strides
        pos = np.minimum(np.searchsorted(basis.codes, codes), basis.dim - 1)
        keep = basis.codes[pos] == codes
        if symmetric:
            amp = amp * np.sqrt(basis.norms[pos] / source.norms[start])
        rows.append(pos[keep])
        cols.append(start[keep])
        data.append(as_complex(coeff) * amp[keep])

    if moved and not any(len(r) for r in rows):
        msg = (
            "Operator takes every state outside the basis. Check its "
            "diagonal symmetries, e.g. the particle number."
        )
        raise ValueError(msg)
    mat = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(basis.dim, source.dim),
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


def _check_sectors(op: OpSum, basis: Basis, source: Basis) -> None:
    """Raise ValueError unless ``op`` takes the sector of ``source`` to ``basis``."""
    same_symmetries = (
        basis.modes == source.modes
        and [m for m, _ in basis.permutation_symmetries]
        == [m for m, _ in source.permutation_symmetries]
        and [f for f, _ in basis.onsite_symmetries]
        == [f for f, _ in source.onsite_symmetries]
        and basis.onsite_powers == source.onsite_powers
    )
    if not same_symmetries:
        msg = "basis and source must have the same modes and symmetries."
        raise ValueError(msg)
    names = [f"the relabelling {m!r}" for m, _ in basis.permutation_symmetries]
    names += ["an on-site symmetry"] * len(basis.onsite_symmetries)
    wanted = [e for _, e in (*basis.permutation_symmetries, *basis.onsite_symmetries)]
    start = [e for _, e in (*source.permutation_symmetries, *source.onsite_symmetries)]
    phases = symmetry_phases(op, basis.permutation_symmetries, basis.onsite_symmetries)
    for name, w, s, phase in zip(names, wanted, start, phases, strict=True):
        if not np.isclose(complex(w), complex(s) * phase):
            msg = (
                f"Operator takes the eigenvalue of {name} from {_rounded(s)} to "
                f"{_rounded(complex(s) * phase)}, but basis has {_rounded(w)}."
            )
            raise ValueError(msg)


def _rounded(value: complex) -> complex:
    """Round to 6 decimals for messages, without negative zeros."""
    value = complex(value)
    return complex(round(value.real, 6) + 0.0, round(value.imag, 6) + 0.0)
