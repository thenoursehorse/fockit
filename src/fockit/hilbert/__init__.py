"""Hilbert spaces, symmetry-reduced bases, and operators as matrices on them."""

from fockit.hilbert.basis import Basis, build_basis, expand_state, state_index
from fockit.hilbert.matrix import expectation, matrix_elements, to_matrix
from fockit.hilbert.spaces import (
    BosonSpace,
    Conjugate,
    FermionSpace,
    LocalSpace,
    PauliSpace,
    doubled,
)

__all__ = [
    "Basis",
    "BosonSpace",
    "Conjugate",
    "FermionSpace",
    "LocalSpace",
    "PauliSpace",
    "build_basis",
    "doubled",
    "expand_state",
    "expectation",
    "matrix_elements",
    "state_index",
    "to_matrix",
]
