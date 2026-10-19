"""Hilbert spaces, symmetry-reduced bases, and operators as matrices on them."""

from fockit.hilbert.basis import (
    Basis,
    build_basis,
    density_matrix,
    expand_state,
    state_index,
    trace_vector,
)
from fockit.hilbert.matrix import expectation, matrix_elements, to_matrix
from fockit.hilbert.spaces import (
    BosonSpace,
    Conjugate,
    FermionSpace,
    LocalSpace,
    PauliSpace,
    doubled,
)
from fockit.hilbert.symmetries import symmetry_components

__all__ = [
    "Basis",
    "BosonSpace",
    "Conjugate",
    "FermionSpace",
    "LocalSpace",
    "PauliSpace",
    "build_basis",
    "density_matrix",
    "doubled",
    "expand_state",
    "expectation",
    "matrix_elements",
    "state_index",
    "symmetry_components",
    "to_matrix",
    "trace_vector",
]
