"""Hilbert spaces, symmetry-reduced bases, and operators as matrices on them."""

from fockit.hilbert.basis import Basis, build_basis, expand_state, state_index
from fockit.hilbert.matrix import expectations, matrix_elements, to_matrix
from fockit.hilbert.spaces import BosonSpace, FermionSpace, LocalSpace, PauliSpace

__all__ = [
    "Basis",
    "BosonSpace",
    "FermionSpace",
    "LocalSpace",
    "PauliSpace",
    "build_basis",
    "expand_state",
    "expectations",
    "matrix_elements",
    "state_index",
    "to_matrix",
]
