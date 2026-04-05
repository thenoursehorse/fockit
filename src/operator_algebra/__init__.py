"""Symbolic operator algebra for fermions, bosons, and beyond."""

from operator_algebra.algebra import (
    BOSON,
    FERMION,
    AlgebraRules,
    BosonRules,
    FermionRules,
    SwapResult,
)
from operator_algebra.constructors import (
    annihilate,
    b,
    bdag,
    c,
    cdag,
    create,
    identity,
    n,
    nb,
    number,
    zero,
)
from operator_algebra.display import pretty_print, to_latex
from operator_algebra.inspection import degree, modes, monomials, num_terms
from operator_algebra.normal_order import is_normal_ordered, normal_order
from operator_algebra.simplify import collect_terms, drop_zeros, simplify_coeffs
from operator_algebra.transforms import anticommutator, commutator, hc, substitute_modes
from operator_algebra.types import LadderOp, Monomial, OpSum

__all__ = [
    "BOSON",
    "FERMION",
    "AlgebraRules",
    "BosonRules",
    "FermionRules",
    "LadderOp",
    "Monomial",
    "OpSum",
    "SwapResult",
    "annihilate",
    "anticommutator",
    "b",
    "bdag",
    "c",
    "cdag",
    "collect_terms",
    "commutator",
    "create",
    "degree",
    "drop_zeros",
    "hc",
    "identity",
    "is_normal_ordered",
    "modes",
    "monomials",
    "n",
    "nb",
    "normal_order",
    "num_terms",
    "number",
    "pretty_print",
    "simplify_coeffs",
    "substitute_modes",
    "to_latex",
    "zero",
]
