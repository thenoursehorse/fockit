"""Converters from OpSum to external operator formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fockit._types import as_complex
from fockit.algebra.operators import OpSum
from fockit.algebra.rules import FermionRules, Ladder

if TYPE_CHECKING:
    from triqs.operators import Operator as TriqsOperator


def to_triqs(op: OpSum) -> TriqsOperator:
    """Convert a fermionic OpSum to a TRIQS Operator.

    Parameters
    ----------
    op : OpSum
        Operator expression with ``FermionRules``.

    Returns
    -------
    triqs.operators.Operator

    """
    for mono in op.terms:
        for single_op in mono.ops:
            if not isinstance(single_op.rules, FermionRules):
                name = type(single_op.rules).__name__
                msg = f"to_triqs requires FermionRules, got {name}"
                raise TypeError(msg)

    from triqs.operators import Operator, c, c_dag

    result = Operator()
    for mono, coeff in op.terms.items():
        c_val = as_complex(coeff)
        term = c_val.real if c_val.imag == 0 else c_val
        for ladder_op in mono.ops:
            indices = (
                ladder_op.mode
                if isinstance(ladder_op.mode, tuple)
                else (ladder_op.mode,)
            )
            if ladder_op.kind is Ladder.CREATE:
                term = term * c_dag(*indices)
            else:
                term = term * c(*indices)
        result = result + term
    return result
