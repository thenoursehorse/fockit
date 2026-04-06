"""Converters from OpSum to external operator formats."""

from __future__ import annotations

from operator_algebra.algebra import FermionRules
from operator_algebra.operators import OpSum


def to_triqs(op: OpSum) -> object:
    """Convert a fermionic OpSum to a TRIQS Operator.

    Parameters
    ----------
    op : OpSum
        Operator expression with ``FermionRules``.

    Returns
    -------
    triqs.operators.Operator

    """
    if not isinstance(op.rules, FermionRules):
        msg = f"to_triqs requires FermionRules, got {type(op.rules).__name__}"
        raise TypeError(msg)

    from triqs.operators import Operator, c, c_dag  # noqa: PLC0415

    result = Operator()
    for mono, coeff in op.terms.items():
        c_val = complex(coeff)
        term = c_val.real if c_val.imag == 0 else c_val
        for ladder_op in mono.ops:
            indices = (
                ladder_op.mode
                if isinstance(ladder_op.mode, tuple)
                else (ladder_op.mode,)
            )
            if ladder_op.kind is True:
                term = term * c_dag(*indices)
            else:
                term = term * c(*indices)
        result = result + term
    return result
