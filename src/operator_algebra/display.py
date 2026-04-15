"""Display and formatting for operator expressions."""

from __future__ import annotations

from collections.abc import Hashable

from operator_algebra._types import Scalar
from operator_algebra.algebra import _mode_sort_key
from operator_algebra.operators import Monomial, OpSum


def _monomial_sort_key(mono: Monomial) -> tuple[object, ...]:
    """Sort key for deterministic term ordering in display."""
    keys = []
    for op in mono.ops:
        keys.append((_kind_sort_key(op.kind), _mode_sort_key(op.mode)))
    return (len(mono.ops), tuple(keys))


def _kind_sort_key(kind: Hashable) -> tuple[object, ...]:
    """Sort key for operator kind within display ordering."""
    if kind is True:
        return (0,)
    if kind is False:
        return (1,)
    if isinstance(kind, str):
        return (2, kind)
    return (3, repr(kind))


def _format_coeff(coeff: Scalar) -> str:
    """Format a coefficient for display."""
    if coeff == 1:
        return ""
    if coeff == -1:
        return "-"
    return str(coeff)


def to_text(op: OpSum) -> str:
    """Format an OpSum as a human-readable string.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    if not op.terms:
        return "0"

    sorted_terms = sorted(
        op.terms.items(), key=lambda item: _monomial_sort_key(item[0])
    )

    parts: list[str] = []
    for mono, coeff in sorted_terms:
        if not mono.ops:
            parts.append(str(coeff))
            continue

        ops_str = " ".join(op.rules.format_op_text(ladder_op) for ladder_op in mono.ops)

        coeff_str = _format_coeff(coeff)
        if coeff_str == "":
            parts.append(ops_str)
        elif coeff_str == "-":
            parts.append(f"-{ops_str}")
        else:
            parts.append(f"{coeff_str} {ops_str}")

    if not parts:
        return "0"

    result = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            result += f" - {part[1:]}"
        else:
            result += f" + {part}"
    return result


def to_latex(op: OpSum) -> str:
    r"""Convert an operator expression to LaTeX.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    if not op.terms:
        return "0"

    sorted_terms = sorted(
        op.terms.items(), key=lambda item: _monomial_sort_key(item[0])
    )

    parts: list[str] = []
    for mono, coeff in sorted_terms:
        if not mono.ops:
            parts.append(str(coeff))
            continue

        ops_str = " ".join(
            op.rules.format_op_latex(ladder_op) for ladder_op in mono.ops
        )

        coeff_str = _format_coeff(coeff)
        if coeff_str == "":
            parts.append(ops_str)
        elif coeff_str == "-":
            parts.append(f"-{ops_str}")
        else:
            parts.append(f"{coeff_str} {ops_str}")

    if not parts:
        return "0"

    result = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            result += f" - {part[1:]}"
        else:
            result += f" + {part}"
    return result
