"""Display and formatting for operator expressions."""

from __future__ import annotations

from operator_algebra.algebra import (
    AlgebraRules,
    BosonRules,
    FermionRules,
    _mode_sort_key,
)
from operator_algebra.types import LadderOp, Monomial, OpSum, _format_mode


def _op_symbols(rules: AlgebraRules) -> tuple[str, str]:
    """Return (creator_symbol, annihilator_symbol) for the given rules."""
    if isinstance(rules, FermionRules):
        return ("c†", "c")
    if isinstance(rules, BosonRules):
        return ("b†", "b")
    return ("a†", "a")


def _latex_symbols(rules: AlgebraRules) -> tuple[str, str]:
    """Return (creator_symbol, annihilator_symbol) in LaTeX."""
    if isinstance(rules, FermionRules):
        return (r"c^{\dagger}", "c")
    if isinstance(rules, BosonRules):
        return (r"b^{\dagger}", "b")
    return (r"a^{\dagger}", "a")


def _format_op(op: LadderOp, creator_sym: str, annihilator_sym: str) -> str:
    """Format a single LadderOp with given symbols."""
    symbol = creator_sym if op.is_creator else annihilator_sym
    return f"{symbol}({_format_mode(op.mode)})"


def _monomial_sort_key(mono: Monomial) -> tuple:  # type: ignore[type-arg]
    """Sort key for deterministic term ordering in display."""
    # Sort by: number of ops (ascending), then mode sort keys
    keys = []
    for op in mono.ops:
        keys.append((0 if op.is_creator else 1, _mode_sort_key(op.mode)))
    return (len(mono.ops), tuple(keys))


def _format_coeff(coeff: object) -> str:
    """Format a coefficient for display."""
    if coeff == 1:
        return ""
    if coeff == -1:
        return "-"
    return str(coeff)


def format_opsum(op: OpSum) -> str:
    """Format an OpSum as a human-readable string.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    if not op._terms:
        return "0"

    creator_sym, annihilator_sym = _op_symbols(op.rules)

    sorted_terms = sorted(
        op._terms.items(), key=lambda item: _monomial_sort_key(item[0])
    )

    parts: list[str] = []
    for mono, coeff in sorted_terms:
        if not mono.ops:
            # Identity term
            parts.append(str(coeff))
            continue

        ops_str = " ".join(
            _format_op(ladder_op, creator_sym, annihilator_sym)
            for ladder_op in mono.ops
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


def pretty_print(op: OpSum) -> str:
    """Pretty-print an operator expression with Unicode daggers.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    return format_opsum(op)


def to_latex(op: OpSum) -> str:
    r"""Convert an operator expression to LaTeX.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    if not op._terms:
        return "0"

    creator_sym, annihilator_sym = _latex_symbols(op.rules)

    sorted_terms = sorted(
        op._terms.items(), key=lambda item: _monomial_sort_key(item[0])
    )

    parts: list[str] = []
    for mono, coeff in sorted_terms:
        if not mono.ops:
            parts.append(str(coeff))
            continue

        ops_parts: list[str] = []
        for ladder_op in mono.ops:
            sym = creator_sym if ladder_op.is_creator else annihilator_sym
            mode_str = _format_mode(ladder_op.mode)
            # Convert mode to LaTeX subscript
            mode_latex = mode_str.replace("up", r"\uparrow").replace(
                "dn", r"\downarrow"
            )
            ops_parts.append(f"{sym}_{{{mode_latex}}}")

        ops_str = " ".join(ops_parts)

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
