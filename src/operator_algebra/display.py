"""Display and formatting for operator expressions."""

from __future__ import annotations

from collections.abc import Callable, Hashable

from operator_algebra._types import Coefficient
from operator_algebra.algebra import (
    BosonRules,
    FermionRules,
    Ladder,
    Pauli,
    PauliRules,
    mode_sort_key,
)
from operator_algebra.operators import Monomial, Op, OpSum, format_mode

# operators of other algebras show their kind as the symbol
_TEXT_SYMBOLS: dict[tuple[type, Hashable], str] = {
    (FermionRules, Ladder.CREATE): "c†",
    (FermionRules, Ladder.ANNIHILATE): "c",
    (BosonRules, Ladder.CREATE): "b†",
    (BosonRules, Ladder.ANNIHILATE): "b",
    (PauliRules, Pauli.X): "σ_x",
    (PauliRules, Pauli.Y): "σ_y",
    (PauliRules, Pauli.Z): "σ_z",
}
_LATEX_SYMBOLS: dict[tuple[type, Hashable], str] = {
    (FermionRules, Ladder.CREATE): r"c^{\dagger}",
    (FermionRules, Ladder.ANNIHILATE): "c",
    (BosonRules, Ladder.CREATE): r"b^{\dagger}",
    (BosonRules, Ladder.ANNIHILATE): "b",
    (PauliRules, Pauli.X): r"\sigma^{x}",
    (PauliRules, Pauli.Y): r"\sigma^{y}",
    (PauliRules, Pauli.Z): r"\sigma^{z}",
}
_LATEX_LABELS = {"up": r"\uparrow", "dn": r"\downarrow"}


def to_text(op: OpSum) -> str:
    """Format an OpSum as a human-readable string.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    return _join_terms(op, _op_text)


def to_latex(op: OpSum) -> str:
    r"""Convert an operator expression to LaTeX.

    Parameters
    ----------
    op : OpSum

    Returns
    -------
    str

    """
    return _join_terms(op, _op_latex)


def _join_terms(op: OpSum, format_op: Callable[[Op], str]) -> str:
    """Write the terms in display order, joined with + and -."""
    if not op.terms:
        return "0"

    sorted_terms = sorted(
        op.terms.items(), key=lambda item: _monomial_sort_key(item[0])
    )

    parts: list[str] = []
    for mono, raw_coeff in sorted_terms:
        coeff = _real_if_possible(raw_coeff)
        if not mono.ops:
            parts.append(str(coeff))
            continue

        ops_str = " ".join(format_op(single_op) for single_op in mono.ops)

        coeff_str = _format_coeff(coeff)
        if coeff_str == "":
            parts.append(ops_str)
        elif coeff_str == "-":
            parts.append(f"-{ops_str}")
        else:
            parts.append(f"{coeff_str} {ops_str}")

    result = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            result += f" - {part[1:]}"
        else:
            result += f" + {part}"
    return result


def _op_text(op: Op) -> str:
    """Format as e.g. ``c†(0, up)``."""
    symbol = _TEXT_SYMBOLS.get((type(op.rules), op.kind), op.kind)
    return f"{symbol}({format_mode(op.mode)})"


def _op_latex(op: Op) -> str:
    r"""Format as e.g. ``c^{\dagger}_{0, \uparrow}``."""
    symbol = _LATEX_SYMBOLS.get((type(op.rules), op.kind), op.kind)
    labels = op.mode if isinstance(op.mode, tuple) else (op.mode,)
    subscript = ", ".join(_LATEX_LABELS.get(label, str(label)) for label in labels)
    return f"{symbol}_{{{subscript}}}"


def _monomial_sort_key(mono: Monomial) -> tuple[object, ...]:
    """Sort key for deterministic term ordering in display."""
    keys = []
    for op in mono.ops:
        keys.append((_kind_sort_key(op.kind), mode_sort_key(op.mode)))
    return (len(mono.ops), tuple(keys))


def _kind_sort_key(kind: Hashable) -> tuple[object, ...]:
    """Sort key for operator kind within display ordering."""
    if isinstance(kind, str):
        return (2, kind)
    return (3, repr(kind))


def _real_if_possible(coeff: Coefficient) -> Coefficient:
    """Show a complex number with no imaginary part as a real one."""
    if isinstance(coeff, complex) and coeff.imag == 0:
        return coeff.real
    return coeff


def _format_coeff(coeff: Coefficient) -> str:
    """Format a coefficient for display."""
    if coeff == 1:
        return ""
    if coeff == -1:
        return "-"
    return str(coeff)
