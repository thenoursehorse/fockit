"""Tests for display functions."""

from operator_algebra import b, bdag, c, cdag, sigma_x, to_latex, to_text, zero
from operator_algebra.algebra import SwapResult
from operator_algebra.operators import Monomial, Op, OpSum


def test_fermion_display():
    s = repr(cdag(1) * c(2))
    assert "c†(1)" in s
    assert "c(2)" in s


def test_boson_display():
    s = repr(bdag(1) * b(2))
    assert "b†(1)" in s
    assert "b(2)" in s


def test_pauli_display():
    s = repr(sigma_x(0))
    assert "σ_x(0)" in s


def test_custom_rules_display():
    """Custom rules with format_op_text get custom notation."""

    class CustomRules:
        def swap(self, left: Op, right: Op) -> SwapResult:
            return SwapResult(terms=((1, (right, left)),))

        def is_normal_ordered_pair(self, _left: Op, _right: Op) -> bool:
            return True

        def is_zero_monomial(self, _ops: tuple[Op, ...]) -> bool:
            return False

        def dagger(self, op: Op) -> Op:
            return Op(op.mode, not op.kind)

        def format_op_text(self, op: Op) -> str:
            symbol = "a†" if op.kind else "a"
            return f"{symbol}({op.mode})"

        def format_op_latex(self, op: Op) -> str:
            symbol = r"a^{\dagger}" if op.kind else "a"
            return f"{symbol}_{{{op.mode}}}"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, CustomRules)

        def __hash__(self) -> int:
            return hash(type(self))

    op = OpSum({Monomial((Op(1, True),)): 1}, rules=CustomRules())
    s = repr(op)
    assert "a†(1)" in s


def test_to_latex_fermion():
    s = to_latex(cdag(1))
    assert r"c^{\dagger}" in s


def test_to_latex_boson():
    s = to_latex(bdag(1))
    assert r"b^{\dagger}" in s


def test_to_latex_pauli():
    s = to_latex(sigma_x(0))
    assert r"\sigma^{x}_{0}" in s


def test_zero_display():
    assert repr(zero()) == "0"


def test_to_text():
    s = to_text(cdag(1))
    assert "c†(1)" in s
