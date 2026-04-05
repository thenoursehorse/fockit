"""Tests for display functions."""

from operator_algebra import (
    LadderOp,
    Monomial,
    OpSum,
    SwapResult,
    b,
    bdag,
    c,
    cdag,
    pretty_print,
    to_latex,
)


def test_fermion_display():
    s = repr(cdag(1) * c(2))
    assert "c†(1)" in s
    assert "c(2)" in s


def test_boson_display():
    s = repr(bdag(1) * b(2))
    assert "b†(1)" in s
    assert "b(2)" in s


def test_custom_rules_display():
    """Custom rules get generic a†/a notation."""

    class CustomRules:
        def swap(self, left, right):
            return SwapResult(terms=((1, (right, left)),))

        def is_normal_ordered_pair(self, left, right):
            return True

        def is_zero_monomial(self, ops):
            return False

        def __eq__(self, other):
            return isinstance(other, CustomRules)

        def __hash__(self):
            return hash(type(self))

    op = OpSum({Monomial((LadderOp(1, True),)): 1}, rules=CustomRules())
    s = repr(op)
    assert "a†(1)" in s


def test_to_latex_fermion():
    s = to_latex(cdag(1))
    assert r"c^{\dagger}" in s


def test_to_latex_boson():
    s = to_latex(bdag(1))
    assert r"b^{\dagger}" in s


def test_zero_display():
    from operator_algebra import zero

    assert repr(zero()) == "0"


def test_pretty_print():
    s = pretty_print(cdag(1))
    assert "c†(1)" in s
