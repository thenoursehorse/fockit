"""Tests for display functions."""

from enum import StrEnum

from fockit.algebra import (
    Monomial,
    Op,
    OpSum,
    SwapResult,
    b,
    bdag,
    c,
    cdag,
    mode_sort_key,
    normal_order,
    sigma_x,
    sigma_z,
    sm,
    sp,
    to_latex,
    to_text,
    zero,
)


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


def test_new_algebra_displays_without_display_code():
    class Majorana(StrEnum):
        GAMMA = "gamma"

    class MajoranaRules:
        def swap(self, left: Op, right: Op) -> SwapResult:
            if left == right:
                return SwapResult(terms=((1, ()),))
            return SwapResult(terms=((-1, (right, left)),))

        def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
            return mode_sort_key(left.mode) < mode_sort_key(right.mode)

        def dagger(self, op: Op) -> Op:
            return op

        def __eq__(self, other: object) -> bool:
            return isinstance(other, MajoranaRules)

        def __hash__(self) -> int:
            return hash(type(self))

    def gamma(mode):
        return OpSum({Monomial((Op(mode, Majorana.GAMMA, MajoranaRules()),)): 1})

    op = normal_order(gamma(1) * gamma(0) + 2 * gamma(2) * gamma(2))
    assert to_text(op) == "2 - gamma(0) gamma(1)"
    assert to_latex(op) == "2 - gamma_{0} gamma_{1}"


def test_latex_spin_labels():
    assert (
        to_latex(cdag(0, "up") * c(1, "dn"))
        == r"c^{\dagger}_{0, \uparrow} c_{1, \downarrow}"
    )
    assert to_latex(cdag("update")) == r"c^{\dagger}_{update}"


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


def test_complex_coefficients_without_imaginary_part_display_as_real():
    assert to_text(normal_order(sp(0) * sm(0))) == "0.5 + 0.5 σ_z(0)"
    assert to_text(normal_order(sp(0) * sigma_z(0))) == "-0.5 σ_x(0) - 0.5j σ_y(0)"
