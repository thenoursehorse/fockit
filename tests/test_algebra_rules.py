"""Tests for algebra rules."""

from operator_algebra import (
    BOSON,
    FERMION,
    BosonRules,
    FermionRules,
    LadderOp,
)


class TestFermionRules:
    def test_swap_annihilator_creator_same_mode(self):
        result = FERMION.swap(LadderOp(1, False), LadderOp(1, True))
        assert len(result.terms) == 2
        # Delta term
        assert result.terms[0] == (1, ())
        # Swapped term with -1
        assert result.terms[1] == (-1, (LadderOp(1, True), LadderOp(1, False)))

    def test_swap_annihilator_creator_diff_mode(self):
        result = FERMION.swap(LadderOp(1, False), LadderOp(2, True))
        assert len(result.terms) == 1
        assert result.terms[0] == (-1, (LadderOp(2, True), LadderOp(1, False)))

    def test_swap_two_creators(self):
        result = FERMION.swap(LadderOp(2, True), LadderOp(1, True))
        assert result.terms == ((-1, (LadderOp(1, True), LadderOp(2, True))),)

    def test_swap_two_annihilators(self):
        result = FERMION.swap(LadderOp(2, False), LadderOp(1, False))
        assert result.terms == ((-1, (LadderOp(1, False), LadderOp(2, False))),)

    def test_normal_ordered_pair_creator_before_annihilator(self):
        assert FERMION.is_normal_ordered_pair(LadderOp(1, True), LadderOp(1, False))

    def test_normal_ordered_pair_annihilator_before_creator(self):
        assert not FERMION.is_normal_ordered_pair(LadderOp(1, False), LadderOp(1, True))

    def test_normal_ordered_pair_same_type_ordered(self):
        assert FERMION.is_normal_ordered_pair(LadderOp(1, True), LadderOp(2, True))

    def test_normal_ordered_pair_same_type_unordered(self):
        assert not FERMION.is_normal_ordered_pair(LadderOp(2, True), LadderOp(1, True))

    def test_zero_monomial_duplicate(self):
        ops = (LadderOp(1, True), LadderOp(1, True))
        assert FERMION.is_zero_monomial(ops)

    def test_zero_monomial_no_duplicate(self):
        ops = (LadderOp(1, True), LadderOp(1, False))
        assert not FERMION.is_zero_monomial(ops)


class TestBosonRules:
    def test_swap_annihilator_creator_same_mode(self):
        result = BOSON.swap(LadderOp(1, False), LadderOp(1, True))
        assert len(result.terms) == 2
        assert result.terms[0] == (1, ())
        assert result.terms[1] == (+1, (LadderOp(1, True), LadderOp(1, False)))

    def test_swap_annihilator_creator_diff_mode(self):
        result = BOSON.swap(LadderOp(1, False), LadderOp(2, True))
        assert len(result.terms) == 1
        assert result.terms[0] == (+1, (LadderOp(2, True), LadderOp(1, False)))

    def test_swap_two_creators(self):
        result = BOSON.swap(LadderOp(2, True), LadderOp(1, True))
        assert result.terms == ((+1, (LadderOp(1, True), LadderOp(2, True))),)

    def test_swap_two_annihilators(self):
        result = BOSON.swap(LadderOp(2, False), LadderOp(1, False))
        assert result.terms == ((+1, (LadderOp(1, False), LadderOp(2, False))),)

    def test_zero_monomial_always_false(self):
        ops = (LadderOp(1, True), LadderOp(1, True))
        assert not BOSON.is_zero_monomial(ops)


class TestRulesComparison:
    def test_fermion_eq(self):
        assert FermionRules() == FermionRules()

    def test_fermion_ne_boson(self):
        assert FermionRules() != BosonRules()
