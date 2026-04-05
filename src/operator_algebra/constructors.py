"""Constructor functions for operator expressions."""

from __future__ import annotations

from collections.abc import Hashable

from operator_algebra.algebra import BOSON, FERMION, AlgebraRules
from operator_algebra.types import LadderOp, Monomial, OpSum


def create(*labels: Hashable, rules: AlgebraRules = FERMION) -> OpSum:
    """Creation operator with the given mode label(s).

    If one label given: mode is that label.
    If multiple labels given: mode is the tuple of labels.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    mode = labels if len(labels) > 1 else labels[0]
    return OpSum({Monomial(ops=(LadderOp(mode, True),)): 1}, rules=rules)


def annihilate(*labels: Hashable, rules: AlgebraRules = FERMION) -> OpSum:
    """Annihilation operator.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    mode = labels if len(labels) > 1 else labels[0]
    return OpSum({Monomial(ops=(LadderOp(mode, False),)): 1}, rules=rules)


def number(*labels: Hashable, rules: AlgebraRules = FERMION) -> OpSum:
    """Number operator: create† * annihilate.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    return create(*labels, rules=rules) * annihilate(*labels, rules=rules)


def cdag(*labels: Hashable) -> OpSum:
    """Fermionic creation operator c†.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return create(*labels, rules=FERMION)


def c(*labels: Hashable) -> OpSum:
    """Fermionic annihilation operator c.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return annihilate(*labels, rules=FERMION)


def n(*labels: Hashable) -> OpSum:
    """Fermionic number operator n = c† c.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return number(*labels, rules=FERMION)


def bdag(*labels: Hashable) -> OpSum:
    """Bosonic creation operator b†.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return create(*labels, rules=BOSON)


def b(*labels: Hashable) -> OpSum:
    """Bosonic annihilation operator b.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return annihilate(*labels, rules=BOSON)


def nb(*labels: Hashable) -> OpSum:
    """Bosonic number operator n_b = b† b.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return number(*labels, rules=BOSON)


def identity(rules: AlgebraRules = FERMION) -> OpSum:
    """The identity operator (empty product with coefficient 1).

    Parameters
    ----------
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    return OpSum({Monomial(()): 1}, rules=rules)


def zero(rules: AlgebraRules = FERMION) -> OpSum:
    """The zero operator (empty sum).

    Parameters
    ----------
    rules : AlgebraRules, optional
        Algebra rules. Default: ``FERMION``.

    Returns
    -------
    OpSum

    """
    return OpSum({}, rules=rules)
