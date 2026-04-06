"""Constructor functions for operator expressions."""

from __future__ import annotations

from collections.abc import Hashable

from operator_algebra.algebra import BOSON, FERMION, PAULI, AlgebraRules
from operator_algebra.operators import Monomial, Op, OpSum


def _mode(*labels: Hashable) -> Hashable:
    return labels if len(labels) > 1 else labels[0]


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
    return OpSum({Monomial(ops=(Op(_mode(*labels), True),)): 1}, rules=rules)


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
    return OpSum({Monomial(ops=(Op(_mode(*labels), False),)): 1}, rules=rules)


def number(*labels: Hashable, rules: AlgebraRules = FERMION) -> OpSum:
    """Return the number operator create† * annihilate.

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


def sp(*labels: Hashable) -> OpSum:
    """Pauli raising operator σ+.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return create(*labels, rules=PAULI)


def sm(*labels: Hashable) -> OpSum:
    """Pauli lowering operator σ-.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return annihilate(*labels, rules=PAULI)


def sigma_plus(*labels: Hashable) -> OpSum:
    """Pauli raising operator σ+.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return sp(*labels)


def sigma_minus(*labels: Hashable) -> OpSum:
    """Pauli lowering operator σ-.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    return sm(*labels)


def sigma_x(*labels: Hashable) -> OpSum:
    """Pauli σ_x operator.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    mode = _mode(*labels)
    return OpSum(
        {Monomial(ops=(Op(mode, "x"),)): 1},
        rules=PAULI,
    )


def sigma_y(*labels: Hashable) -> OpSum:
    """Pauli σ_y operator.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    mode = _mode(*labels)
    return OpSum(
        {Monomial(ops=(Op(mode, "y"),)): 1},
        rules=PAULI,
    )


def sigma_z(*labels: Hashable) -> OpSum:
    """Pauli σ_z operator.

    Parameters
    ----------
    *labels : Hashable
        Mode label components.

    Returns
    -------
    OpSum

    """
    mode = _mode(*labels)
    return OpSum(
        {Monomial(ops=(Op(mode, "z"),)): 1},
        rules=PAULI,
    )
