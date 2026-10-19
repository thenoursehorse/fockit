"""Operators that act on density matrices, and the Lindblad generator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast, override

from fockit.algebra.operators import Monomial, Op, OpSum, as_opsum
from fockit.algebra.rules import AlgebraRules, SwapResult
from fockit.algebra.transforms import conjugate, hc


@dataclass(frozen=True)
class ConjugateRules:
    """Wraps an algebra for operators that multiply a density matrix from the right.

    The wrapped operators follow ``wrapped_rules`` with complex-conjugated
    matrices, and commute with the unwrapped ones.
    """

    wrapped_rules: AlgebraRules

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Swap the way the wrapped algebra does, then conjugate the coefficients."""
        a, b = _unwrapped(left), _unwrapped(right)
        if type(a.rules) is not type(b.rules):
            return SwapResult(terms=((1, (right, left)),))
        return SwapResult(
            terms=tuple(
                (conjugate(coeff), tuple(_wrapped(op) for op in ops))
                for coeff, ops in a.rules.swap(a, b).terms
            )
        )

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Use the wrapped algebra's normal order.

        Operators of different wrapped algebras are sorted by algebra name.
        """
        a, b = _unwrapped(left), _unwrapped(right)
        if type(a.rules) is not type(b.rules):
            return type(a.rules).__name__ < type(b.rules).__name__
        return a.rules.is_normal_ordered_pair(a, b)

    def dagger(self, op: Op) -> Op:
        """Take the adjoint with the wrapped algebra's rules."""
        plain = _unwrapped(op)
        return _wrapped(plain.rules.dagger(plain))

    @override
    def __repr__(self) -> str:
        """Show the wrapped rules, e.g. ``ConjugateRules(FermionRules())``."""
        return f"ConjugateRules({self.wrapped_rules!r})"


def left(op: OpSum | complex) -> OpSum:
    """Multiply a density matrix from the left: ``rho → op rho``.

    This just returns ``op``. It's here so expressions like
    ``left(N) - right(N)`` read clearly.
    """
    return as_opsum(op)


def right(op: OpSum | complex) -> OpSum:
    """Multiply a density matrix from the right: ``rho → rho op``.

    The result acts on the column index of ``rho``. Products come out in
    reverse order: ``right(A) * right(B)`` is ``right(B * A)``.
    """
    return OpSum(
        {
            Monomial(
                tuple(_wrapped(o.rules.dagger(o)) for o in reversed(mono.ops))
            ): coeff
            for mono, coeff in as_opsum(op).terms.items()
        }
    )


def liouvillian(H: OpSum | complex, jumps: Iterable[OpSum]) -> OpSum:
    """Build the Lindblad generator.

    It's the right-hand side of
    ``d rho/dt = -i[H, rho] + Σ_j (L_j rho L_j† - ½{L_j† L_j, rho})``.
    Use ``fockit.hilbert.doubled`` to build a basis for it.

    Parameters
    ----------
    H : OpSum
        Hermitian Hamiltonian.
    jumps : Iterable[OpSum]
        Jump operators with their rates folded in, e.g.
        ``[np.sqrt(gamma) * c(i) for i in sites]``.

    Returns
    -------
    OpSum

    Notes
    -----
    The result isn't normal ordered. For truncated bosons, normal ordering
    would make the trace of ``rho`` leak at ``n_max``.

    A jump that changes ``N``, like ``c``, only conserves
    ``left(N) - right(N)``. A jump that commutes with ``N``, like ``n``,
    also conserves ``left(N)`` and ``right(N)`` separately. ``to_matrix``
    silently drops transitions that leave the basis, so pick sectors that
    match the jumps.

    """
    out = -1j * (left(H) - right(H))
    for jump in jumps:
        decay = hc(jump) * jump
        out += left(jump) * right(hc(jump)) - 0.5 * (left(decay) + right(decay))
    return out


def _unwrapped(op: Op) -> Op:
    """Return the same operator in the algebra its ``ConjugateRules`` wraps."""
    return Op(op.mode, op.kind, cast(ConjugateRules, op.rules).wrapped_rules)


def _wrapped(op: Op) -> Op:
    """Return the same operator with its algebra wrapped in ``ConjugateRules``."""
    return Op(op.mode, op.kind, ConjugateRules(op.rules))
