"""Superoperators on density matrices, and the Lindblad generator.

A density matrix ``rho = Σ rho_mn |m⟩⟨n|`` is a state on twice the modes: the
original modes hold the row index ``m``, and a conjugate copy of each mode
holds the column index ``n``. Then ``A rho B`` is ``A`` on the original modes
times ``Bᵀ = (B†)*`` on the copy, so superoperators are ordinary ``OpSum``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from fockit.algebra.operators import Monomial, Op, OpSum, as_opsum
from fockit.algebra.rules import AlgebraRules, SwapResult
from fockit.algebra.transforms import conjugate, hc


@dataclass(frozen=True)
class ConjugateRules:
    """Algebra of the complex-conjugated matrices of ``inner``'s operators.

    If ``X Y = Σ c Z`` then ``X* Y* = Σ c̄ Z*``, so swaps are ``inner``'s with
    conjugated coefficients. Its operators commute with ``inner``'s.
    """

    inner: AlgebraRules

    def swap(self, left: Op, right: Op) -> SwapResult:
        """Swap as ``inner`` does, with conjugated coefficients."""
        a, b = _original(left), _original(right)
        if type(a.rules) is not type(b.rules):
            return SwapResult(terms=((1, (right, left)),))
        return SwapResult(
            terms=tuple(
                (conjugate(coeff), tuple(_copy(op) for op in ops))
                for coeff, ops in a.rules.swap(a, b).terms
            )
        )

    def is_normal_ordered_pair(self, left: Op, right: Op) -> bool:
        """Order as ``inner`` does; copies of different algebras by name."""
        a, b = _original(left), _original(right)
        if type(a.rules) is not type(b.rules):
            return type(a.rules).__name__ < type(b.rules).__name__
        return a.rules.is_normal_ordered_pair(a, b)

    def dagger(self, op: Op) -> Op:
        """Adjoint as ``inner`` gives it, since ``(X*)† = (X†)*``."""
        original = _original(op)
        return _copy(original.rules.dagger(original))


def left(op: OpSum | complex) -> OpSum:
    """Multiply a density matrix by ``op`` from the left: ``rho → op rho``.

    This is ``op`` itself, acting on the original modes.
    """
    return as_opsum(op)


def right(op: OpSum | complex) -> OpSum:
    """Multiply a density matrix by ``op`` from the right: ``rho → rho op``.

    Acts on the conjugate copy of each mode. Products compose in reverse:
    ``right(A) * right(B) == right(B * A)``.
    """
    return OpSum(
        {
            Monomial(tuple(_copy(o.rules.dagger(o)) for o in reversed(mono.ops))): coeff
            for mono, coeff in as_opsum(op).terms.items()
        }
    )


def liouvillian(H: OpSum | complex, jumps: Iterable[OpSum]) -> OpSum:
    """Lindblad generator, the right-hand side of the master equation.

    ``d rho/dt = -i[H, rho] + Σ_j (L_j rho L_j† - ½{L_j†L_j, rho})``, acting
    on the original modes and their conjugate copies (see
    ``fockit.hilbert.doubled``).

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
    The result is as written, not normal ordered. Truncated bosons break
    ``b b† = 1 + b†b`` at ``n_max``, so normal ordering ``L_j†L_j`` there no
    longer conserves the trace of ``rho``.

    Jumps that change a quantity ``N`` keep only ``left(N) - right(N)``
    (weak symmetry); only jumps that commute with ``N`` also keep
    ``left(N)`` and ``right(N)`` separately (strong). ``to_matrix`` drops
    transitions out of the basis silently, so a sector too strong for the
    jumps gives a wrong generator.

    """
    out = -1j * (left(H) - right(H))
    for jump in jumps:
        decay = hc(jump) * jump
        out += left(jump) * right(hc(jump)) - 0.5 * (left(decay) + right(decay))
    return out


def _original(op: Op) -> Op:
    """Return the operator ``op`` is the conjugate copy of."""
    return Op(op.mode, op.kind, cast(ConjugateRules, op.rules).inner)


def _copy(op: Op) -> Op:
    """Return the conjugate copy of ``op``."""
    return Op(op.mode, op.kind, ConjugateRules(op.rules))
