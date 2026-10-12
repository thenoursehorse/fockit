"""Shared typing helpers."""

from __future__ import annotations

from collections.abc import Hashable
from typing import (
    Protocol,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    cast,
    override,
)

type Mode = Hashable


class SymbolicCoefficient(Protocol):
    """Protocol for symbolic coefficient objects."""

    @override
    def __hash__(self) -> int: ...

    def __add__(self, other: object, /) -> object: ...

    def __radd__(self, other: object, /) -> object: ...

    def __sub__(self, other: object, /) -> object: ...

    def __rsub__(self, other: object, /) -> object: ...

    def __mul__(self, other: object, /) -> object: ...

    def __rmul__(self, other: object, /) -> object: ...

    def __neg__(self) -> object: ...

    def __abs__(self) -> object: ...

    @override
    def __eq__(self, other: object, /) -> bool: ...

    def conjugate(self) -> object: ...


type Coefficient = int | float | complex | SymbolicCoefficient
type ComplexConvertible = complex | SupportsComplex | SupportsFloat | SupportsIndex


def as_coefficient(value: object) -> Coefficient:
    """Cast a coefficient result back to ``Coefficient``."""
    return cast(Coefficient, value)


def as_complex(value: Coefficient) -> complex:
    """Convert a coefficient to ``complex``."""
    return complex(cast(ComplexConvertible, value))
