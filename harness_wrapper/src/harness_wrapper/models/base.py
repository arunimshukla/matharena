"""Common model contract used by harness implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .apis import APIType


def parse_reasoning(value: str | None) -> str | None:
    """Normalize an optional provider reasoning/effort value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("reasoning must be a string or None")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("reasoning must not be empty")
    return normalized


class AbstractModel(ABC):
    """Minimal model interface independent of any agent CLI."""

    model: str
    reasoning: str | None
    auth_mode: str
    fallbacks: tuple[object, ...]

    @abstractmethod
    def supported_endpoints(self) -> frozenset[str]:
        """Return the wire protocols this model can be passed to."""

    @abstractmethod
    def supported_api_types(self) -> frozenset[APIType]:
        """Typed equivalent of :meth:`supported_endpoints`."""

    @abstractmethod
    def assert_compatible(self, accepted: APIType | str | Iterable[APIType | str]) -> APIType:
        """Return the selected protocol or raise a descriptive ``ValueError``."""

    @abstractmethod
    def model_for(self, accepted: APIType | str | Iterable[APIType | str]) -> object:
        """Select this model or a compatible configured fallback."""
