"""Explicit API endpoint models.

The module deliberately never reads environment variables.  ``cli_environment``
only translates values which were supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from urllib.parse import urlsplit

from .base import AbstractModel, parse_reasoning


class APIType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

    @classmethod
    def parse(cls, value: APIType | str) -> APIType:
        if isinstance(value, cls):
            return value
        aliases = {
            "claude": cls.ANTHROPIC,
            "claude-code": cls.ANTHROPIC,
            "codex": cls.OPENAI,
            "codex-cli": cls.OPENAI,
            "kimi": cls.OPENAI,
            "kimi-code": cls.OPENAI,
        }
        try:
            normalized = value.lower().replace("_", "-").replace("-compatible", "")
            return aliases[normalized] if normalized in aliases else cls(normalized)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"unknown API type: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class APIEndpoint:
    """One wire-compatible endpoint and its explicit credential."""

    api_type: APIType
    url: str
    api_key: str
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_url must be an absolute http(s) URL")
        if not self.api_key:
            raise ValueError("api_key must be explicitly provided")
        object.__setattr__(self, "url", self.url.rstrip("/"))
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    def cli_environment(self) -> dict[str, str]:
        """Return the conventional variables a compatible CLI expects."""
        if self.api_type is APIType.OPENAI:
            return {"OPENAI_BASE_URL": self.url, "OPENAI_API_KEY": self.api_key}
        if self.api_type is APIType.ANTHROPIC:
            return {"ANTHROPIC_BASE_URL": self.url, "ANTHROPIC_API_KEY": self.api_key}
        return {"GOOGLE_GEMINI_BASE_URL": self.url, "GEMINI_API_KEY": self.api_key}

    def redacted(self) -> dict[str, object]:
        sensitive_fragments = (
            "authorization",
            "api-key",
            "apikey",
            "token",
            "secret",
            "cookie",
            "x-bf-vk",
        )
        return {
            "api_type": self.api_type.value,
            "url": self.url,
            "api_key": "***",
            "headers": {
                name: (
                    "***"
                    if any(fragment in name.lower() for fragment in sensitive_fragments)
                    else value
                )
                for name, value in self.headers.items()
            },
        }


class APIModel(AbstractModel):
    """A model served by one or more API-compatible endpoints.

    Constructor-level headers apply to every endpoint. Headers configured on
    an individual endpoint take precedence over common headers with the same
    name.
    """

    auth_mode = "api"
    oauth = None

    def __init__(
        self,
        model: str,
        *,
        endpoints: Iterable[APIEndpoint],
        headers: Mapping[str, str] | None = None,
        request_overrides: Mapping[str, object] | None = None,
        request_drop_fields: Iterable[str] = (),
        fallbacks: Iterable[object] = (),
        reasoning: str | None = None,
        openrouter_providers: Iterable[str] = (),
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        common_headers = dict(headers or {})
        by_type: dict[APIType, APIEndpoint] = {}
        for endpoint in endpoints:
            if endpoint.api_type in by_type:
                raise ValueError(f"duplicate {endpoint.api_type.value} endpoint")
            if common_headers:
                endpoint = APIEndpoint(
                    endpoint.api_type,
                    endpoint.url,
                    endpoint.api_key,
                    {**common_headers, **endpoint.headers},
                )
            by_type[endpoint.api_type] = endpoint
        if not by_type:
            raise ValueError("at least one endpoint is required")
        self.model = model
        self.reasoning = parse_reasoning(reasoning)
        self._endpoints = MappingProxyType(by_type)
        self._request_overrides = MappingProxyType(dict(request_overrides or {}))
        self._request_drop_fields = frozenset(str(field) for field in request_drop_fields)
        if any(not field for field in self._request_drop_fields):
            raise ValueError("request_drop_fields must contain non-empty strings")
        self.fallbacks = tuple(fallbacks)
        self.openrouter_providers = (
            (openrouter_providers,)
            if isinstance(openrouter_providers, str)
            else tuple(openrouter_providers)
        )
        if any(
            not isinstance(provider, str) or not provider.strip()
            for provider in self.openrouter_providers
        ):
            raise ValueError("openrouter provider names must be non-empty strings")

    def request_overrides(self) -> Mapping[str, object]:
        overrides = dict(self._request_overrides)
        if self.openrouter_providers:
            overrides["provider"] = {"only": list(self.openrouter_providers)}
        return overrides

    def request_drop_fields(self) -> frozenset[str]:
        """Return client-specific request fields that this endpoint must not receive."""

        return self._request_drop_fields

    @property
    def provider(self) -> str:
        return "api"

    @property
    def api_url(self) -> str:
        return next(iter(self._endpoints.values())).url

    @property
    def api_key(self) -> str:
        return next(iter(self._endpoints.values())).api_key

    def supported_api_types(self) -> frozenset[APIType]:
        return frozenset(self._endpoints)

    def supported_endpoints(self) -> frozenset[str]:
        return frozenset(item.value for item in self._endpoints)

    def endpoint_for(self, api_type: APIType | str) -> APIEndpoint:
        requested = APIType.parse(api_type)
        try:
            return self._endpoints[requested]
        except KeyError as exc:
            supported = ", ".join(sorted(self.supported_endpoints()))
            raise ValueError(
                f"model {self.model!r} does not support {requested.value}; "
                f"supported endpoint types: {supported}"
            ) from exc

    def assert_compatible(self, accepted: APIType | str | Iterable[APIType | str]) -> APIType:
        if isinstance(accepted, str | APIType):
            candidates: tuple[APIType, ...] = (APIType.parse(accepted),)
        else:
            candidates = tuple(APIType.parse(item) for item in accepted)
        for candidate in candidates:
            if candidate in self._endpoints:
                return candidate
        wanted = ", ".join(item.value for item in candidates) or "none"
        have = ", ".join(sorted(self.supported_endpoints()))
        raise ValueError(f"incompatible API types: harness accepts {wanted}; model provides {have}")

    def cli_environment(self, api_type: APIType | str) -> dict[str, str]:
        return self.endpoint_for(api_type).cli_environment()

    def cli_args(self, api_type: APIType | str) -> tuple[str, ...]:
        self.assert_compatible(api_type)
        return ()

    def model_for(self, accepted: APIType | str | Iterable[APIType | str]) -> object:
        """Return this model or the first compatible configured fallback."""
        try:
            self.assert_compatible(accepted)
            return self
        except ValueError as original:
            for fallback in self.fallbacks:
                check = getattr(fallback, "assert_compatible", None)
                if check is None:
                    continue
                try:
                    check(accepted)
                    return fallback
                except ValueError:
                    pass
            raise original

    def __repr__(self) -> str:
        kinds = ",".join(sorted(self.supported_endpoints()))
        return f"APIModel(model={self.model!r}, endpoints={kinds!r})"
