"""Public model facade and factory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from loguru import logger

from .models.apis import APIEndpoint, APIModel, APIType
from .models.base import AbstractModel
from .models.oauth import (
    AnthropicOAuthModel,
    KimiOAuthModel,
    OAuthConfig,
    OpenAIOAuthModel,
)


def create_model(
    model: str,
    *,
    oauth: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    api_type: APIType | str = APIType.OPENAI,
    endpoints: Iterable[APIEndpoint] | None = None,
    fallbacks: Iterable[object] = (),
    oauth_config: OAuthConfig | None = None,
    headers: Mapping[str, str] | None = None,
    request_overrides: Mapping[str, object] | None = None,
    request_drop_fields: Iterable[str] = (),
    reasoning: str | None = None,
    openrouter_providers: Iterable[str] = (),
) -> AbstractModel:
    """Build an explicit API model or provider-specific OAuth model.

    API and OAuth settings are mutually exclusive. ``reasoning`` is an
    optional provider-neutral effort value (for example, ``"high"``) which
    each supported harness translates to its native setting. No key or URL is
    ever sourced from the process environment.
    """
    fallback_tuple = tuple(fallbacks)
    provider_tuple = (
        (openrouter_providers,)
        if isinstance(openrouter_providers, str)
        else tuple(openrouter_providers)
    )
    if oauth is not None:
        if provider_tuple:
            raise ValueError("openrouter_providers requires API auth")
        if (
            api_url is not None
            or api_key is not None
            or endpoints is not None
            or headers is not None
            or request_overrides is not None
            or tuple(request_drop_fields)
        ):
            raise ValueError("oauth cannot be combined with API endpoint settings")
        provider = oauth.lower().replace("_", "-")
        classes = {
            "openai": OpenAIOAuthModel,
            "codex": OpenAIOAuthModel,
            "anthropic": AnthropicOAuthModel,
            "claude": AnthropicOAuthModel,
            "kimi": KimiOAuthModel,
            "moonshot": KimiOAuthModel,
        }
        try:
            cls = classes[provider]
        except KeyError as exc:
            raise ValueError(f"unsupported OAuth provider: {oauth!r}") from exc
        aliases = {
            OpenAIOAuthModel: {"openai", "codex"},
            AnthropicOAuthModel: {"anthropic", "claude"},
            KimiOAuthModel: {"kimi", "moonshot"},
        }
        if oauth_config is not None and oauth_config.provider.lower() not in aliases[cls]:
            raise ValueError(
                f"oauth_config provider {oauth_config.provider!r} does not match {oauth!r}"
            )
        logger.bind(component="model", provider=provider, model=model).debug(
            "Creating OAuth model: fallback_count={fallback_count}",
            fallback_count=len(fallback_tuple),
        )
        return cls(
            model,
            config=oauth_config,
            fallbacks=fallback_tuple,
            reasoning=reasoning,
        )

    if oauth_config is not None:
        raise ValueError("oauth_config requires oauth")
    if endpoints is not None:
        if api_url is not None or api_key is not None:
            raise ValueError("endpoints cannot be combined with api_url/api_key")
        return APIModel(
            model,
            endpoints=endpoints,
            headers=headers,
            request_overrides=request_overrides,
            request_drop_fields=request_drop_fields,
            fallbacks=fallback_tuple,
            reasoning=reasoning,
            openrouter_providers=provider_tuple,
        )
    if api_url is None or api_key is None:
        missing = (
            "api_url and api_key"
            if api_url is None and api_key is None
            else ("api_url" if api_url is None else "api_key")
        )
        raise ValueError(f"{missing} must be explicitly provided for API auth")
    logger.bind(component="model", model=model, provider="api").debug(
        "Creating explicit API model: api_type={api_type}, fallback_count={fallback_count}",
        api_type=APIType.parse(api_type).value,
        fallback_count=len(fallback_tuple),
    )
    return APIModel(
        model,
        endpoints=(APIEndpoint(APIType.parse(api_type), api_url, api_key),),
        headers=headers,
        request_overrides=request_overrides,
        request_drop_fields=request_drop_fields,
        fallbacks=fallback_tuple,
        reasoning=reasoning,
        openrouter_providers=provider_tuple,
    )


class Model:
    """Constructor-compatible factory retained as the simplest public API."""

    def __new__(cls, model: str, **kwargs: object) -> Any:
        return create_model(model, **kwargs)  # type: ignore[arg-type]


__all__ = ["APIEndpoint", "APIModel", "APIType", "AbstractModel", "Model", "create_model"]
