from __future__ import annotations

import json
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from harness_wrapper.agent import Agent
from harness_wrapper.model import Model, create_model
from harness_wrapper.models import (
    AnthropicOAuthModel,
    APIEndpoint,
    APIModel,
    APIType,
    AuthenticationError,
    CredentialStore,
    KimiOAuthModel,
    OAuthConfig,
    OAuthCredential,
    OpenAIOAuthModel,
    RateLimit,
    RateLimitPollingUnsupported,
    RateLimitSnapshot,
    RateLimitTimeout,
)
from harness_wrapper.models.oauth.anthropic_model import anthropic_oauth_config
from harness_wrapper.models.oauth.kimi_model import kimi_oauth_config
from harness_wrapper.models.oauth.openai_model import openai_oauth_config
from harness_wrapper.models.oauth.rate_limits import (
    parse_anthropic_rate_limits,
    parse_kimi_rate_limits,
    parse_openai_rate_limits,
    poll_anthropic_rate_limits,
    poll_kimi_rate_limits,
    poll_openai_rate_limits,
)


def test_explicit_openai_api_model_ignores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid")
    model = Model("small", api_url="http://localhost:8000/v1/", api_key="explicit")
    assert isinstance(model, APIModel)
    assert model.supported_endpoints() == frozenset({"openai"})
    assert model.cli_environment("openai") == {
        "OPENAI_API_KEY": "explicit",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
    }


def test_openrouter_provider_allowlist_and_custom_headers() -> None:
    model = Model(
        "openrouter/z-ai/glm-5.3-flash",
        api_url="https://bifrost.invalid/v1",
        api_key="key",
        headers={
            "x-bf-vk": "key",
            "x-bf-passthrough-extra-params": "true",
        },
        openrouter_providers=("z-ai", "together"),
    )

    provider_filter = {"provider": {"only": ["z-ai", "together"]}}
    assert model.request_overrides() == provider_filter
    assert model.endpoint_for("openai").headers == {
        "x-bf-vk": "key",
        "x-bf-passthrough-extra-params": "true",
    }


def test_api_model_exposes_provider_request_drop_fields() -> None:
    model = Model(
        "gemini-test",
        api_url="https://provider.invalid/v1",
        api_key="key",
        request_drop_fields=("prompt_cache_key",),
    )

    assert model.request_drop_fields() == frozenset({"prompt_cache_key"})


def test_anthropic_api_model() -> None:
    model = create_model(
        "claude-test",
        api_url="https://proxy.invalid/anthropic",
        api_key="key",
        api_type="anthropic",
    )
    assert isinstance(model, APIModel)
    assert model.endpoint_for(APIType.ANTHROPIC).api_key == "key"
    assert model.cli_environment("anthropic")["ANTHROPIC_BASE_URL"].endswith("/anthropic")
    with pytest.raises(ValueError, match="does not support openai"):
        model.endpoint_for("openai")


def test_reasoning_is_parsed_by_api_factory_variants() -> None:
    direct = Model(
        "reasoner",
        api_url="https://api.invalid/v1",
        api_key="key",
        reasoning="  HIGH  ",
    )
    endpoints = Model(
        "reasoner",
        endpoints=(APIEndpoint(APIType.OPENAI, "https://api.invalid/v1", "key"),),
        reasoning="XHIGH",
    )
    custom_headers = Model(
        "reasoner",
        api_url="https://api.invalid/v1",
        api_key="key",
        headers={"x-bf-vk": "key"},
        reasoning="medium",
    )

    assert direct.reasoning == "high"
    assert endpoints.reasoning == "xhigh"
    assert custom_headers.reasoning == "medium"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "kimi"])
def test_reasoning_is_parsed_by_each_oauth_provider(provider: str) -> None:
    model = Model(
        "reasoner",
        oauth=provider,
        oauth_config=OAuthConfig(provider=provider, auto_login=False),
        reasoning="  Low ",
    )

    assert model.reasoning == "low"


@pytest.mark.parametrize("reasoning", ["", "   "])
def test_reasoning_rejects_empty_values(reasoning: str) -> None:
    with pytest.raises(ValueError, match="reasoning must not be empty"):
        Model(
            "reasoner",
            api_url="https://api.invalid/v1",
            api_key="key",
            reasoning=reasoning,
        )


@pytest.mark.parametrize("url", ["relative/v1", "ftp://host/v1", "", "localhost:8000"])
def test_invalid_api_urls(url: str) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        APIEndpoint(APIType.OPENAI, url, "key")


def test_api_credentials_required_and_repr_redacted() -> None:
    with pytest.raises(ValueError, match="api_url and api_key"):
        Model("tiny")
    with pytest.raises(ValueError, match="api_key"):
        Model("tiny", api_url="http://localhost/v1")
    endpoint = APIEndpoint(APIType.OPENAI, "https://example.invalid", "super-secret")
    assert "super-secret" not in repr(APIModel("tiny", endpoints=(endpoint,)))
    assert endpoint.redacted()["api_key"] == "***"


def test_redacted_endpoint_hides_sensitive_custom_headers() -> None:
    endpoint = APIEndpoint(
        APIType.OPENAI,
        "https://example.invalid",
        "key",
        {
            "Authorization": "Bearer secret",
            "X-API-Key": "also-secret",
            "X-Request-Source": "test-suite",
        },
    )
    assert endpoint.redacted()["headers"] == {
        "Authorization": "***",
        "X-API-Key": "***",
        "X-Request-Source": "test-suite",
    }


def test_multi_protocol_and_fallback_selection() -> None:
    both = APIModel(
        "both",
        endpoints=(
            APIEndpoint(APIType.OPENAI, "https://example.invalid/v1", "o"),
            APIEndpoint(APIType.ANTHROPIC, "https://example.invalid", "a"),
        ),
    )
    assert both.assert_compatible(["anthropic", "openai"]) is APIType.ANTHROPIC
    anthropic = Model("backup", api_url="https://a.invalid", api_key="a", api_type="anthropic")
    openai = Model("main", api_url="https://o.invalid", api_key="o", fallbacks=[anthropic])
    assert openai.model_for("openai") is openai
    assert openai.model_for("anthropic") is anthropic


def test_custom_api_headers_are_explicit_and_redacted() -> None:
    model = Model(
        "tiny",
        api_url="https://srlx.inf.ethz.ch/v1",
        api_key="x",
        headers={"x-bf-vk": "x"},
    )
    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://srlx.inf.ethz.ch/v1"
    assert endpoint.headers == {"x-bf-vk": "x"}
    assert endpoint.redacted()["headers"] == {"x-bf-vk": "***"}


def test_factory_rejects_conflicting_auth() -> None:
    config = OAuthConfig(provider="openai", auto_login=False)
    with pytest.raises(ValueError, match="cannot be combined"):
        Model("x", oauth="openai", oauth_config=config, api_url="https://x.invalid", api_key="x")
    with pytest.raises(ValueError, match="unsupported OAuth"):
        Model("x", oauth="unknown")
    with pytest.raises(ValueError, match="does not match"):
        Model(
            "x",
            oauth="openai",
            oauth_config=OAuthConfig(provider="anthropic", auto_login=False),
        )


def test_credential_store_round_trip_private_and_preserves_providers(tmp_path: Path) -> None:
    path = tmp_path / "private" / "auth.json"
    store = CredentialStore(path)
    first = OAuthCredential("access", "refresh", 200.0, {"scope": "all"})
    store.save("openai", first)
    store.save("anthropic", OAuthCredential(extra={"cli_managed": True}))
    assert store.load("openai") == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert set(json.loads(path.read_text())) == {"openai", "anthropic"}
    store.delete("openai")
    assert store.load("openai") is None
    assert store.load("anthropic") is not None
    assert "access" not in repr(first)
    assert "refresh" not in repr(first)


def test_corrupt_credential_store_has_context(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text("not json")
    with pytest.raises(AuthenticationError, match=str(path)):
        CredentialStore(path).load("openai")


def test_oauth_login_hook_persists(tmp_path: Path) -> None:
    calls: list[bool] = []

    def login() -> OAuthCredential:
        calls.append(True)
        return OAuthCredential("token", expires_at=10_000_000_000)

    config = OAuthConfig(
        provider="openai", store=CredentialStore(tmp_path / "auth.json"), login_hook=login
    )
    model = OpenAIOAuthModel("gpt", config=config)
    assert calls == [True]
    credential = model.credential
    assert credential is not None and credential.access_token == "token"
    second = OpenAIOAuthModel("gpt", config=config)
    second_credential = second.credential
    assert second_credential is not None and second_credential.access_token == "token"
    assert calls == [True]


def test_oauth_discovers_existing_cli_login_without_relogin(tmp_path: Path) -> None:
    login_calls = []
    config = OAuthConfig(
        provider="anthropic",
        store=CredentialStore(tmp_path / "auth.json"),
        login_hook=lambda: login_calls.append(True),
        status_hook=lambda: True,
    )
    model = AnthropicOAuthModel("haiku", config=config)
    assert model.credential is not None
    assert model.credential.extra["cli_managed"] is True
    assert login_calls == []


def test_oauth_discovery_imports_native_token_into_central_store(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    native = OAuthCredential(
        "native-access",
        "native-refresh",
        4_000_000_000,
        {"cli_managed": True},
    )
    config = OAuthConfig(
        provider="anthropic",
        store=store,
        status_hook=lambda: True,
        credential_loader=lambda: native,
    )
    model = AnthropicOAuthModel("haiku", config=config)
    assert model.credential == native
    assert store.load("anthropic") == native
    assert model.cli_environment("anthropic") == {"CLAUDE_CODE_OAUTH_TOKEN": "native-access"}


def test_oauth_model_discovery_is_cached_and_includes_selected_model(tmp_path: Path) -> None:
    calls: list[object] = []

    def discover(credential: OAuthCredential | None) -> object:
        calls.append(credential)
        return ["gpt-fast", "gpt-large", "gpt-fast"]

    model = OpenAIOAuthModel(
        "gpt-selected",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            auto_login=False,
            available_models_hook=discover,
        ),
    )

    assert model.available_models() == ("gpt-selected", "gpt-fast", "gpt-large")
    assert model.available_models() == ("gpt-selected", "gpt-fast", "gpt-large")
    assert len(calls) == 1
    model.available_models(refresh=True)
    assert len(calls) == 2


def test_provider_oauth_catalog_parsers_filter_hidden_and_non_oauth_models(
    tmp_path: Path,
) -> None:
    openai = OpenAIOAuthModel(
        "gpt-main",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "openai.json"),
            auto_login=False,
            available_models_hook=lambda credential: {
                "models": [
                    {"slug": "gpt-visible", "visibility": "list"},
                    {"slug": "gpt-internal", "visibility": "hide"},
                ]
            },
        ),
    )
    assert openai.available_models() == ("gpt-main", "gpt-visible")

    kimi = KimiOAuthModel(
        "kimi-code/k3",
        config=OAuthConfig(
            provider="kimi",
            store=CredentialStore(tmp_path / "kimi.json"),
            auto_login=False,
            available_models_hook=lambda credential: {
                "providers": {
                    "managed": {"oauth": {"storage": "file"}},
                    "api": {"apiKey": "redacted"},
                },
                "models": {
                    "kimi-code/k3": {"provider": "managed"},
                    "kimi-code/fast": {"provider": "managed"},
                    "external/model": {"provider": "api"},
                },
            },
        ),
    )
    assert kimi.available_models() == ("kimi-code/k3", "kimi-code/fast")


def test_cli_managed_oauth_is_revalidated_and_relogged_in(tmp_path: Path) -> None:
    logged_in = [True]
    login_calls: list[bool] = []

    def login() -> OAuthCredential:
        login_calls.append(True)
        return OAuthCredential("fresh", expires_at=4_000_000_000)

    config = OAuthConfig(
        provider="openai",
        store=CredentialStore(tmp_path / "auth.json"),
        status_hook=lambda: logged_in[0],
        login_hook=login,
    )
    model = OpenAIOAuthModel("gpt", config=config)
    logged_in[0] = False
    assert model.ensure_authenticated().access_token == "fresh"
    assert login_calls == [True]


def test_cli_managed_logout_respects_disabled_relogin(tmp_path: Path) -> None:
    logged_in = [True]
    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            status_hook=lambda: logged_in[0],
            auto_relogin=False,
        ),
    )
    logged_in[0] = False
    with pytest.raises(AuthenticationError, match="no longer valid"):
        model.ensure_authenticated()


def test_oauth_auto_relogin_on_expiry(tmp_path: Path) -> None:
    sequence = iter(
        (
            OAuthCredential("old", expires_at=2_000_000_000),
            OAuthCredential("new", expires_at=4_000_000_000),
        )
    )
    config = OAuthConfig(
        provider="openai",
        store=CredentialStore(tmp_path / "auth.json"),
        login_hook=lambda: next(sequence),
    )
    model = OpenAIOAuthModel("gpt", config=config)
    assert model.ensure_authenticated(now=3_000_000_000).access_token == "new"


@pytest.mark.parametrize(
    ("model_class", "provider"),
    [
        (OpenAIOAuthModel, "openai"),
        (AnthropicOAuthModel, "anthropic"),
        (KimiOAuthModel, "kimi"),
    ],
)
def test_expired_native_credential_cannot_bypass_relogin(
    tmp_path: Path, model_class: type, provider: str
) -> None:
    expired = OAuthCredential(
        "expired",
        "refresh",
        expires_at=1,
        extra={"cli_managed": True},
    )
    login_calls: list[bool] = []
    model = model_class(
        "model",
        config=OAuthConfig(
            provider=provider,
            store=CredentialStore(tmp_path / f"{provider}.json"),
            auto_login=False,
            status_hook=lambda: True,
            credential_loader=lambda: expired,
            login_hook=lambda: (
                login_calls.append(True) or OAuthCredential("fresh", expires_at=4_000_000_000)
            ),
        ),
    )
    model.credential = expired

    assert model.ensure_authenticated(now=100).access_token == "fresh"
    assert login_calls == [True]


def test_refresh_hook_updates_memory_and_central_store(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    expired = OAuthCredential("old", "refresh", expires_at=1)
    store.save("kimi", expired)
    refreshed = OAuthCredential("fresh", "next", expires_at=4_000_000_000)
    model = KimiOAuthModel(
        "kimi",
        config=OAuthConfig(
            provider="kimi",
            store=store,
            auto_login=False,
            credential_refresh_hook=lambda credential: refreshed,
        ),
    )

    assert model.ensure_authenticated(now=100) == refreshed
    assert model.credential == refreshed
    assert store.load("kimi") == refreshed


@pytest.mark.parametrize(
    ("model_class", "provider"),
    [
        (OpenAIOAuthModel, "openai"),
        (AnthropicOAuthModel, "anthropic"),
        (KimiOAuthModel, "kimi"),
    ],
)
def test_rate_limit_401_relogs_in_and_retries_once(
    tmp_path: Path, model_class: type, provider: str
) -> None:
    store = CredentialStore(tmp_path / f"{provider}.json")
    store.save(provider, OAuthCredential("old", expires_at=4_000_000_000))
    polls = [0]
    logins: list[bool] = []

    def poll(credential: OAuthCredential | None) -> object:
        polls[0] += 1
        if polls[0] == 1:
            raise AuthenticationError("HTTP 401")
        assert credential is not None and credential.access_token == "fresh"
        return {"limits": []}

    model = model_class(
        "model",
        config=OAuthConfig(
            provider=provider,
            store=store,
            auto_login=False,
            login_hook=lambda: (
                logins.append(True) or OAuthCredential("fresh", expires_at=4_000_000_000)
            ),
            rate_limit_hook=poll,
        ),
    )

    assert not model.poll_rate_limits().limited
    assert polls == [2]
    assert logins == [True]


def test_oauth_can_disable_auto_relogin(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("old", expires_at=1))
    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(provider="openai", store=store, auto_login=False, auto_relogin=False),
    )
    with pytest.raises(AuthenticationError, match="expired"):
        model.ensure_authenticated(now=2)


def test_provider_compatibility_is_descriptive(tmp_path: Path) -> None:
    config = OAuthConfig(
        provider="anthropic", store=CredentialStore(tmp_path / "a.json"), auto_login=False
    )
    model = AnthropicOAuthModel("claude", config=config)
    assert model.assert_compatible("anthropic") is APIType.ANTHROPIC
    with pytest.raises(ValueError, match="anthropic-only"):
        model.assert_compatible("openai")


@pytest.mark.parametrize(
    ("oauth", "provider", "harness"),
    [
        ("anthropic", "anthropic", "kimi"),
        ("kimi", "kimi", "codex"),
    ],
)
def test_cli_owned_oauth_rejects_other_harnesses(
    tmp_path: Path, oauth: str, provider: str, harness: str
) -> None:
    model = Model(
        "test-model",
        oauth=oauth,
        oauth_config=OAuthConfig(
            provider=provider,
            store=CredentialStore(tmp_path / f"{provider}.json"),
            auto_login=False,
        ),
    )
    with pytest.raises(ValueError, match="cannot authenticate"):
        Agent(type=harness, model=model, executable="/bin/true")  # type: ignore[abstract]


def test_openai_oauth_accepts_kimi_harness(tmp_path: Path) -> None:
    model = Model(
        "gpt-test",
        oauth="openai",
        oauth_config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "openai.json"),
            auto_login=False,
        ),
    )
    agent = Agent(  # type: ignore[abstract]
        type="kimi", model=model, executable="/bin/true"
    )
    assert agent.harness_name == "kimi-code"


def test_rate_limit_parser_accepts_claude_camel_case() -> None:
    snapshot = RateLimitSnapshot.parse(
        {"limits": [{"rateLimitType": "five_hour", "status": "rejected", "resetsAt": 123.0}]}
    )
    assert snapshot.limits == (RateLimit("five_hour", 123.0, limited=True),)
    assert snapshot.limited and snapshot.next_reset_at == 123.0


def test_rate_limit_parser_accepts_single_mapping_and_string_false() -> None:
    rejected = RateLimitSnapshot.parse(
        {"status": "rejected", "rate_limit_type": "five_hour", "resets_at": 123}
    )
    available = RateLimitSnapshot.parse({"name": "weekly", "limited": "false"})
    assert rejected.limited and rejected.limits[0].name == "five_hour"
    assert not available.limited


def test_default_oauth_clients_configure_rate_limit_polling() -> None:
    assert openai_oauth_config(auto_login=False).rate_limit_hook is poll_openai_rate_limits
    assert anthropic_oauth_config(auto_login=False).rate_limit_hook is poll_anthropic_rate_limits
    assert kimi_oauth_config(auto_login=False).rate_limit_hook is poll_kimi_rate_limits


def test_anthropic_rate_limit_parser_normalizes_usage_windows() -> None:
    snapshot = parse_anthropic_rate_limits(
        {
            "five_hour": {"utilization": 12.5, "resets_at": "2030-01-01T00:00:00Z"},
            "seven_day": {"utilization": 100, "resets_at": 2_000_000_000},
            "seven_day_opus": {"utilization": 50, "resets_at": 2_000_000_100},
        }
    )
    assert [limit.name for limit in snapshot.limits] == [
        "five_hour",
        "seven_day",
        "seven_day_opus",
    ]
    assert snapshot.limits[0].utilization == 0.125
    assert snapshot.limits[0].resets_at == 1_893_456_000
    assert snapshot.limits[1].limited


def test_openai_rate_limit_parser_prefers_documented_multi_bucket_view() -> None:
    snapshot = parse_openai_rate_limits(
        {
            "rateLimits": {
                "limitId": "legacy",
                "primary": {"usedPercent": 99, "windowDurationMins": 5},
            },
            "rateLimitsByLimitId": {
                "codex": {
                    "limitId": "codex",
                    "primary": {
                        "usedPercent": 25,
                        "windowDurationMins": 300,
                        "resetsAt": 2_000_000_000,
                    },
                    "secondary": {
                        "usedPercent": 100,
                        "windowDurationMins": 10080,
                        "resetsAt": 2_000_000_100,
                    },
                    "rateLimitReachedType": "secondary",
                }
            },
        }
    )
    assert snapshot.limits == (
        RateLimit("five_hour", 2_000_000_000, 0.25, False),
        RateLimit("seven_day", 2_000_000_100, 1.0, True),
    )


def test_kimi_rate_limit_parser_includes_weekly_rolling_and_monthly() -> None:
    snapshot = parse_kimi_rate_limits(
        {
            "usage": {
                "limit": "100",
                "used": "25",
                "remaining": "75",
                "resetTime": "2030-01-01T00:00:00.123456789Z",
            },
            "limits": [
                {
                    "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
                    "detail": {"limit": 10, "remaining": 0},
                }
            ],
            "totalQuota": {"limit": 200, "remaining": 100},
        }
    )
    assert [limit.name for limit in snapshot.limits] == [
        "seven_day",
        "five_hour",
        "monthly",
    ]
    assert snapshot.limits[0].utilization == 0.25
    assert snapshot.limits[1].limited
    assert snapshot.limits[2].utilization == 0.5


def test_kimi_rate_limit_poll_refreshes_expired_oauth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = OAuthCredential("old", "refresh", expires_at=1)
    fresh = OAuthCredential("fresh", "next-refresh", expires_at=4_000_000_000)
    refreshed: list[OAuthCredential] = []

    def refresh(credential: OAuthCredential) -> OAuthCredential:
        refreshed.append(credential)
        return fresh

    def get_json(url: str, token: str, **kwargs: object) -> object:
        assert token == "fresh"
        return {"usage": {"limit": 100, "remaining": 50}}

    monkeypatch.setattr(
        "harness_wrapper.models.oauth.rate_limits._refresh_kimi_credential", refresh
    )
    monkeypatch.setattr("harness_wrapper.models.oauth.rate_limits._get_json", get_json)

    snapshot = poll_kimi_rate_limits(old)
    assert refreshed == [old]
    assert snapshot.limits == (RateLimit("seven_day", None, 0.5, False),)


def test_rate_limit_json_command_adapter(tmp_path: Path) -> None:
    config = OAuthConfig(
        provider="openai",
        store=CredentialStore(tmp_path / "auth.json"),
        login_hook=lambda: OAuthCredential("ok", expires_at=4_000_000_000),
        rate_limit_command=(
            sys.executable,
            "-c",
            'print(\'{"limits": [{"name": "weekly", "utilization": 0.25}]}\')',
        ),
    )
    snapshot = OpenAIOAuthModel("gpt", config=config).poll_rate_limits()
    assert snapshot.limits[0].name == "weekly"
    assert snapshot.limits[0].utilization == 0.25


def test_rate_limit_polling_without_adapter_is_explicit(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("ok", expires_at=4_000_000_000))
    model = OpenAIOAuthModel(
        "gpt", config=OAuthConfig(provider="openai", store=store, auto_login=False)
    )
    with pytest.raises(RateLimitPollingUnsupported, match="no rate-limit polling adapter"):
        model.poll_rate_limits()


def test_rate_limit_command_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("ok", expires_at=4_000_000_000))
    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            rate_limit_command=("poll-usage",),
            rate_limit_command_timeout=10,
        ),
    )

    def timeout(*args: object, **kwargs: object) -> object:
        assert kwargs["timeout"] == 2
        raise __import__("subprocess").TimeoutExpired("poll-usage", 2)

    monkeypatch.setattr("harness_wrapper.models.oauth.oauth.subprocess.run", timeout)
    with pytest.raises(RateLimitTimeout, match="timed out polling"):
        model.poll_rate_limits(timeout=2)


def test_rate_limit_hook_receives_remaining_timeout(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("ok", expires_at=4_000_000_000))
    timeouts: list[float | None] = []

    def poll(credential: OAuthCredential | None, *, timeout: float | None = None) -> object:
        timeouts.append(timeout)
        return {"limits": []}

    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            rate_limit_hook=poll,
        ),
    )

    model.poll_rate_limits(timeout=2.5)
    assert timeouts == [2.5]


def test_rate_limited_oauth_selects_compatible_api_fallback(tmp_path: Path) -> None:
    fallback = Model("api", api_url="https://example.invalid/v1", api_key="key")
    oauth = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            login_hook=lambda: OAuthCredential("ok", expires_at=4_000_000_000),
            rate_limit_hook=lambda credential: [{"name": "five_hour", "limited": True}],
        ),
        fallbacks=(fallback,),
    )
    assert oauth.model_for("openai") is fallback


def test_polling_failure_selects_compatible_api_fallback(tmp_path: Path) -> None:
    fallback = Model("api", api_url="https://example.invalid/v1", api_key="key")
    oauth = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            login_hook=lambda: OAuthCredential("ok", expires_at=4_000_000_000),
            rate_limit_hook=lambda credential: (_ for _ in ()).throw(
                RuntimeError("usage endpoint unavailable")
            ),
        ),
        fallbacks=(fallback,),
    )

    assert oauth.model_for("openai") is fallback


def test_unavailable_oauth_selects_compatible_api_fallback(tmp_path: Path) -> None:
    fallback = Model("api", api_url="https://example.invalid/v1", api_key="key")
    oauth = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            auto_login=False,
            auto_relogin=False,
        ),
        fallbacks=(fallback,),
    )
    assert oauth.model_for("openai") is fallback


def test_login_hook_failure_selects_compatible_api_fallback(tmp_path: Path) -> None:
    fallback = Model("api", api_url="https://example.invalid/v1", api_key="key")

    def fail_login() -> OAuthCredential:
        raise RuntimeError("browser login was cancelled")

    oauth = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            login_hook=fail_login,
        ),
        fallbacks=(fallback,),
    )
    assert oauth.model_for("openai") is fallback


def test_auto_wait_polls_and_resumes(tmp_path: Path) -> None:
    now = [100.0]
    poll_values: list[dict[str, object]] = [
        {"limits": [{"name": "five_hour", "limited": True, "resets_at": 103}]},
        {"limits": []},
    ]
    polls: Iterator[dict[str, object]] = iter(poll_values)
    config = OAuthConfig(
        provider="openai",
        store=CredentialStore(tmp_path / "auth.json"),
        login_hook=lambda: OAuthCredential("ok", expires_at=1000),
        rate_limit_hook=lambda credential: next(polls),
    )
    model = OpenAIOAuthModel("gpt", config=config)

    def sleep(delay: float) -> None:
        now[0] += delay

    assert (
        model.wait_and_resume(
            lambda: "resumed", timeout=20, poll_interval=10, sleep=sleep, clock=lambda: now[0]
        )
        == "resumed"
    )
    assert now[0] == 103


@pytest.mark.parametrize(
    ("reset_delay", "expected_sleep"),
    ((15 * 60, 15 * 60), (2 * 60 * 60, 60 * 60)),
)
def test_auto_wait_uses_reset_time_with_hourly_safety_poll(
    tmp_path: Path, reset_delay: float, expected_sleep: float
) -> None:
    now = [100.0]
    polls: Iterator[object] = iter(
        (
            {
                "limits": [
                    {
                        "name": "five_hour",
                        "limited": True,
                        "resets_at": now[0] + reset_delay,
                    }
                ]
            },
            {"limits": []},
        )
    )
    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=CredentialStore(tmp_path / "auth.json"),
            login_hook=lambda: OAuthCredential("ok", expires_at=10_000),
            rate_limit_hook=lambda credential: next(polls),
        ),
    )
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    model.auto_wait(sleep=sleep, clock=lambda: now[0])

    assert sleeps == [expected_sleep]


def test_oauth_operations_emit_secret_safe_structured_logs(tmp_path: Path) -> None:
    secret_token = "oauth-secret-that-must-never-be-logged"
    polls: Iterator[object] = iter(
        (
            {"limits": [{"name": "five_hour", "limited": True}]},
            {"limits": []},
        )
    )
    records: list[Any] = []
    sink = logger.add(records.append, level="DEBUG")
    try:
        model = OpenAIOAuthModel(
            "gpt",
            config=OAuthConfig(
                provider="openai",
                store=CredentialStore(tmp_path / "auth.json"),
                login_hook=lambda: OAuthCredential(secret_token, expires_at=4_000_000_000),
                rate_limit_hook=lambda credential: next(polls),
            ),
        )
        model.auto_wait(poll_interval=1, sleep=lambda delay: None)
    finally:
        logger.remove(sink)

    messages = "\n".join(str(record) for record in records)
    assert "Starting interactive OAuth login" in messages
    assert "OAuth rate limits polled" in messages
    assert "OAuth capacity exhausted" in messages
    assert "OAuth capacity is available" in messages
    assert secret_token not in messages
    assert any(record.record["extra"].get("provider") == "openai" for record in records)


def test_auto_wait_relogs_in_when_token_expires_during_sleep(tmp_path: Path) -> None:
    now = [1_999_999_968.0]
    polls: Iterator[object] = iter(
        (
            {
                "limits": [
                    {
                        "name": "five_hour",
                        "limited": True,
                        "resets_at": 2_000_000_001,
                    }
                ]
            },
            {"limits": []},
        )
    )
    expired = OAuthCredential(
        "old",
        "refresh",
        expires_at=2_000_000_000,
        extra={"cli_managed": True},
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", expired)
    logins: list[bool] = []
    model = OpenAIOAuthModel(
        "gpt",
        config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            status_hook=lambda: True,
            credential_loader=lambda: expired,
            login_hook=lambda: (
                logins.append(True) or OAuthCredential("fresh", expires_at=4_000_000_000)
            ),
            rate_limit_hook=lambda credential: next(polls),
        ),
    )

    def sleep(delay: float) -> None:
        now[0] += delay

    model.auto_wait(timeout=50, poll_interval=40, sleep=sleep, clock=lambda: now[0])

    assert logins == [True]
    assert model.credential is not None and model.credential.access_token == "fresh"


def test_auto_wait_timeout(tmp_path: Path) -> None:
    now = [0.0]
    config = OAuthConfig(
        provider="openai",
        store=CredentialStore(tmp_path / "auth.json"),
        login_hook=lambda: OAuthCredential("ok", expires_at=1000),
        rate_limit_hook=lambda credential: [{"name": "weekly", "limited": True}],
    )
    model = OpenAIOAuthModel("gpt", config=config)

    def sleep(delay: float) -> None:
        now[0] += delay

    with pytest.raises(RateLimitTimeout):
        model.auto_wait(timeout=4, poll_interval=3, sleep=sleep, clock=lambda: now[0])
