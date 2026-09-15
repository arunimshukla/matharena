from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..apis import APIType
from .native_credentials import kimi_oauth_credential
from .oauth import OAuthConfig, OAuthModel
from .rate_limits import _refresh_kimi_credential, poll_kimi_rate_limits


def _kimi_executable() -> str:
    executable = shutil.which("kimi")
    if executable is not None:
        return executable
    candidates = (
        Path.home() / ".harness-wrapper" / "clis" / "native" / "bin" / "kimi",
        Path(os.environ.get("KIMI_CODE_HOME", "~/.kimi-code")).expanduser() / "bin" / "kimi",
    )
    return str(next((path for path in candidates if path.is_file()), "kimi"))


def _kimi_logged_in() -> bool:
    try:
        credential = kimi_oauth_credential()
    except Exception:
        return False
    return bool(credential.access_token and (not credential.expired or credential.refresh_token))


def kimi_oauth_config(
    *,
    auto_login: bool = True,
    auto_relogin: bool = True,
    expiry_leeway: float = 30.0,
) -> OAuthConfig:
    # Kimi login is already a remote-friendly device-code flow.
    executable = _kimi_executable()
    return OAuthConfig(
        provider="kimi",
        login_command=(executable, "login"),
        credential_loader=kimi_oauth_credential,
        credential_refresh_hook=_refresh_kimi_credential,
        status_hook=_kimi_logged_in,
        rate_limit_hook=poll_kimi_rate_limits,
        available_models_command=(executable, "provider", "list", "--json"),
        auto_login=auto_login,
        auto_relogin=auto_relogin,
        expiry_leeway=expiry_leeway,
    )


class KimiOAuthModel(OAuthModel):
    # Kimi Code exposes an OpenAI-compatible model surface to wrappers.
    api_type = APIType.OPENAI
    compatible_harnesses = frozenset({"kimi-code"})

    def _parse_available_models(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, Mapping):
            return super()._parse_available_models(value)
        providers = value.get("providers")
        models = value.get("models")
        if not isinstance(providers, Mapping) or not isinstance(models, Mapping):
            return super()._parse_available_models(value)
        oauth_providers = {
            str(name)
            for name, config in providers.items()
            if isinstance(config, Mapping) and isinstance(config.get("oauth"), Mapping)
        }
        return tuple(
            str(name)
            for name, config in models.items()
            if isinstance(config, Mapping) and str(config.get("provider")) in oauth_providers
        )

    def cli_environment(self, api_type: APIType | str) -> dict[str, str]:
        self.assert_compatible(api_type)
        self.ensure_authenticated()
        return {}

    def __init__(
        self,
        model: str,
        *,
        config: OAuthConfig | None = None,
        fallbacks: Iterable[object] = (),
        reasoning: str | None = None,
    ) -> None:
        super().__init__(
            model,
            config=config or kimi_oauth_config(),
            fallbacks=fallbacks,
            reasoning=reasoning,
        )
