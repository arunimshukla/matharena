from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..apis import APIType
from .native_credentials import codex_oauth_credential
from .oauth import OAuthConfig, OAuthModel
from .rate_limits import poll_openai_rate_limits


def _codex_executable() -> str:
    executable = shutil.which("codex")
    if executable is not None:
        return executable
    private = Path.home() / ".harness-wrapper" / "clis" / "native" / "bin" / "codex"
    return str(private) if private.is_file() else "codex"


def _codex_oauth_logged_in() -> bool:
    try:
        result = subprocess.run(
            (_codex_executable(), "login", "status"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        status = f"{result.stdout}\n{result.stderr}".lower()
        return "chatgpt" in status or "oauth" in status
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def openai_oauth_config(
    *,
    device_auth: bool = False,
    auto_login: bool = True,
    auto_relogin: bool = True,
    expiry_leeway: float = 30.0,
) -> OAuthConfig:
    """Build the Codex-owned OAuth adapter.

    Device authorization avoids a localhost callback and is therefore the
    appropriate flow for SSH and other headless sessions.
    """

    executable = _codex_executable()
    login_command = (executable, "login", "--device-auth") if device_auth else (executable, "login")
    return OAuthConfig(
        provider="openai",
        login_command=login_command,
        status_hook=_codex_oauth_logged_in,
        credential_loader=codex_oauth_credential,
        rate_limit_hook=poll_openai_rate_limits,
        available_models_command=(executable, "debug", "models"),
        auto_login=auto_login,
        auto_relogin=auto_relogin,
        expiry_leeway=expiry_leeway,
    )


class OpenAIOAuthModel(OAuthModel):
    api_type = APIType.OPENAI
    compatible_harnesses = frozenset({"codex-cli", "kimi-code"})

    def _parse_available_models(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, Mapping) or not isinstance(value.get("models"), list):
            return super()._parse_available_models(value)
        models: list[str] = []
        for item in value["models"]:
            if not isinstance(item, Mapping) or item.get("visibility") == "hide":
                continue
            slug = item.get("slug")
            if isinstance(slug, str):
                models.append(slug)
        return tuple(models)

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
            config=config or openai_oauth_config(),
            fallbacks=fallbacks,
            reasoning=reasoning,
        )
