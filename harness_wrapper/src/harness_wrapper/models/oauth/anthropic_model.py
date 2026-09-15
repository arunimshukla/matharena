from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ..apis import APIType
from .native_credentials import claude_oauth_credential
from .oauth import OAuthConfig, OAuthModel
from .rate_limits import poll_anthropic_rate_limits


def _claude_executable() -> str:
    executable = shutil.which("claude")
    if executable is not None:
        return executable
    private = Path.home() / ".harness-wrapper" / "clis" / "native" / "bin" / "claude"
    return str(private) if private.is_file() else "claude"


def _claude_logged_in() -> bool:
    try:
        result = subprocess.run(
            (_claude_executable(), "auth", "status"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        value = json.loads(result.stdout)
        return bool(value.get("loggedIn"))
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ):
        return False


def anthropic_oauth_config(
    *,
    auto_login: bool = True,
    auto_relogin: bool = True,
    expiry_leeway: float = 30.0,
) -> OAuthConfig:
    executable = _claude_executable()
    return OAuthConfig(
        provider="anthropic",
        login_command=(executable, "auth", "login"),
        status_hook=_claude_logged_in,
        credential_loader=claude_oauth_credential,
        rate_limit_hook=poll_anthropic_rate_limits,
        # Claude Code resolves these stable family aliases against the
        # subscription and organization allowlist when a subagent starts.
        available_models_hook=lambda credential: ("fable", "opus", "sonnet", "haiku"),
        auto_login=auto_login,
        auto_relogin=auto_relogin,
        expiry_leeway=expiry_leeway,
    )


class AnthropicOAuthModel(OAuthModel):
    """Claude subscription OAuth; only Anthropic-compatible harnesses may use it."""

    api_type = APIType.ANTHROPIC
    compatible_harnesses = frozenset({"claude-code"})

    def cli_environment(self, api_type: APIType | str) -> dict[str, str]:
        self.assert_compatible(api_type)
        credential = self.ensure_authenticated()
        if credential.access_token is None:
            return {}
        return {"CLAUDE_CODE_OAUTH_TOKEN": credential.access_token}

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
            config=config or anthropic_oauth_config(),
            fallbacks=fallbacks,
            reasoning=reasoning,
        )
