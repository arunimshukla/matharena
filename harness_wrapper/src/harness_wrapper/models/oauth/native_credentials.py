"""Import OAuth token bundles from the supported native agent CLIs."""

from __future__ import annotations

import base64
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from loguru import logger

from .oauth import AuthenticationError, OAuthCredential


def _read_private_json(path: Path, *, product: str) -> Mapping[str, object]:
    logger.bind(component="oauth_credentials", provider=product, credential_path=str(path)).debug(
        "Importing native CLI OAuth credential"
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise AuthenticationError(f"cannot import {product} OAuth credentials: {error}") from error
    try:
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise AuthenticationError(f"{product} OAuth store must be one regular file")
        if details.st_mode & 0o077:
            raise AuthenticationError(f"{product} OAuth store must have mode 0600")
        with os.fdopen(fd, encoding="utf-8", closefd=False) as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise AuthenticationError(f"invalid {product} OAuth store: {error}") from error
    finally:
        os.close(fd)
    if not isinstance(value, dict):
        raise AuthenticationError(f"{product} OAuth store must contain an object")
    return value


def _required_string(value: object, *, name: str, product: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthenticationError(f"{product} OAuth store has no {name}")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _seconds(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result / 1000 if result > 100_000_000_000 else result


def _jwt_expiry(token: str) -> float | None:
    try:
        encoded = token.split(".", 2)[1]
        encoded += "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded))
        expires = payload.get("exp")
        return float(expires) if isinstance(expires, (int, float)) else None
    except (IndexError, TypeError, ValueError):
        return None


def codex_oauth_credential(
    path: str | os.PathLike[str] | None = None,
) -> OAuthCredential:
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    auth_path = Path(path).expanduser() if path is not None else root / "auth.json"
    value = _read_private_json(auth_path, product="Codex")
    if value.get("auth_mode") not in {"chatgpt", "oauth"}:
        raise AuthenticationError("Codex is not logged in with a ChatGPT OAuth account")
    tokens = value.get("tokens")
    if not isinstance(tokens, Mapping):
        raise AuthenticationError("Codex OAuth store has no tokens")
    access_token = _required_string(
        tokens.get("access_token"), name="access token", product="Codex"
    )
    account_id = _required_string(tokens.get("account_id"), name="account ID", product="Codex")
    extra: dict[str, object] = {
        "account_id": account_id,
        "auth_mode": value.get("auth_mode", "chatgpt"),
        "cli_managed": True,
    }
    id_token = _optional_string(tokens.get("id_token"))
    if id_token is not None:
        extra["id_token"] = id_token
    last_refresh = _optional_string(value.get("last_refresh"))
    if last_refresh is not None:
        extra["last_refresh"] = last_refresh
    credential = OAuthCredential(
        access_token=access_token,
        refresh_token=_optional_string(tokens.get("refresh_token")),
        expires_at=_jwt_expiry(access_token),
        extra=extra,
    )
    logger.bind(component="oauth_credentials", provider="openai").debug(
        "Native OAuth credential imported: expires_at={expires_at}",
        expires_at=credential.expires_at,
    )
    return credential


def claude_oauth_credential(
    path: str | os.PathLike[str] | None = None,
) -> OAuthCredential:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
    auth_path = Path(path).expanduser() if path is not None else root / ".credentials.json"
    value = _read_private_json(auth_path, product="Claude Code")
    oauth = value.get("claudeAiOauth")
    if not isinstance(oauth, Mapping):
        raise AuthenticationError("Claude Code OAuth store has no claudeAiOauth token bundle")
    extra: dict[str, object] = {"cli_managed": True}
    mappings = {
        "scopes": "scopes",
        "subscriptionType": "subscription_type",
        "rateLimitTier": "rate_limit_tier",
    }
    for source, target in mappings.items():
        item = oauth.get(source)
        if isinstance(item, (str, list)):
            extra[target] = item
    refresh_expires = _seconds(oauth.get("refreshTokenExpiresAt"))
    if refresh_expires is not None:
        extra["refresh_token_expires_at"] = refresh_expires
    organization = _optional_string(value.get("organizationUuid"))
    if organization is not None:
        extra["organization_uuid"] = organization
    credential = OAuthCredential(
        access_token=_required_string(
            oauth.get("accessToken"), name="access token", product="Claude Code"
        ),
        refresh_token=_optional_string(oauth.get("refreshToken")),
        expires_at=_seconds(oauth.get("expiresAt")),
        extra=extra,
    )
    logger.bind(component="oauth_credentials", provider="anthropic").debug(
        "Native OAuth credential imported: expires_at={expires_at}",
        expires_at=credential.expires_at,
    )
    return credential


def kimi_oauth_credential(
    path: str | os.PathLike[str] | None = None,
) -> OAuthCredential:
    candidates: tuple[Path, ...]
    if path is not None:
        candidates = (Path(path).expanduser(),)
    else:
        root = Path(os.environ.get("KIMI_CODE_HOME", "~/.kimi-code")).expanduser()
        candidates = (
            root / "credentials" / "kimi-code.json",
            Path("~/.kimi/credentials/kimi-code.json").expanduser(),
        )
    existing = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    value = _read_private_json(existing, product="Kimi Code")
    credential = OAuthCredential(
        access_token=_required_string(
            value.get("access_token"), name="access token", product="Kimi Code"
        ),
        refresh_token=_optional_string(value.get("refresh_token")),
        expires_at=_seconds(value.get("expires_at")),
        extra={
            "cli_managed": True,
            "scope": value.get("scope", ""),
            "token_type": value.get("token_type", "Bearer"),
            "expires_in": value.get("expires_in", 0),
        },
    )
    logger.bind(component="oauth_credentials", provider="kimi").debug(
        "Native OAuth credential imported: expires_at={expires_at}",
        expires_at=credential.expires_at,
    )
    return credential


__all__ = ["claude_oauth_credential", "codex_oauth_credential", "kimi_oauth_credential"]
