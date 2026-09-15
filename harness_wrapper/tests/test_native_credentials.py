from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from harness_wrapper.models.oauth.native_credentials import (
    claude_oauth_credential,
    codex_oauth_credential,
    kimi_oauth_credential,
)


def _private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def _jwt(expires_at: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires_at}).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def test_import_codex_oauth_bundle(tmp_path: Path) -> None:
    path = tmp_path / "codex.json"
    expires = time.time() + 3600
    _private_json(
        path,
        {
            "auth_mode": "chatgpt",
            "last_refresh": "now",
            "tokens": {
                "access_token": _jwt(expires),
                "refresh_token": "openai-refresh",
                "id_token": "openai-id",
                "account_id": "account-1",
            },
        },
    )
    credential = codex_oauth_credential(path)
    assert credential.refresh_token == "openai-refresh"
    assert credential.expires_at == expires
    assert credential.extra["account_id"] == "account-1"
    assert credential.extra["id_token"] == "openai-id"


def test_import_claude_oauth_bundle_converts_milliseconds(tmp_path: Path) -> None:
    path = tmp_path / "claude.json"
    expires_ms = 4_000_000_000_000
    _private_json(
        path,
        {
            "organizationUuid": "org-1",
            "claudeAiOauth": {
                "accessToken": "anthropic-access",
                "refreshToken": "anthropic-refresh",
                "expiresAt": expires_ms,
                "refreshTokenExpiresAt": expires_ms + 1000,
                "scopes": ["user:inference"],
                "subscriptionType": "pro",
                "rateLimitTier": "default",
            },
        },
    )
    credential = claude_oauth_credential(path)
    assert credential.access_token == "anthropic-access"
    assert credential.refresh_token == "anthropic-refresh"
    assert credential.expires_at == expires_ms / 1000
    assert credential.extra["organization_uuid"] == "org-1"


def test_import_kimi_oauth_bundle(tmp_path: Path) -> None:
    path = tmp_path / "kimi-code.json"
    _private_json(
        path,
        {
            "access_token": "kimi-access",
            "refresh_token": "kimi-refresh",
            "expires_at": 4_000_000_000,
            "scope": "openid",
            "token_type": "Bearer",
            "expires_in": 3600,
        },
    )
    credential = kimi_oauth_credential(path)
    assert credential.access_token == "kimi-access"
    assert credential.refresh_token == "kimi-refresh"
    assert credential.expires_at == 4_000_000_000
    assert credential.extra["scope"] == "openid"
