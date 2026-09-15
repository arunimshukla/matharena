"""Provider-native OAuth subscription rate-limit polling."""

from __future__ import annotations

import json
import math
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .oauth import AuthenticationError, OAuthCredential, RateLimit, RateLimitSnapshot

_ANTHROPIC_USAGE_ENDPOINTS = (
    "https://api.anthropic.com/api/oauth/usage",
    "https://api.anthropic.com/oauth/usage",
    "https://claude.ai/api/oauth/usage",
)
_KIMI_USAGE_ENDPOINT = "https://api.kimi.com/coding/v1/usages"
_KIMI_TOKEN_ENDPOINT = "https://auth.kimi.com/api/oauth/token"
_KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
_DEFAULT_POLL_TIMEOUT = 15.0


def _number(value: object) -> float | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _timestamp(value: object) -> float | None:
    number = _number(value)
    if number is not None:
        return number / 1000.0 if number > 10_000_000_000 else number
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    # Kimi currently returns nanoseconds while datetime accepts microseconds.
    normalized = re.sub(r"(\.\d{6})\d+", r"\1", normalized)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_limited(status: object, utilization: float) -> bool:
    return utilization >= 1.0 or str(status or "").lower() in {
        "blocked",
        "exhausted",
        "limited",
        "rejected",
    }


def parse_anthropic_rate_limits(payload: Mapping[str, object]) -> RateLimitSnapshot:
    """Normalize Claude's OAuth usage windows."""
    aliases = {"weekly": "seven_day", "week": "seven_day"}
    window_names = (
        "five_hour",
        "seven_day",
        "weekly",
        "week",
        "seven_day_opus",
        "seven_day_sonnet",
    )
    limits: list[RateLimit] = []
    seen: set[str] = set()
    for source_name in window_names:
        raw = payload.get(source_name)
        if not isinstance(raw, Mapping):
            continue
        name = aliases.get(source_name, source_name)
        if name in seen:
            continue
        used_percent = _number(raw.get("utilization", raw.get("used_percentage")))
        if used_percent is None or used_percent < 0:
            continue
        utilization = min(1.0, used_percent / 100.0)
        limits.append(
            RateLimit(
                name=name,
                resets_at=_timestamp(raw.get("resets_at", raw.get("resetsAt"))),
                utilization=utilization,
                limited=_is_limited(raw.get("status"), utilization),
            )
        )
        seen.add(name)
    return RateLimitSnapshot(tuple(limits))


def _window_name(seconds: int, fallback: str) -> str:
    if seconds == 5 * 60 * 60:
        return "five_hour"
    if seconds == 7 * 24 * 60 * 60:
        return "seven_day"
    return fallback


def parse_openai_rate_limits(payload: Mapping[str, object]) -> RateLimitSnapshot:
    """Normalize the documented Codex ``account/rateLimits/read`` result."""
    by_id = payload.get("rateLimitsByLimitId", payload.get("rate_limits_by_limit_id"))
    if isinstance(by_id, Mapping) and by_id:
        limit_sets = [item for item in by_id.values() if isinstance(item, Mapping)]
    else:
        primary = payload.get("rateLimits", payload.get("rate_limits"))
        limit_sets = [primary] if isinstance(primary, Mapping) else []

    limits: list[RateLimit] = []
    multiple_limit_sets = len(limit_sets) > 1
    for limit_set in limit_sets:
        limit_id = str(limit_set.get("limitId", limit_set.get("limit_id", "codex")))
        reached = str(
            limit_set.get("rateLimitReachedType", limit_set.get("rate_limit_reached_type", ""))
            or ""
        ).lower()
        for role in ("primary", "secondary"):
            window = limit_set.get(role)
            if not isinstance(window, Mapping):
                continue
            used_percent = _number(window.get("usedPercent", window.get("used_percent")))
            minutes = _number(window.get("windowDurationMins", window.get("window_duration_mins")))
            if used_percent is None or minutes is None or minutes <= 0:
                continue
            utilization = max(0.0, min(1.0, used_percent / 100.0))
            reached_this_window = bool(reached) and (
                role in reached or reached not in {"primary", "secondary"}
            )
            window_name = _window_name(int(minutes * 60), f"{limit_id}_{role}")
            if multiple_limit_sets:
                window_name = f"{limit_id}_{window_name}"
            limits.append(
                RateLimit(
                    name=window_name,
                    resets_at=_timestamp(window.get("resetsAt", window.get("resets_at"))),
                    utilization=utilization,
                    limited=reached_this_window or utilization >= 1.0,
                )
            )
    return RateLimitSnapshot(tuple(limits))


def _kimi_limit(detail: Mapping[str, object], *, name: str) -> RateLimit | None:
    limit = _number(detail.get("limit"))
    used = _number(detail.get("used"))
    remaining = _number(detail.get("remaining"))
    if limit is None or limit <= 0:
        return None
    if used is None and remaining is not None:
        used = limit - remaining
    if used is None:
        return None
    utilization = max(0.0, min(1.0, used / limit))
    return RateLimit(
        name=name,
        resets_at=_timestamp(
            detail.get("resetTime", detail.get("reset_time", detail.get("resetAt")))
        ),
        utilization=utilization,
        limited=(remaining is not None and remaining <= 0) or utilization >= 1.0,
    )


def parse_kimi_rate_limits(payload: Mapping[str, object]) -> RateLimitSnapshot:
    """Normalize Kimi Code's OAuth usage response."""
    limits: list[RateLimit] = []
    weekly = payload.get("usage")
    if isinstance(weekly, Mapping):
        parsed = _kimi_limit(weekly, name="seven_day")
        if parsed is not None:
            limits.append(parsed)

    rolling = payload.get("limits")
    if isinstance(rolling, list):
        for index, item in enumerate(rolling):
            if not isinstance(item, Mapping):
                continue
            detail = item.get("detail")
            detail = detail if isinstance(detail, Mapping) else item
            window = item.get("window")
            if not isinstance(window, Mapping):
                continue
            duration = _number(window.get("duration"))
            if duration is None or duration <= 0:
                continue
            unit = str(window.get("timeUnit", window.get("time_unit", ""))).upper()
            multiplier = 1
            if "MINUTE" in unit:
                multiplier = 60
            elif "HOUR" in unit:
                multiplier = 60 * 60
            elif "DAY" in unit:
                multiplier = 24 * 60 * 60
            seconds = int(duration * multiplier)
            parsed = _kimi_limit(
                detail,
                name=_window_name(seconds, f"limit_{index + 1}"),
            )
            if parsed is not None:
                limits.append(parsed)

    total = payload.get("totalQuota", payload.get("total_quota"))
    if isinstance(total, Mapping):
        parsed = _kimi_limit(total, name="monthly")
        if parsed is not None:
            limits.append(parsed)
    return RateLimitSnapshot(tuple(limits))


def _require_token(credential: OAuthCredential | None, provider: str) -> str:
    if credential is None or not credential.access_token:
        raise RuntimeError(f"cannot poll {provider} rate limits without an OAuth access token")
    return credential.access_token


def _get_json(
    url: str,
    token: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = _DEFAULT_POLL_TIMEOUT,
) -> Mapping[str, object]:
    logger.bind(component="oauth_rate_limits", endpoint=url).debug(
        "Requesting provider usage data: timeout={timeout}", timeout=timeout
    )
    request_headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if headers is not None:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        logger.bind(component="oauth_rate_limits", endpoint=url).debug(
            "Provider usage request completed: status={status}",
            status=getattr(response, "status", None),
        )
    if not isinstance(payload, Mapping):
        raise RuntimeError("usage endpoint returned a non-object JSON response")
    return payload


def poll_anthropic_rate_limits(
    credential: OAuthCredential | None,
    *,
    timeout: float = _DEFAULT_POLL_TIMEOUT,
) -> RateLimitSnapshot:
    bound_logger = logger.bind(component="oauth_rate_limits", provider="anthropic")
    bound_logger.debug("Polling Anthropic OAuth usage windows")
    token = _require_token(credential, "anthropic")
    failures: list[str] = []
    for endpoint in _ANTHROPIC_USAGE_ENDPOINTS:
        try:
            payload = _get_json(
                endpoint,
                token,
                headers={
                    "anthropic-beta": "oauth-2025-04-20",
                    "anthropic-version": "2023-06-01",
                },
                timeout=timeout,
            )
            snapshot = parse_anthropic_rate_limits(payload)
            if snapshot.limits:
                bound_logger.info(
                    "Anthropic OAuth usage windows received: count={count}",
                    count=len(snapshot.limits),
                )
                return snapshot
            failures.append(f"{endpoint}: response contained no usage windows")
        except urllib.error.HTTPError as exc:
            failures.append(f"{endpoint}: HTTP {exc.code}")
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(f"{endpoint}: {exc}")
    if failures and any("HTTP 401" in failure for failure in failures):
        raise AuthenticationError("anthropic OAuth token was rejected while polling rate limits")
    raise RuntimeError("cannot poll anthropic rate limits: " + "; ".join(failures))


def _kimi_credentials_path() -> Path:
    root = Path(os.environ.get("KIMI_CODE_HOME", "~/.kimi-code")).expanduser()
    candidates = (
        root / "credentials" / "kimi-code.json",
        Path("~/.kimi/credentials/kimi-code.json").expanduser(),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


@contextmanager
def _kimi_refresh_lock(credentials_path: Path) -> Iterator[None]:
    lock_path = credentials_path.parent / ".oauth_refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            import fcntl
        except ImportError:
            yield
            return
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _refresh_kimi_credential(stale: OAuthCredential) -> OAuthCredential:
    credentials_path = _kimi_credentials_path()
    bound_logger = logger.bind(component="oauth_rate_limits", provider="kimi")
    bound_logger.info("Refreshing native Kimi OAuth credential")
    try:
        with _kimi_refresh_lock(credentials_path):
            raw = json.loads(credentials_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise RuntimeError("Kimi OAuth store does not contain an object")
            current_token = raw.get("access_token")
            if isinstance(current_token, str) and current_token != stale.access_token:
                bound_logger.info("Using Kimi credential refreshed by another process")
                return OAuthCredential.from_dict(raw)
            refresh_token = raw.get("refresh_token", stale.refresh_token)
            if not isinstance(refresh_token, str) or not refresh_token:
                raise RuntimeError("Kimi OAuth credentials do not contain a refresh token")
            body = urllib.parse.urlencode(
                {
                    "client_id": _KIMI_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                }
            ).encode("ascii")
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Msh-Platform": "kimi_cli",
                "X-Msh-Version": "harness-wrapper/0.1.0",
            }
            device_id = credentials_path.parents[1] / "device_id"
            if device_id.is_file():
                headers["X-Msh-Device-Id"] = device_id.read_text(encoding="utf-8").strip()
            request = urllib.request.Request(
                _KIMI_TOKEN_ENDPOINT,
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=_DEFAULT_POLL_TIMEOUT) as response:
                refreshed = json.loads(response.read().decode("utf-8"))
            if not isinstance(refreshed, Mapping) or not isinstance(
                refreshed.get("access_token"), str
            ):
                raise RuntimeError("Kimi OAuth refresh returned no access token")
            raw.update(refreshed)
            expires_in = _number(refreshed.get("expires_in"))
            if expires_in is not None:
                raw["expires_at"] = time.time() + expires_in
            _write_private_json(credentials_path, raw)
            bound_logger.info("Native Kimi OAuth credential refresh completed")
            return OAuthCredential.from_dict(raw)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 401, 403}:
            raise AuthenticationError(
                f"Kimi OAuth refresh was rejected with HTTP {exc.code}"
            ) from exc
        raise RuntimeError(f"Kimi OAuth refresh failed: HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Kimi OAuth refresh failed: {exc}") from exc


def poll_kimi_rate_limits(
    credential: OAuthCredential | None,
    *,
    timeout: float = _DEFAULT_POLL_TIMEOUT,
) -> RateLimitSnapshot:
    bound_logger = logger.bind(component="oauth_rate_limits", provider="kimi")
    bound_logger.debug("Polling Kimi OAuth usage windows")
    token = _require_token(credential, "kimi")
    if credential is not None and credential.expired and credential.refresh_token:
        bound_logger.info("Kimi access token expired before usage polling")
        credential = _refresh_kimi_credential(credential)
        token = _require_token(credential, "kimi")
    try:
        snapshot = parse_kimi_rate_limits(_get_json(_KIMI_USAGE_ENDPOINT, token, timeout=timeout))
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and credential is not None and credential.refresh_token:
            refreshed = _refresh_kimi_credential(credential)
            try:
                snapshot = parse_kimi_rate_limits(
                    _get_json(
                        _KIMI_USAGE_ENDPOINT,
                        _require_token(refreshed, "kimi"),
                        timeout=timeout,
                    )
                )
            except urllib.error.HTTPError as retry_exc:
                if retry_exc.code in {401, 403}:
                    raise AuthenticationError(
                        "Kimi OAuth token was rejected after refresh"
                    ) from retry_exc
                raise RuntimeError(
                    f"cannot poll kimi rate limits after OAuth refresh: HTTP {retry_exc.code}"
                ) from retry_exc
        else:
            if exc.code in {401, 403}:
                raise AuthenticationError("Kimi OAuth token was rejected") from exc
            raise RuntimeError(f"cannot poll kimi rate limits: HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot poll kimi rate limits: {exc}") from exc
    if not snapshot.limits:
        raise RuntimeError("cannot poll kimi rate limits: response contained no usage windows")
    bound_logger.info(
        "Kimi OAuth usage windows received: count={count}, limited={limited}",
        count=len(snapshot.limits),
        limited=snapshot.limited,
    )
    return snapshot


def _codex_executable() -> str:
    executable = shutil.which("codex")
    if executable is not None:
        return executable
    private = Path.home() / ".harness-wrapper" / "clis" / "native" / "bin" / "codex"
    if private.is_file() and os.access(private, os.X_OK):
        return str(private)
    raise RuntimeError("cannot poll openai rate limits: codex executable was not found")


def poll_openai_rate_limits(
    _credential: OAuthCredential | None,
    *,
    timeout: float = _DEFAULT_POLL_TIMEOUT,
) -> RateLimitSnapshot:
    """Poll ChatGPT limits through Codex's documented app-server method."""
    bound_logger = logger.bind(component="oauth_rate_limits", provider="openai")
    bound_logger.debug(
        "Polling OpenAI OAuth usage windows through the Codex app-server: timeout={timeout}",
        timeout=timeout,
    )
    messages = (
        {
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "harness_wrapper",
                    "title": "Harness Wrapper",
                    "version": "0.1.0",
                }
            },
        },
        {"method": "initialized", "params": {}},
        {"method": "account/rateLimits/read", "id": 1},
    )
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            (_codex_executable(), "app-server"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        for message in messages:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [process.stdout], [], [], min(1.0, deadline - time.monotonic())
            )
            if not ready:
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if not line:
                break
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, Mapping) or response.get("id") != 1:
                continue
            result = response.get("result")
            if isinstance(result, Mapping):
                snapshot = parse_openai_rate_limits(result)
                bound_logger.info(
                    "OpenAI OAuth usage windows received: count={count}, limited={limited}",
                    count=len(snapshot.limits),
                    limited=snapshot.limited,
                )
                return snapshot
            error = str(response.get("error"))
            if re.search(r"(?:unauthori[sz]ed|token|oauth|\b401\b)", error, re.IGNORECASE):
                raise AuthenticationError(
                    f"OpenAI OAuth session was rejected while polling rate limits: {error}"
                )
            raise RuntimeError(f"cannot poll openai rate limits: {error}")
        raise RuntimeError("cannot poll openai rate limits: Codex app-server timed out")
    except OSError as exc:
        raise RuntimeError(f"cannot poll openai rate limits: {exc}") from exc
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


__all__ = [
    "parse_anthropic_rate_limits",
    "parse_kimi_rate_limits",
    "parse_openai_rate_limits",
    "poll_anthropic_rate_limits",
    "poll_kimi_rate_limits",
    "poll_openai_rate_limits",
]
