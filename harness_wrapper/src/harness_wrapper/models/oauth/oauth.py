"""OAuth configuration, persistence, and rate-limit coordination."""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from ..apis import APIType
from ..base import AbstractModel, parse_reasoning


class AuthenticationError(RuntimeError):
    pass


class RateLimitTimeout(TimeoutError):
    pass


class RateLimitPollingUnsupported(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OAuthCredential:
    """Opaque persisted OAuth state.

    Providers whose CLI owns the token can persist only metadata.  Providers
    integrated by downstream code may store access and refresh tokens here.
    """

    access_token: str | None = field(default=None, repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    extra: Mapping[str, object] = field(default_factory=dict, repr=False)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = dict(self.extra)
        if self.access_token is not None:
            result["access_token"] = self.access_token
        if self.refresh_token is not None:
            result["refresh_token"] = self.refresh_token
        if self.expires_at is not None:
            result["expires_at"] = self.expires_at
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OAuthCredential:
        known = {"access_token", "refresh_token", "expires_at"}
        expires = value.get("expires_at")
        return cls(
            access_token=_optional_str(value.get("access_token")),
            refresh_token=_optional_str(value.get("refresh_token")),
            expires_at=_required_float(expires) if expires is not None else None,
            extra={key: item for key, item in value.items() if key not in known},
        )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _required_float(value: object) -> float:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    return float(value)


class CredentialStore:
    """Small atomic JSON store, private to harness-wrapper."""

    def __init__(self, path: Path | str | None = None) -> None:
        # expanduser uses account information, not credential-bearing env vars.
        self.path = (
            Path(path) if path is not None else Path("~/.harness-wrapper/auth.json").expanduser()
        )

    def _read_all(self) -> dict[str, object]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise AuthenticationError(f"cannot read OAuth store {self.path}: {exc}") from exc
        try:
            details = os.fstat(fd)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise AuthenticationError(f"OAuth store {self.path} must be one regular file")
            if details.st_mode & 0o077:
                raise AuthenticationError(f"OAuth store {self.path} must have mode 0600")
            with os.fdopen(fd, encoding="utf-8", closefd=False) as stream:
                raw = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthenticationError(f"cannot read OAuth store {self.path}: {exc}") from exc
        finally:
            os.close(fd)
        if not isinstance(raw, dict):
            raise AuthenticationError(f"OAuth store {self.path} must contain a JSON object")
        return raw

    def load(self, provider: str) -> OAuthCredential | None:
        value = self._read_all().get(provider)
        credential = OAuthCredential.from_dict(value) if isinstance(value, dict) else None
        logger.bind(provider=provider, auth_store=str(self.path)).debug(
            "OAuth credential store read: present={present}",
            present=credential is not None,
        )
        return credential

    def save(self, provider: str, credential: OAuthCredential) -> None:
        data = self._read_all()
        data[provider] = credential.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        fd, temporary = tempfile.mkstemp(prefix=".auth-", suffix=".json", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            logger.bind(provider=provider, auth_store=str(self.path)).debug(
                "OAuth credential store updated"
            )
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    def delete(self, provider: str) -> None:
        data = self._read_all()
        if provider in data:
            del data[provider]
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.path.parent, 0o700)
            # Preserve atomicity and permissions through the normal writer.
            fd, temporary = tempfile.mkstemp(prefix=".auth-", suffix=".json", dir=self.path.parent)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, self.path)
                os.chmod(self.path, 0o600)
                logger.bind(provider=provider, auth_store=str(self.path)).info(
                    "OAuth credential removed from central store"
                )
            except BaseException:
                with suppress(FileNotFoundError):
                    os.unlink(temporary)
                raise


@dataclass(frozen=True, slots=True)
class RateLimit:
    name: str
    resets_at: float | None = None
    utilization: float | None = None
    limited: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RateLimit:
        resets = value.get("resets_at", value.get("resetsAt"))
        utilization = value.get("utilization")
        status = str(value.get("status", "")).lower()
        raw_limited = value.get("limited", False)
        explicitly_limited = raw_limited is True or (
            isinstance(raw_limited, str) and raw_limited.lower() in {"true", "1", "yes"}
        )
        limited = explicitly_limited or status in {
            "limited",
            "rejected",
            "exhausted",
            "blocked",
        }
        return cls(
            name=str(
                value.get(
                    "name", value.get("rate_limit_type", value.get("rateLimitType", "unknown"))
                )
            ),
            resets_at=_required_float(resets) if resets is not None else None,
            utilization=_required_float(utilization) if utilization is not None else None,
            limited=limited,
        )


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    limits: tuple[RateLimit, ...] = ()
    checked_at: float = field(default_factory=time.time)

    @classmethod
    def parse(cls, value: object) -> RateLimitSnapshot:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            if "limits" in value:
                raw = value["limits"]
            elif "rate_limits" in value:
                raw = value["rate_limits"]
            elif "rate_limit_info" in value and isinstance(value["rate_limit_info"], Mapping):
                raw = value["rate_limit_info"]
            elif any(
                key in value
                for key in (
                    "name",
                    "status",
                    "limited",
                    "resets_at",
                    "resetsAt",
                    "rate_limit_type",
                    "rateLimitType",
                    "utilization",
                )
            ):
                return cls((RateLimit.from_mapping(value),))
            else:
                raw = value
            if isinstance(raw, Mapping):
                if any(
                    key in raw
                    for key in (
                        "status",
                        "limited",
                        "resets_at",
                        "resetsAt",
                        "rate_limit_type",
                        "rateLimitType",
                        "utilization",
                    )
                ):
                    return cls((RateLimit.from_mapping(raw),))
                limits = []
                for name, item in raw.items():
                    if isinstance(item, Mapping):
                        limits.append(RateLimit.from_mapping({"name": name, **item}))
                return cls(tuple(limits))
            value = raw
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return cls(
                tuple(
                    item if isinstance(item, RateLimit) else RateLimit.from_mapping(item)
                    for item in value
                )
            )
        raise TypeError("rate-limit poller must return a snapshot, mapping, iterable, or None")

    @property
    def limited(self) -> bool:
        return any(item.limited for item in self.limits)

    @property
    def next_reset_at(self) -> float | None:
        values = [
            item.resets_at for item in self.limits if item.limited and item.resets_at is not None
        ]
        return min(values) if values else None


LoginHook = Callable[[], OAuthCredential | Mapping[str, object] | None]
CredentialLoader = Callable[[], OAuthCredential | Mapping[str, object] | None]
CredentialRefreshHook = Callable[[OAuthCredential], OAuthCredential | Mapping[str, object] | None]
PollHook = Callable[[OAuthCredential | None], object]
ModelDiscoveryHook = Callable[[OAuthCredential | None], object]
StatusHook = Callable[[], bool]


@dataclass(slots=True)
class OAuthConfig:
    provider: str
    store: CredentialStore = field(default_factory=CredentialStore)
    login_command: Sequence[str] | None = None
    login_hook: LoginHook | None = None
    credential_loader: CredentialLoader | None = None
    credential_refresh_hook: CredentialRefreshHook | None = None
    status_hook: StatusHook | None = None
    rate_limit_hook: PollHook | None = None
    rate_limit_command: Sequence[str] | None = None
    rate_limit_command_timeout: float = 15.0
    available_models_hook: ModelDiscoveryHook | None = None
    available_models_command: Sequence[str] | None = None
    available_models_command_timeout: float = 15.0
    auto_login: bool = True
    auto_relogin: bool = True
    expiry_leeway: float = 30.0


class OAuthModel(AbstractModel):
    """CLI-owned OAuth plan with pluggable login and usage polling."""

    auth_mode = "oauth"
    api_url = None
    api_key = None
    api_type: APIType
    compatible_harnesses: frozenset[str] = frozenset()
    _RATE_LIMIT_FAILURE = re.compile(
        r"(?:rate[ _-]?limit|too many requests|usage limit|quota (?:is )?(?:exhausted|reached)|"
        r"(?:5|five)[ -]?hour|weekly limit|http\s*429|status\s*429|\b429\b)",
        re.IGNORECASE,
    )
    _AUTHENTICATION_FAILURE = re.compile(
        r"(?:unauthori[sz]ed|authentication (?:failed|required)|not logged in|login required|"
        r"token (?:is )?(?:expired|invalid|revoked)|invalid[_ -]?grant|oauth.*(?:expired|invalid)|"
        r"http\s*401|status\s*401|\b401\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model: str,
        *,
        config: OAuthConfig,
        fallbacks: Iterable[object] = (),
        reasoning: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.reasoning = parse_reasoning(reasoning)
        self.config = config
        self.fallbacks = tuple(fallbacks)
        self._available_models: tuple[str, ...] | None = None
        self._authentication_error: AuthenticationError | None = None
        self.credential = config.store.load(config.provider)
        bound_logger = logger.bind(provider=config.provider, model=model)
        bound_logger.debug(
            "OAuth model initialized: credential_present={present}, auto_login={auto_login}, "
            "auto_relogin={auto_relogin}",
            present=self.credential is not None,
            auto_login=config.auto_login,
            auto_relogin=config.auto_relogin,
        )
        if config.auto_login and (self.credential is None or self.credential.expired):
            try:
                if self._cli_logged_in():
                    bound_logger.info("Importing existing native CLI OAuth session")
                    self.credential = self._save_discovered_credential()
                else:
                    self.login()
            except AuthenticationError as exc:
                if not self.fallbacks:
                    raise
                bound_logger.warning(
                    "OAuth initialization failed; a configured fallback may be used: "
                    "error_type={error_type}",
                    error_type=type(exc).__name__,
                )
                self._authentication_error = exc

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def oauth(self) -> str:
        return self.config.provider

    def supported_endpoints(self) -> frozenset[str]:
        return frozenset((self.api_type.value,))

    def supported_api_types(self) -> frozenset[APIType]:
        return frozenset((self.api_type,))

    def assert_compatible(self, accepted: APIType | str | Iterable[APIType | str]) -> APIType:
        if isinstance(accepted, (str, APIType)):
            candidates: tuple[APIType, ...] = (APIType.parse(accepted),)
        else:
            candidates = tuple(APIType.parse(item) for item in accepted)
        if self.api_type in candidates:
            return self.api_type
        wanted = ", ".join(item.value for item in candidates) or "none"
        raise ValueError(
            f"OAuth provider {self.provider!r} is {self.api_type.value}-only; "
            f"harness accepts {wanted}"
        )

    def assert_harness_compatible(self, harness: str) -> None:
        """Ensure a CLI-owned OAuth session is used only by approved harnesses."""
        normalized = harness.strip().lower().replace("_", "-")
        if self.compatible_harnesses and normalized not in self.compatible_harnesses:
            supported = ", ".join(sorted(self.compatible_harnesses))
            raise ValueError(
                f"OAuth provider {self.provider!r} cannot authenticate {harness!r}; "
                f"supported harnesses: {supported}"
            )

    def cli_environment(self, api_type: APIType | str) -> dict[str, str]:
        """OAuth credentials remain CLI-owned and are never copied to env."""
        self.assert_compatible(api_type)
        return {}

    def cli_args(self, api_type: APIType | str) -> tuple[str, ...]:
        self.assert_compatible(api_type)
        return ()

    def available_models(self, *, refresh: bool = False) -> tuple[str, ...]:
        """Return models exposed by this OAuth account's native CLI.

        Discovery deliberately stays with the provider CLI so account tier,
        staged rollouts, and organization policy are reflected without a
        hard-coded model catalog in harness-wrapper.
        """

        if self._available_models is not None and not refresh:
            return self._available_models
        if self.config.available_models_hook is not None:
            try:
                raw = self.config.available_models_hook(self.credential)
            except Exception as exc:
                raise RuntimeError(f"cannot discover {self.provider} OAuth models: {exc}") from exc
        elif self.config.available_models_command is not None:
            timeout = self.config.available_models_command_timeout
            if timeout <= 0:
                raise ValueError("available_models_command_timeout must be positive")
            try:
                command = list(self.config.available_models_command)
                executable = command[0]
                if not Path(executable).is_absolute() and shutil.which(executable) is None:
                    private = (
                        Path.home() / ".harness-wrapper" / "clis" / "native" / "bin" / executable
                    )
                    if private.is_file() and os.access(private, os.X_OK):
                        command[0] = str(private)
                result = subprocess.run(
                    tuple(command),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                raw = json.loads(result.stdout)
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"cannot discover {self.provider} OAuth models: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"cannot parse {self.provider} OAuth model catalog: {exc}"
                ) from exc
        else:
            raw = ()

        discovered = self._parse_available_models(raw)
        # The selected main model remains a valid candidate when an older CLI
        # has no discovery command or omits aliases from its catalog.
        ordered = (self.model, *discovered)
        self._available_models = tuple(dict.fromkeys(item for item in ordered if item))
        return self._available_models

    def _parse_available_models(self, value: object) -> tuple[str, ...]:
        if isinstance(value, Mapping):
            value = value.get("models", value)
            if isinstance(value, Mapping):
                return tuple(str(item) for item in value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            result: list[str] = []
            for item in value:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, Mapping):
                    model = item.get("model", item.get("id", item.get("slug")))
                    if isinstance(model, str):
                        result.append(model)
            return tuple(result)
        raise TypeError("OAuth model discovery must return a mapping or iterable")

    def login(self) -> OAuthCredential:
        bound_logger = logger.bind(provider=self.provider, model=self.model)
        bound_logger.info("Starting interactive OAuth login")
        if self.config.login_hook is not None:
            try:
                result = self.config.login_hook()
                if isinstance(result, OAuthCredential):
                    credential = result
                elif isinstance(result, Mapping):
                    credential = OAuthCredential.from_dict(result)
                elif result is None:
                    credential = self._load_native_credential() or self._managed_marker(
                        "logged_in_at"
                    )
                else:
                    raise TypeError("login hook must return OAuthCredential, a mapping, or None")
            except AuthenticationError:
                raise
            except Exception as exc:
                raise AuthenticationError(f"{self.provider} OAuth login failed: {exc}") from exc
        elif self.config.login_command:
            try:
                subprocess.run(tuple(self.config.login_command), check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise AuthenticationError(f"{self.provider} OAuth login failed: {exc}") from exc
            credential = self._load_native_credential() or self._managed_marker("logged_in_at")
        else:
            raise AuthenticationError(f"no OAuth login adapter configured for {self.provider}")
        self.config.store.save(self.provider, credential)
        self.credential = credential
        self._authentication_error = None
        bound_logger.info(
            "OAuth login completed: expires_at={expires_at}, refresh_available={refresh_available}",
            expires_at=credential.expires_at,
            refresh_available=credential.refresh_token is not None,
        )
        return credential

    def _cli_logged_in(self) -> bool:
        if self.config.status_hook is None:
            return False
        try:
            logged_in = bool(self.config.status_hook())
            logger.bind(provider=self.provider, model=self.model).debug(
                "Native CLI OAuth status checked: logged_in={logged_in}",
                logged_in=logged_in,
            )
            return logged_in
        except Exception as exc:
            raise AuthenticationError(f"cannot check {self.provider} OAuth status: {exc}") from exc

    def _save_discovered_credential(self) -> OAuthCredential:
        credential = self._load_native_credential() or self._managed_marker("discovered_at")
        self.config.store.save(self.provider, credential)
        self._authentication_error = None
        return credential

    def _managed_marker(self, timestamp_name: str) -> OAuthCredential:
        return OAuthCredential(
            extra={
                "cli_managed": True,
                timestamp_name: datetime.now(timezone.utc).isoformat(),
            }
        )

    def _load_native_credential(self) -> OAuthCredential | None:
        loader = self.config.credential_loader
        if loader is None:
            return None
        try:
            value = loader()
            if isinstance(value, OAuthCredential):
                return value
            if isinstance(value, Mapping):
                return OAuthCredential.from_dict(value)
            if value is None:
                return None
            raise TypeError("credential loader must return OAuthCredential, a mapping, or None")
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(
                f"cannot import {self.provider} OAuth credentials: {exc}"
            ) from exc

    @staticmethod
    def _coerce_credential(value: object, *, source: str) -> OAuthCredential | None:
        if isinstance(value, OAuthCredential):
            return value
        if isinstance(value, Mapping):
            return OAuthCredential.from_dict(value)
        if value is None:
            return None
        raise TypeError(f"{source} must return OAuthCredential, a mapping, or None")

    def _save_credential(self, credential: OAuthCredential) -> OAuthCredential:
        self.config.store.save(self.provider, credential)
        self.credential = credential
        self._authentication_error = None
        logger.bind(provider=self.provider, model=self.model).debug(
            "OAuth credential synchronized: expires_at={expires_at}",
            expires_at=credential.expires_at,
        )
        return credential

    def _refresh_credential(self, credential: OAuthCredential) -> OAuthCredential | None:
        hook = self.config.credential_refresh_hook
        if hook is None:
            return None
        bound_logger = logger.bind(provider=self.provider, model=self.model)
        bound_logger.info("Refreshing OAuth credential")
        try:
            refreshed = self._coerce_credential(hook(credential), source="credential refresh hook")
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(
                f"cannot refresh {self.provider} OAuth credentials: {exc}"
            ) from exc
        if refreshed is None:
            bound_logger.warning("OAuth refresh hook returned no credential")
            return None
        bound_logger.info("OAuth credential refresh completed")
        return self._save_credential(refreshed)

    def _sync_native_credential(self, *, now: float | None = None) -> OAuthCredential | None:
        if self.config.credential_loader is None:
            return None
        imported = self._load_native_credential()
        if imported is None:
            return None
        existing = self.credential
        if (
            existing is not None
            and existing.access_token != imported.access_token
            and existing.expires_at is not None
            and imported.expires_at is not None
            and imported.expires_at <= existing.expires_at
        ):
            # A concurrently refreshed central credential must not be replaced
            # by an older native bundle observed just before token rotation.
            return existing
        current = time.time() if now is None else now
        if imported.expires_at is not None and imported.expires_at <= current:
            return imported
        return self._save_credential(imported)

    def ensure_authenticated(self, *, now: float | None = None) -> OAuthCredential:
        credential = self.credential
        current = time.time() if now is None else now
        if self.config.expiry_leeway < 0:
            raise ValueError("expiry_leeway cannot be negative")
        valid_after = current + self.config.expiry_leeway
        expired = credential is None or (
            credential.expires_at is not None and credential.expires_at <= valid_after
        )
        if expired:
            bound_logger = logger.bind(provider=self.provider, model=self.model)
            bound_logger.warning(
                "OAuth credential is missing, expired, or within the expiry safety window"
            )
            if credential is not None and credential.refresh_token is not None:
                try:
                    refreshed = self._refresh_credential(credential)
                except AuthenticationError:
                    refreshed = None
                if refreshed is not None and (
                    refreshed.expires_at is None or refreshed.expires_at > valid_after
                ):
                    bound_logger.info("OAuth credential restored by refresh")
                    return refreshed
            if self._cli_logged_in():
                imported = self._sync_native_credential(now=current)
                if imported is not None and (
                    imported.expires_at is None or imported.expires_at > valid_after
                ):
                    bound_logger.info("OAuth credential restored from native CLI state")
                    return imported
            if not self.config.auto_relogin:
                bound_logger.error("OAuth re-login required but automatic re-login is disabled")
                raise AuthenticationError(f"{self.provider} OAuth credential is missing or expired")
            bound_logger.info("OAuth credential requires interactive re-login")
            return self.login()
        assert credential is not None
        if (
            credential.extra.get("cli_managed") is True
            and credential.access_token is None
            and self.config.credential_loader is not None
            and self._cli_logged_in()
        ):
            imported = self._load_native_credential()
            if imported is not None:
                self.config.store.save(self.provider, imported)
                self.credential = imported
                credential = imported
        if (
            credential.extra.get("cli_managed") is True
            and self.config.status_hook is not None
            and not self._cli_logged_in()
        ):
            if not self.config.auto_relogin:
                raise AuthenticationError(f"{self.provider} CLI OAuth session is no longer valid")
            return self.login()
        return credential

    def force_relogin(self) -> OAuthCredential:
        """Run the interactive provider login even when CLI status looks valid."""

        if not self.config.auto_relogin:
            raise AuthenticationError(f"automatic {self.provider} OAuth re-login is disabled")
        logger.bind(provider=self.provider, model=self.model).warning(
            "Forcing OAuth re-login after provider rejection"
        )
        return self.login()

    def classify_cli_failure(self, message: str) -> str | None:
        """Classify provider-neutral CLI failures used by Agent recovery."""

        if self._RATE_LIMIT_FAILURE.search(message):
            return "rate_limit"
        if self._AUTHENTICATION_FAILURE.search(message):
            return "authentication"
        return None

    def compatible_fallbacks(
        self, accepted: APIType | str | Iterable[APIType | str]
    ) -> tuple[object, ...]:
        """Return every configured fallback compatible with a harness."""

        compatible: list[object] = []
        for fallback in self.fallbacks:
            check = getattr(fallback, "assert_compatible", None)
            if check is None:
                continue
            try:
                check(accepted)
            except ValueError:
                continue
            compatible.append(fallback)
        return tuple(compatible)

    @staticmethod
    def _hook_accepts_timeout(hook: PollHook) -> bool:
        try:
            parameters = inspect.signature(hook).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "timeout" or parameter.kind is parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _poll_hook(self, timeout: float | None) -> RateLimitSnapshot:
        hook = self.config.rate_limit_hook
        assert hook is not None
        if timeout is not None and timeout <= 0:
            raise RateLimitTimeout(f"timed out polling {self.provider} rate limits")
        value = (
            hook(self.credential, timeout=timeout)  # type: ignore[call-arg]
            if timeout is not None and self._hook_accepts_timeout(hook)
            else hook(self.credential)
        )
        snapshot = RateLimitSnapshot.parse(value)
        # Native refreshers (notably Kimi) may rotate the token while polling.
        # Mirror the new bundle into the wrapper-owned central store.
        with suppress(AuthenticationError):
            self._sync_native_credential()
        return snapshot

    def poll_rate_limits(self, *, timeout: float | None = None) -> RateLimitSnapshot:
        bound_logger = logger.bind(provider=self.provider, model=self.model)
        bound_logger.debug("Polling OAuth rate limits: timeout={timeout}", timeout=timeout)
        self.ensure_authenticated()
        if self.config.rate_limit_hook is not None:
            started = time.monotonic()
            try:
                snapshot = self._poll_hook(timeout)
            except AuthenticationError:
                bound_logger.warning("Rate-limit polling rejected the OAuth credential")
                self.force_relogin()
                remaining = (
                    None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
                )
                snapshot = self._poll_hook(remaining)
            bound_logger.info(
                "OAuth rate limits polled: limited={limited}, windows={windows}",
                limited=snapshot.limited,
                windows=[
                    {
                        "name": item.name,
                        "limited": item.limited,
                        "utilization": item.utilization,
                        "resets_at": item.resets_at,
                    }
                    for item in snapshot.limits
                ],
            )
            return snapshot
        if self.config.rate_limit_command is not None:
            command_timeout = self.config.rate_limit_command_timeout
            if command_timeout <= 0:
                raise ValueError("rate_limit_command_timeout must be positive")
            if timeout is not None:
                command_timeout = min(command_timeout, timeout)
            if command_timeout <= 0:
                raise RateLimitTimeout(f"timed out polling {self.provider} rate limits")
            try:
                result = subprocess.run(
                    tuple(self.config.rate_limit_command),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                )
                snapshot = RateLimitSnapshot.parse(json.loads(result.stdout))
                bound_logger.info(
                    "OAuth rate limits polled through command: limited={limited}, count={count}",
                    limited=snapshot.limited,
                    count=len(snapshot.limits),
                )
                return snapshot
            except subprocess.TimeoutExpired as exc:
                raise RateLimitTimeout(f"timed out polling {self.provider} rate limits") from exc
            except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot poll {self.provider} rate limits: {exc}") from exc
        raise RateLimitPollingUnsupported(
            f"{self.provider} has no rate-limit polling adapter configured"
        )

    def auto_wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 60.0 * 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> RateLimitSnapshot:
        """Wait for the reported reset, with periodic safety polls."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        started = clock()
        bound_logger = logger.bind(provider=self.provider, model=self.model)
        bound_logger.info(
            "Starting OAuth capacity wait: timeout={timeout}, poll_interval={poll_interval}",
            timeout=timeout,
            poll_interval=poll_interval,
        )
        while True:
            now = clock()
            remaining = None if timeout is None else timeout - (now - started)
            if remaining is not None and remaining <= 0:
                raise RateLimitTimeout(f"{self.provider} remains rate limited after {timeout:g}s")
            snapshot = self.poll_rate_limits(timeout=remaining)
            if not snapshot.limited:
                bound_logger.info("OAuth capacity is available; wait completed")
                return snapshot
            now = clock()
            if timeout is not None and now - started >= timeout:
                raise RateLimitTimeout(f"{self.provider} remains rate limited after {timeout:g}s")
            reset_delay = (
                None if snapshot.next_reset_at is None else max(0.0, snapshot.next_reset_at - now)
            )
            delay = (
                poll_interval
                if reset_delay is None
                else min(poll_interval, reset_delay or poll_interval)
            )
            if timeout is not None:
                delay = min(delay, max(0.0, timeout - (now - started)))
            if delay <= 0:
                raise RateLimitTimeout(f"{self.provider} remains rate limited after {timeout:g}s")
            bound_logger.warning(
                "OAuth capacity exhausted; sleeping for {delay:.3f}s before polling again",
                delay=delay,
            )
            sleep(delay)
            self.ensure_authenticated(now=clock())

    def wait_and_resume(self, resume: Callable[[], object], **wait_options: object) -> object:
        self.auto_wait(**wait_options)  # type: ignore[arg-type]
        self.ensure_authenticated()
        logger.bind(provider=self.provider, model=self.model).info(
            "Resuming callback after OAuth capacity wait"
        )
        return resume()

    def model_for(self, accepted: APIType | str | Iterable[APIType | str]) -> object:
        try:
            self.assert_compatible(accepted)
        except ValueError as original:
            fallback = self._compatible_fallback(accepted)
            if fallback is not None:
                logger.bind(provider=self.provider, model=self.model).info(
                    "Selected compatible fallback because OAuth protocol is incompatible: "
                    "fallback_model={fallback_model}",
                    fallback_model=getattr(fallback, "model", type(fallback).__name__),
                )
                return fallback
            raise original
        fallback = self._compatible_fallback(accepted)
        if fallback is None:
            if self._authentication_error is not None:
                raise self._authentication_error
            return self
        if self._authentication_error is not None:
            logger.bind(provider=self.provider, model=self.model).warning(
                "Selected fallback because OAuth authentication is unavailable: "
                "fallback_model={fallback_model}",
                fallback_model=getattr(fallback, "model", type(fallback).__name__),
            )
            return fallback
        try:
            if self.poll_rate_limits().limited:
                logger.bind(provider=self.provider, model=self.model).warning(
                    "Selected fallback because OAuth capacity is exhausted: "
                    "fallback_model={fallback_model}",
                    fallback_model=getattr(fallback, "model", type(fallback).__name__),
                )
                return fallback
        except (AuthenticationError, RateLimitTimeout, RuntimeError):
            logger.bind(provider=self.provider, model=self.model).warning(
                "Selected fallback because OAuth availability could not be established: "
                "fallback_model={fallback_model}",
                fallback_model=getattr(fallback, "model", type(fallback).__name__),
            )
            return fallback
        except RateLimitPollingUnsupported:
            pass
        return self

    def _compatible_fallback(
        self, accepted: APIType | str | Iterable[APIType | str]
    ) -> object | None:
        return next(iter(self.compatible_fallbacks(accepted)), None)
