"""Small administrative CLI for harness-wrapper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from loguru import logger

from .installation import cli_environment, install_clis, installed_versions
from .model import create_model
from .models.oauth import (
    AuthenticationError,
    OAuthConfig,
    OAuthCredential,
    anthropic_oauth_config,
    kimi_oauth_config,
    openai_oauth_config,
)

OAUTH_PROVIDERS = {
    "openai": "OpenAI plan through Codex CLI",
    "anthropic": "Anthropic plan through Claude Code",
    "kimi": "Kimi plan through Kimi Code",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-wrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install-clis", help="install agent CLIs")
    # Validation lives in install_clis(). argparse's choices handling for a
    # zero-length ``nargs='*'`` positional is inconsistent across supported
    # Python releases (3.10 treats the empty list as a choice).
    install.add_argument("names", nargs="*", metavar="CLI")
    install.add_argument("--prefix")
    install.add_argument(
        "--version",
        action="append",
        default=[],
        metavar="CLI=VERSION",
        help="pin one selected CLI; omitted CLIs resolve latest",
    )
    install.add_argument("--dry-run", action="store_true")
    versions = subparsers.add_parser("versions", help="show private CLI versions")
    versions.add_argument("--prefix")
    login = subparsers.add_parser("login", help="log in to an OAuth plan provider")
    login.add_argument("provider", nargs="?", metavar="PROVIDER")
    login.add_argument("--force", action="store_true", help="run login even if already signed in")
    login.add_argument("--list", action="store_true", help="list providers without prompting")
    login.add_argument(
        "--local-callback",
        action="store_true",
        help="use Codex's localhost callback instead of device authorization",
    )
    return parser


def _show_oauth_providers() -> None:
    print("Available OAuth providers (✅ logged in, ❌ not logged in):")
    with _private_cli_path():
        for index, (provider, description) in enumerate(OAUTH_PROVIDERS.items(), 1):
            status = "✅" if _oauth_logged_in(provider) else "❌"
            print(f"  {index}. {status} {provider:<10} {description}")


def _select_oauth_provider() -> str | None:
    _show_oauth_providers()
    if not sys.stdin.isatty():
        print("Run `harness-wrapper login PROVIDER` to log in.")
        return None
    try:
        selection = input("Select a provider by name or number: ").strip().lower()
    except EOFError:
        return None
    names = tuple(OAUTH_PROVIDERS)
    if selection.isdigit() and 1 <= int(selection) <= len(names):
        return names[int(selection) - 1]
    return selection or None


@contextmanager
def _private_cli_path() -> Iterator[None]:
    """Make CLIs installed by ``install-clis`` visible to OAuth adapters."""

    previous = os.environ.get("PATH")
    os.environ["PATH"] = cli_environment(base=os.environ)["PATH"]
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous


def _oauth_config(
    provider: str,
    *,
    auto_login: bool,
    local_callback: bool = False,
) -> OAuthConfig:
    if provider == "openai":
        return openai_oauth_config(
            device_auth=not local_callback,
            auto_login=auto_login,
        )
    if provider == "anthropic":
        return anthropic_oauth_config(auto_login=auto_login)
    if provider == "kimi":
        return kimi_oauth_config(auto_login=auto_login)
    raise ValueError(f"unknown OAuth provider: {provider}")


def _oauth_logged_in(provider: str) -> bool:
    """Check native CLI state and synchronize its token into the central store."""

    bound_logger = logger.bind(component="oauth_cli", provider=provider)
    bound_logger.debug("Checking provider OAuth login status")
    try:
        config = _oauth_config(provider, auto_login=False)
        if config.status_hook is not None:
            if not config.status_hook():
                return False
            if config.credential_loader is not None:
                loaded = config.credential_loader()
                if isinstance(loaded, OAuthCredential):
                    synced = loaded
                elif isinstance(loaded, Mapping):
                    synced = OAuthCredential.from_dict(loaded)
                else:
                    return False
                config.store.save(provider, synced)
                active = not synced.expired and synced.access_token is not None
                bound_logger.debug(
                    "Provider OAuth status checked: logged_in={logged_in}", logged_in=active
                )
                return active
            return True
        credential = config.store.load(provider)
        if credential is None:
            return False
        if credential.access_token is None:
            config.store.delete(provider)
            return False
        return not credential.expired
    except (AuthenticationError, OSError) as error:
        bound_logger.warning(
            "Provider OAuth status check failed: error_type={error_type}",
            error_type=type(error).__name__,
        )
        return False


def _login(provider: str, *, force: bool, local_callback: bool) -> int:
    if provider not in OAUTH_PROVIDERS:
        choices = ", ".join(OAUTH_PROVIDERS)
        print(f"Unknown OAuth provider {provider!r}; choose one of: {choices}", file=sys.stderr)
        return 2
    if local_callback and provider != "openai":
        print("--local-callback is only supported for the openai provider", file=sys.stderr)
        return 2
    print(f"Authenticating with {provider}...")
    bound_logger = logger.bind(component="oauth_cli", provider=provider)
    bound_logger.info(
        "Starting OAuth login command: force={force}, local_callback={local_callback}",
        force=force,
        local_callback=local_callback,
    )
    try:
        with _private_cli_path():
            config = _oauth_config(
                provider,
                auto_login=not force,
                local_callback=local_callback,
            )
            model: Any = create_model(
                "oauth-login",
                oauth=provider,
                oauth_config=config,
            )
            credential = model.login() if force else model.ensure_authenticated()
    except (AuthenticationError, OSError) as error:
        bound_logger.error(
            "OAuth login command failed: error_type={error_type}",
            error_type=type(error).__name__,
        )
        print(f"OAuth login failed for {provider}: {error}", file=sys.stderr)
        return 1
    mode = "CLI-managed session" if credential.extra.get("cli_managed") else "credential"
    print(f"Authenticated with {provider} ({mode}).")
    print(f"OAuth tokens stored in {config.store.path}.")
    bound_logger.info("OAuth login command completed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "install-clis":
        versions: dict[str, str] = {}
        for item in args.version:
            name, separator, version = item.partition("=")
            if not separator or not name or not version:
                raise ValueError("--version must use CLI=VERSION")
            versions[name] = version
        urls = install_clis(
            args.names or None,
            prefix=args.prefix,
            versions=versions,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("\n".join(urls))
        return 0
    if args.command == "versions":
        print(json.dumps(installed_versions(args.prefix), indent=2, sort_keys=True))
        return 0
    if args.command == "login":
        if args.list:
            _show_oauth_providers()
            return 0
        provider = args.provider or _select_oauth_provider()
        return (
            0
            if provider is None
            else _login(
                provider,
                force=args.force,
                local_callback=args.local_callback,
            )
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
