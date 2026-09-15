from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from harness_wrapper.models import AuthenticationError
from harness_wrapper.models.oauth import CredentialStore, OAuthCredential, openai_bridge
from harness_wrapper.models.oauth.openai_bridge import (
    CodexOAuthCredentials,
    CodexOAuthResponsesProxy,
    load_central_openai_oauth_credentials,
    load_codex_oauth_credentials,
    pace_codex_connections,
)


def _token(*, expires_at: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires_at}).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def _write_auth(path: Path, *, expires_at: float | None = None) -> None:
    token = "opaque-token" if expires_at is None else _token(expires_at=expires_at)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": token, "account_id": "account-1"},
            }
        )
    )
    path.chmod(0o600)


def test_codex_credentials_are_loaded_securely_and_redacted(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth(path, expires_at=time.time() + 3600)
    credentials = load_codex_oauth_credentials(path)
    assert credentials.access_token.startswith("header.")
    assert credentials.account_id == "account-1"
    assert credentials.access_token not in repr(credentials)
    assert credentials.account_id not in repr(credentials)


def test_codex_credentials_reject_unsafe_or_expired_files(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth(path)
    path.chmod(0o644)
    with pytest.raises(AuthenticationError, match="mode 0600"):
        load_codex_oauth_credentials(path)

    path.chmod(0o600)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(AuthenticationError, match="cannot import"):
        load_codex_oauth_credentials(link)

    _write_auth(path, expires_at=time.time() - 60)
    with pytest.raises(AuthenticationError, match="expired"):
        load_codex_oauth_credentials(path)


def test_bridge_credentials_come_from_central_wrapper_store(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save(
        "openai",
        OAuthCredential(
            "central-access",
            "central-refresh",
            time.time() + 3600,
            {"account_id": "central-account", "cli_managed": True},
        ),
    )
    credentials = load_central_openai_oauth_credentials(store)
    assert credentials.access_token == "central-access"
    assert credentials.account_id == "central-account"


def test_bridge_can_reload_credentials_after_relogin(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = CodexOAuthCredentials("fresh-token", "fresh-account")
    monkeypatch.setattr(
        "harness_wrapper.models.oauth.openai_bridge.load_central_openai_oauth_credentials",
        lambda: fresh,
    )
    bridge = CodexOAuthResponsesProxy(credentials=CodexOAuthCredentials("old-token", "old-account"))

    assert bridge.reload_credentials() == fresh
    assert bridge.credentials == fresh


@contextmanager
def _upstream(handler, **kwargs):
    with serve(handler, "127.0.0.1", 0, **kwargs) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/responses"
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _request(bridge, payload=None):
    return urllib.request.Request(
        bridge.base_url + "/responses",
        data=json.dumps(payload or {"model": "gpt-test", "input": []}).encode(),
        headers={"Authorization": f"Bearer {bridge.client_api_key}"},
    )


def test_bridge_translates_to_websocket_without_leaking_credentials():
    captured = {}
    observed = []
    responses = []
    completed = {
        "type": "response.completed",
        "response": {
            "id": "r1",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 11,
            },
        },
    }

    def handler(ws):
        captured.update(
            path=ws.request.path, headers=dict(ws.request.headers), payload=json.loads(ws.recv())
        )
        ws.send(json.dumps(completed))

    def response_headers(connection, request, response):
        response.headers["x-codex-turn-state"] = "upstream-turn-state"
        response.headers["set-cookie"] = "private=upstream-cookie"
        return response

    with (
        _upstream(handler, process_response=response_headers) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("secret-token", "account-1"),
            upstream_url=url,
            on_request=lambda path, payload: observed.append((path, dict(payload))),
            on_response=lambda path, payload: responses.append((path, dict(payload))),
            request_overrides={"tools": [], "tool_choice": "none", "parallel_tool_calls": False},
        ) as bridge,
    ):
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(
                urllib.request.Request(bridge.base_url + "/responses", data=b"{}"), timeout=5
            )
        assert rejected.value.code == 403
        assert captured == {}
        request = _request(
            bridge,
            {
                "model": "gpt-test",
                "input": [],
                "stream": True,
                "background": False,
                "max_output_tokens": 32768,
                "tools": [{"type": "function", "name": "forbidden"}],
            },
        )
        for key, value in {
            "ChatGPT-Account-Id": "untrusted-account",
            "Session-Id": "native-session",
            "Thread-Id": "native-thread",
            "Originator": "codex_exec",
            "User-Agent": "codex-cli/local-test",
            "X-Codex-Turn-State": "previous-turn-state",
            "X-Codex-Turn-Metadata": "native-metadata",
            "X-Codex-Beta-Features": "native-features",
            "X-Client-Request-Id": "native-request",
            "X-OpenAI-Internal-Codex-Responses-Lite": "true",
            "Cookie": "private=client-cookie",
        }.items():
            request.add_header(key, value)
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.headers["x-codex-turn-state"] == "upstream-turn-state"
            assert "set-cookie" not in response.headers
            assert json.loads(response.read().removeprefix(b"data: ")) == completed

    expected = {
        "type": "response.create",
        "model": "gpt-test",
        "input": [],
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": False,
    }
    assert captured["path"] == "/responses"
    assert captured["payload"] == expected
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["authorization"] == "Bearer secret-token"
    assert headers["chatgpt-account-id"] == "account-1"
    assert headers["session-id"] == "native-session"
    assert headers["thread-id"] == "native-thread"
    assert headers["originator"] == "codex_exec"
    assert headers["user-agent"] == "codex-cli/local-test"
    assert headers["x-codex-turn-state"] == "previous-turn-state"
    assert headers["x-codex-turn-metadata"] == "native-metadata"
    assert headers["x-codex-beta-features"] == "native-features"
    assert headers["x-client-request-id"] == "native-request"
    assert headers["x-openai-internal-codex-responses-lite"] == "true"
    assert "cookie" not in headers
    assert bridge.client_api_key not in json.dumps(captured)
    assert observed == [("/responses", expected)]
    assert responses == [("/responses", completed)]


def test_bridge_streams_websocket_events_before_completion():
    release = threading.Event()
    first = {"type": "response.created", "response": {"id": "r1"}}
    last = {"type": "response.completed", "response": {"id": "r1"}}

    def handler(ws):
        ws.recv()
        encoded = json.dumps(first)
        ws.send([encoded[:10], encoded[10:]])
        release.wait(5)
        with suppress(ConnectionClosed):
            ws.send(json.dumps(last))

    with (
        _upstream(handler) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"),
            upstream_url=url,
        ) as bridge,
    ):
        try:
            with urllib.request.urlopen(_request(bridge), timeout=2) as response:
                assert json.loads(response.readline().removeprefix(b"data: ")) == first
                release.set()
                assert json.loads(response.read().strip().removeprefix(b"data: ")) == last
        finally:
            release.set()


@pytest.mark.parametrize(
    "kind", ["response.completed", "response.failed", "response.incomplete", "error"]
)
def test_bridge_closes_websocket_on_terminal_event(kind):
    closed = threading.Event()

    def handler(ws):
        ws.recv()
        ws.send(json.dumps({"type": kind}))
        with suppress(ConnectionClosed):
            ws.recv(timeout=3)
        closed.set()

    with (
        _upstream(handler) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"),
            upstream_url=url,
        ) as bridge,
    ):
        with urllib.request.urlopen(_request(bridge), timeout=2) as response:
            assert json.loads(response.read().removeprefix(b"data: ")) == {"type": kind}
        assert closed.wait(2)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_bridge_preserves_websocket_handshake_errors(status):
    def reject(connection, request):
        response = connection.respond(status, '{"error":{"message":"rejected"}}')
        response.headers["retry-after"] = "12"
        return response

    with (
        _upstream(lambda ws: None, process_request=reject) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"),
            upstream_url=url,
        ) as bridge,
    ):
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(_request(bridge), timeout=2)
        assert rejected.value.code == status
        assert rejected.value.headers["retry-after"] == "12"
        assert json.loads(rejected.value.read()) == {"error": {"message": "rejected"}}


def test_bridge_does_not_report_completion_after_websocket_disconnect():
    observed = []
    first = {"type": "response.created", "response": {"id": "r1"}}

    def handler(ws):
        ws.recv()
        ws.send(json.dumps(first))
        ws.close(code=1011, reason="simulated upstream disconnect")

    with (
        _upstream(handler) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"),
            upstream_url=url,
            on_response=lambda path, payload: observed.append(dict(payload)),
        ) as bridge,
        urllib.request.urlopen(_request(bridge), timeout=2) as response,
    ):
        assert json.loads(response.read().removeprefix(b"data: ")) == first
    assert observed == [first]


def test_connection_pacing_is_shared_and_does_not_bank_idle_slots(monkeypatch):
    now = [100.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(openai_bridge, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=sleep,
    ))
    assert openai_bridge._wait_for_connection is None
    with pytest.raises(RuntimeError, match="stop batch"), pace_codex_connections(2):
        with ThreadPoolExecutor(max_workers=16) as workers:
            list(workers.map(lambda _: openai_bridge._wait_for_connection(), range(64)))
        assert sleeps == [2] * 63
        now[0] += 100  # No accumulated burst allowance after an idle period.
        openai_bridge._wait_for_connection()
        assert len(sleeps) == 63
        openai_bridge._wait_for_connection()
        assert sleeps == [2] * 64
        raise RuntimeError("stop batch")
    assert openai_bridge._wait_for_connection is None


def test_pacing_applies_to_multiple_bridges_retries_and_continuations(monkeypatch):
    starts = []
    real_connect = openai_bridge.connect

    def connect(*args, **kwargs):
        starts.append(time.monotonic())
        return real_connect(*args, **kwargs)

    def reject_first(connection, request):
        if len(starts) == 1:
            return connection.respond(403, "rejected")

    def handler(ws):
        ws.recv()
        ws.send(json.dumps({"type": "response.completed"}))

    monkeypatch.setattr(openai_bridge, "connect", connect)
    with (
        _upstream(handler, process_request=reject_first) as url,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"), upstream_url=url,
        ) as first,
        CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("token", "account"), upstream_url=url,
        ) as second,
        pace_codex_connections(0.1),
    ):
        # A rejected handshake consumes a slot, as does its retry.
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(_request(first), timeout=5)
        assert rejected.value.code == 403

        def request(bridge):
            with urllib.request.urlopen(_request(bridge), timeout=5) as response:
                response.read()

        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(request, [first, second, first, second]))
        assert len(starts) == 5
        assert all(b - a >= 0.09 for a, b in pairwise(starts))
    assert openai_bridge._wait_for_connection is None
