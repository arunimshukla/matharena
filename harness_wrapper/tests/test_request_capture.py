from __future__ import annotations

import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import urlsplit

import pytest

from harness_wrapper.models.api_header_bridge import APIHeaderProxy
from harness_wrapper.models.request_capture import (
    RequestCaptureProxy,
    ResponseCapture,
    response_chunks,
)


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class _UpstreamHandler(BaseHTTPRequestHandler):
    captured_request: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        type(self).captured_request = {
            "path": self.path,
            "headers": dict(self.headers),
            "payload": json.loads(self.rfile.read(length)),
        }
        body = b'data: {"type":"response.completed","response":{"id":"r1"}}\n\n'
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_request_capture_proxy_observes_json_and_forwards_auth() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    observed: list[tuple[str, dict[str, Any]]] = []
    responses: list[tuple[str, dict[str, Any]]] = []
    try:
        with RequestCaptureProxy(
            upstream_url,
            lambda path, payload: observed.append((path, dict(payload))),
            upstream_headers={"X-Extra": "configured"},
            on_response=lambda path, payload: responses.append((path, dict(payload))),
            request_overrides={"provider": {"only": ["z-ai"]}},
        ) as proxy:
            parsed = urlsplit(proxy.base_url)
            unprotected = urllib.request.Request(
                f"http://127.0.0.1:{parsed.port}/v1/responses",
                data=b"{}",
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as rejected:
                urllib.request.urlopen(unprotected, timeout=5)
            assert rejected.value.code == 404

            payload = {"instructions": "system", "input": []}
            request = urllib.request.Request(
                f"{proxy.base_url}/responses?stream=true",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer oauth-secret",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert b"response.completed" in response.read()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    expected_payload = {**payload, "provider": {"only": ["z-ai"]}}
    assert observed == [("/v1/responses", expected_payload)]
    assert responses == [
        ("/v1/responses", {"type": "response.completed", "response": {"id": "r1"}})
    ]
    captured = _UpstreamHandler.captured_request
    assert captured["path"] == "/v1/responses?stream=true"
    assert captured["headers"]["Authorization"] == "Bearer oauth-secret"
    assert captured["headers"]["X-Extra"] == "configured"
    assert captured["payload"] == expected_payload


def test_request_capture_proxy_drops_nested_fields_after_merging() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    observed: list[tuple[str, dict[str, Any]]] = []
    try:
        with RequestCaptureProxy(
            upstream_url,
            lambda path, payload: observed.append((path, dict(payload))),
            request_overrides={"generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}},
            request_drop_fields=frozenset({"generationConfig.thinkingConfig.thinkingBudget"}),
            merge_request_overrides=True,
        ) as proxy:
            payload = {
                "generationConfig": {
                    "temperature": 0.4,
                    "thinkingConfig": {"thinkingBudget": -1},
                }
            }
            request = urllib.request.Request(
                f"{proxy.base_url}/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    expected = {
        "generationConfig": {
            "temperature": 0.4,
            "thinkingConfig": {"thinkingLevel": "high"},
        }
    }
    assert observed == [("/generate", expected)]
    assert _UpstreamHandler.captured_request["payload"] == expected


def test_response_capture_reports_final_cumulative_usage_once() -> None:
    observed: list[dict[str, Any]] = []
    capture = ResponseCapture(
        "/v1beta/models/test:streamGenerateContent",
        "text/event-stream",
        lambda path, payload: observed.append(dict(payload)),
        component="test",
    )
    capture.feed(
        b'data: {"candidates":[{"index":0}],"usageMetadata":'
        b'{"promptTokenCount":8,"candidatesTokenCount":1,'
        b'"cachedContentTokenCount":3}}\n\n'
    )
    capture.feed(
        b'data: {"candidates":[{"index":0,"finishReason":"STOP"}],"usageMetadata":'
        b'{"promptTokenCount":8,"candidatesTokenCount":2,'
        b'"cachedContentTokenCount":3}}\n\n'
    )
    capture.finish()

    assert [payload for payload in observed if "candidates" in payload] == [
        {"candidates": [{"index": 0}]},
        {"candidates": [{"index": 0, "finishReason": "STOP"}]},
    ]
    assert [payload for payload in observed if "usageMetadata" in payload] == [
        {
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 2,
                "cachedContentTokenCount": 3,
            }
        }
    ]


@pytest.mark.parametrize("complete", [False, True])
def test_chat_stream_counts_latest_usage_once_even_when_interrupted(complete) -> None:
    observed = []
    capture = ResponseCapture(
        "/v1/chat/completions", "text/event-stream",
        lambda path, payload: observed.append(dict(payload)), component="test",
    )
    outputs = [4, 20, 51] if complete else [4, 20]
    for output in outputs + outputs[-1:]:
        payload = {
            "object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": 19, "completion_tokens": output,
                      "total_tokens": 19 + output},
        }
        capture.feed(("data: " + json.dumps(payload) + "\n\n").encode())
    assert not any("usage" in payload for payload in observed)
    capture.finish()
    capture.finish()
    usages = [payload["usage"] for payload in observed if "usage" in payload]
    assert usages == [{"prompt_tokens": 19, "completion_tokens": outputs[-1],
                       "total_tokens": 19 + outputs[-1]}]
    assert len([payload for payload in observed if "choices" in payload]) == len(outputs) + 1


def test_request_capture_proxy_can_listen_on_unix_socket(tmp_path) -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    socket_path = tmp_path / "proxy.sock"
    try:
        with RequestCaptureProxy(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            lambda path, payload: None,
            client_host="10.200.0.1",
            allow_remote_clients=True,
            unix_socket=socket_path,
            client_port=32123,
        ) as proxy:
            advertised = urlsplit(proxy.base_url)
            assert advertised.hostname == "10.200.0.1"
            assert advertised.port == 32123
            connection = _UnixHTTPConnection(str(socket_path))
            connection.request(
                "POST",
                f"{advertised.path}/responses",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with connection.getresponse() as response:
                assert response.status == 200
                assert b"response.completed" in response.read()
            connection.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "error", [http.client.IncompleteRead(b"buffered tail"), ConnectionResetError()]
)
def test_interrupted_response_preserves_buffered_bytes(error):
    class BrokenStream:
        calls = 0

        def read(self, size):
            self.calls += 1
            if self.calls == 1:
                return b"first chunk"
            raise error

    assert b"".join(response_chunks(BrokenStream())) == b"first chunk" + getattr(
        error, "partial", b""
    )


@pytest.mark.parametrize("proxy_kind", ["capture", "api_header"])
@pytest.mark.parametrize("chunked", [False, True])
def test_proxy_forwards_small_events_before_upstream_finishes(proxy_kind, chunked):
    release = threading.Event()
    first = b'data: {"type":"response.created"}\n\n'
    last = b'data: {"type":"response.completed"}\n\n'

    class StreamingUpstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                for part in (first, last):
                    self.wfile.write(
                        f"{len(part):x}\r\n".encode() + part + b"\r\n" if chunked else part
                    )
                    self.wfile.flush()
                    if part == first:
                        release.wait(5)
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{upstream.server_address[1]}"
    if proxy_kind == "api_header":
        proxy = APIHeaderProxy(url, {"x-api-key": "token"})
    else:
        proxy = RequestCaptureProxy(url, lambda path, payload: None)
    try:
        with proxy:
            request = urllib.request.Request(proxy.base_url + "/responses", data=b"{}")
            if proxy_kind != "capture":
                request.add_header("Authorization", f"Bearer {proxy.client_api_key}")
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    # The upstream waits for us: forwarding must not wait for EOF or 64 KiB.
                    assert response.readline() == first.splitlines(keepends=True)[0]
                    release.set()
                    assert response.read() == b"\n" + last
            finally:
                release.set()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("proxy_kind", ["capture", "api_header", "codex_oauth"])
def test_stopped_run_cannot_send_an_upstream_request(proxy_kind, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Stopped attempt contacted an upstream provider")

    def gate(path, payload):
        return False
    if proxy_kind == "codex_oauth":
        from harness_wrapper.models.oauth.openai_bridge import CodexOAuthResponsesProxy
        monkeypatch.setattr("harness_wrapper.models.oauth.openai_bridge.connect", unexpected)
        monkeypatch.setattr(
            "harness_wrapper.models.oauth.openai_bridge.load_central_openai_oauth_credentials",
            unexpected,
        )
        proxy = CodexOAuthResponsesProxy(on_request=gate)
    elif proxy_kind == "api_header":
        proxy = APIHeaderProxy("http://127.0.0.1:1", {"x-api-key": "test"}, on_request=gate)
    else:
        proxy = RequestCaptureProxy("http://127.0.0.1:1", gate)
    with proxy:
        request = urllib.request.Request(proxy.base_url + "/responses", data=b"{}")
        if proxy_kind != "capture":
            request.add_header("Authorization", f"Bearer {proxy.client_api_key}")
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(request, timeout=2)
        assert rejected.value.code == 409
        assert b"harness run limit reached" in rejected.value.read()
