from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from harness_wrapper.models.api_header_bridge import APIHeaderProxy


class _UpstreamHandler(BaseHTTPRequestHandler):
    captured_request: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        type(self).captured_request = {
            "path": self.path,
            "headers": dict(self.headers),
            "body": self.rfile.read(length),
        }
        body = b'{"type":"reasoning","encrypted_content":"opaque"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_api_header_proxy_protects_and_injects_custom_headers() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    virtual_key = "secret-virtual-key"
    observed: list[tuple[str, dict[str, Any]]] = []
    responses: list[tuple[str, dict[str, Any]]] = []
    try:
        with APIHeaderProxy(
            upstream_url,
            {"x-bf-vk": virtual_key},
            on_request=lambda path, payload: observed.append((path, dict(payload))),
            on_response=lambda path, payload: responses.append((path, dict(payload))),
            request_overrides={"extra_params": {"provider": {"only": ["z-ai"]}}},
            request_drop_fields=("prompt_cache_key",),
        ) as bridge:
            unauthorized = urllib.request.Request(
                f"{bridge.base_url}/chat/completions",
                data=b"{}",
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as rejected:
                urllib.request.urlopen(unauthorized, timeout=5)
            assert rejected.value.code == 403

            request = urllib.request.Request(
                f"{bridge.base_url}/chat/completions?stream=false",
                data=b'{"model":"test","prompt_cache_key":"unsupported"}',
                headers={
                    "Accept-Encoding": "gzip",
                    "Authorization": f"Bearer {bridge.client_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.read() == b'{"type":"reasoning","encrypted_content":"opaque"}'
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    captured = _UpstreamHandler.captured_request
    assert captured["path"] == "/v1/chat/completions?stream=false"
    assert captured["headers"]["X-Bf-Vk"] == virtual_key
    assert captured["headers"].get("Accept-Encoding", "identity") != "gzip"
    assert "Authorization" not in captured["headers"]
    assert bridge.client_api_key not in str(captured)
    expected_payload = {
        "model": "test",
        "extra_params": {"provider": {"only": ["z-ai"]}},
    }
    assert json.loads(captured["body"]) == expected_payload
    assert observed == [("/v1/chat/completions", expected_payload)]
    assert responses == [
        ("/v1/chat/completions", {"type": "reasoning", "encrypted_content": "opaque"})
    ]


class _ToolRoundTripUpstreamHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        type(self).requests.append(json.loads(self.rfile.read(length)))
        if len(type(self).requests) == 1:
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "Bash",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "extra_content": {
                                            "google": {"thought_signature": "opaque-signature"}
                                        },
                                        "function": {"arguments": "{}"},
                                    }
                                ]
                            },
                        }
                    ]
                },
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            body += "data: [DONE]\n\n"
            encoded = body.encode()
            content_type = "text/event-stream"
        else:
            encoded = b'{"choices":[{"message":{"content":"done"}}]}'
            content_type = "application/json"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_api_header_proxy_restores_streamed_tool_call_metadata() -> None:
    _ToolRoundTripUpstreamHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ToolRoundTripUpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    try:
        with APIHeaderProxy(upstream_url, {"x-api-key": "secret"}) as bridge:
            headers = {
                "Authorization": f"Bearer {bridge.client_api_key}",
                "Content-Type": "application/json",
            }
            first = urllib.request.Request(
                f"{bridge.base_url}/chat/completions",
                data=b'{"model":"gemini","messages":[{"role":"user","content":"run it"}]}',
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(first, timeout=5) as response:
                response.read()

            second_payload = {
                "model": "gemini",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "42"},
                ],
            }
            second = urllib.request.Request(
                f"{bridge.base_url}/chat/completions",
                data=json.dumps(second_payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(second, timeout=5) as response:
                response.read()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    restored_call = _ToolRoundTripUpstreamHandler.requests[1]["messages"][0]["tool_calls"][0]
    assert restored_call["extra_content"] == {"google": {"thought_signature": "opaque-signature"}}
