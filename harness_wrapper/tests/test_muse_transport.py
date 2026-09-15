"""Muse retries the exact HTTP request, never a new conversation turn."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness_wrapper.models.request_capture import RequestCaptureProxy
from test_muse_code import _agent


@pytest.mark.parametrize("kind", ["server_error", "http_server_error", "unauthorized", "empty", "limit"])
def test_muse_request_retry_preserves_payload_and_hides_failed_output(tmp_path, monkeypatch, kind):
    agent = _agent(tmp_path)
    requests, delays = [], []
    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.time.sleep", delays.append)

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.rfile.read(int(self.headers['content-length'])), self.headers['Authorization']))
            number = len(requests)
            status = 200
            item = {"type": "function_call", "name": "muse.bash", "id": "failed-tool",
                    "call_id": "failed-call", "arguments": '{"command":"must not execute"}'}
            response = {"id": f"r{number}", "status": "completed", "output": [],
                        "usage": {"input_tokens": 10, "output_tokens": 7}}
            event = {"type": "response.completed", "response": response}
            if kind in {"server_error", "http_server_error"} and number <= 2:
                response.update(status="failed", error={"code": "server_error", "message": "Failed"})
                event['type'] = 'response.failed'
            elif kind == "unauthorized":
                status = 401
            elif kind == "limit":
                event['type'] = 'response.incomplete'
                response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            elif kind != "empty":
                response['output'] = [{"type": "message", "content": [{"type": "output_text", "text": "x" * 1100000}]}]
            content_type = 'text/event-stream'
            if status == 401 or (kind == 'http_server_error' and number <= 2):
                content_type = 'application/json'
                status = 401 if kind == 'unauthorized' else 500
                body = json.dumps({"error": {"code": "invalid_api_key" if status == 401 else "server_error"}}).encode()
            else:
                events = ([{"type": "response.output_item.done", "item": item}]
                          if event['type'] == 'response.failed' else []) + [event]
                body = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        with RequestCaptureProxy(f'http://127.0.0.1:{upstream.server_port}/v1', lambda *args: None,
                                 on_response=agent._capture_model_response,
                                 open_upstream=agent._open_model_request) as proxy:
            payload = {"model": "spark", "input": [{"role": "user", "content": "Original"},
                       {"type": "reasoning", "encrypted_content": "opaque"}],
                       "max_output_tokens": 262144, "reasoning": {"effort": "max"}}
            request = urllib.request.Request(proxy.base_url + '/responses', data=json.dumps(payload).encode(),
                                             headers={'Authorization': 'Bearer fake'})
            try:
                response = urllib.request.urlopen(request, timeout=10)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                delivered = response.read()
                assert response.status == (401 if kind == 'unauthorized' else 500 if kind == 'http_server_error' else 200)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
    retry = kind == 'server_error'
    assert len(requests) == (3 if retry else 1)
    assert all(r == requests[0] for r in requests)
    assert json.loads(requests[0][0]) == payload
    assert delays == ([60, 60] if retry else [])
    assert b'failed-tool' not in delivered
    assert not any(e.type == 'tool_call' for e in agent._drain_provider_events())
    assert agent.get_tokens().output_tokens == (21 if kind == 'server_error' else 0 if kind in {'unauthorized', 'http_server_error'} else 7)


def test_muse_request_retry_is_bounded_and_surfaces_last_failure(tmp_path, monkeypatch):
    import io
    from urllib.response import addinfourl
    agent = _agent(tmp_path)
    agent.max_recovery_attempts = 2
    calls, delays = [], []
    body = b'data: {"type":"response.failed","response":{"id":"failed","error":{"code":"server_error"}}}\n\n'

    def open_request(request, **kwargs):
        calls.append(request)
        return addinfourl(io.BytesIO(body), {"content-type": "text/event-stream"}, request.full_url, 200)

    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.urllib.request.urlopen", open_request)
    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.time.sleep", delays.append)
    request = urllib.request.Request('http://local/v1/responses', data=b'{"input":[]}')
    with agent._open_model_request(request, timeout=5) as response:
        assert response.read() == body
    assert calls == [request, request, request]
    assert delays == [60, 60]


@pytest.mark.parametrize("kind", ["generic", "different", "non_json", "summary", "disabled"])
def test_muse_budget_retry_is_bounded_and_preserves_errors(tmp_path, monkeypatch, kind):
    import io

    agent = _agent(tmp_path)
    agent.max_recovery_attempts = 0 if kind == "disabled" else 2
    agent._request_context.is_compaction = kind == "summary"
    body = json.dumps({"error": {
        "type": "invalid_request_error",
        "message": ("Unknown model" if kind == "different" else
                    "The request contains invalid parameters. Check the request body for any errors or inconsistencies."),
    }}).encode() if kind != "non_json" else b"Invalid request"
    requests, delays = [], []
    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.time.sleep", delays.append)

    def reject(request, **kwargs):
        requests.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.urllib.request.urlopen", reject)
    original = {"max_output_tokens": 1000000, "input": [
        {"type": "reasoning", "encrypted_content": "preserve-me"},
        {"role": "user", "content": "Solve it"}], "reasoning": {"effort": "max"}}
    request = urllib.request.Request("https://local/v1/responses", data=json.dumps(original).encode())
    with pytest.raises(urllib.error.HTTPError) as caught:
        agent._open_model_request(request, timeout=5)
    assert caught.value.code == 400
    assert caught.value.read() == body
    assert [p["max_output_tokens"] for p in requests] == (
        [1000000, 500000, 250000] if kind == "generic" else [1000000]
    )
    assert all({**p, "max_output_tokens": 1000000} == original for p in requests)
    assert delays == ([60, 60] if kind == "generic" else [])


def test_muse_reduced_budget_does_not_override_lower_limits_or_summary(tmp_path):
    agent = _agent(tmp_path)
    agent._solver_output_budget = 500000
    for requested, expected in [(1000000, 500000), (128000, 128000)]:
        payload = {"max_output_tokens": requested}
        agent._filter_request("/v1/responses", payload)
        assert payload["max_output_tokens"] == expected
    summary = {"max_output_tokens": 600000,
               "tools": [{"type": "function", "name": "generate_summary"}]}
    agent._filter_request("/v1/responses", summary)
    assert summary["max_output_tokens"] == 600000
