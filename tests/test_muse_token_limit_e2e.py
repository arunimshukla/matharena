"""One native continuation at the token limit, with Muse's own placeholder."""

import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.muse_code import MuseCodeAgent
from test_harness_e2e import (
    _FakeCodexResponsesHandler, _TEST_CLI_RELEASES, _environment,
    _image_available, _sandbox, docker_workspace,
)


@pytest.mark.skipif(not _image_available(), reason="harness Docker image is not built")
@pytest.mark.parametrize("second_cutoff", [False, True])
def test_muse_token_limit_has_one_placeholder_continuation(docker_workspace, monkeypatch, second_cutoff):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)
    requests, modes, catalog_requests = [], [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer provider-key":
                self.send_error(401)
                return
            catalog_requests.append(self.path)
            body = json.dumps({"object": "list", "data": [{"id": "muse-spark-1.3", "metadata": {
                "muse-code": {"name": "muse-spark-1.3", "release_date": "2026-09-02", "is_hidden": False},
            }}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer provider-key":
                self.send_error(401)
                return
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append((modes[-1], payload))
            limited = not modes[-1] or second_cutoff
            item = None if limited else {
                "type": "message", "id": "final", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "CONTINUATION_OK", "annotations": []}],
            }
            response = _FakeCodexResponsesHandler._response(f"response_{len(requests)}", item)
            response["model"] = "muse-spark-1.3"
            if limited:
                response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[])
                response["usage"].update(output_tokens=262144, total_tokens=262152,
                                         output_tokens_details={"reasoning_tokens": 262141})
            events = [
                {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                *([{"type": "response.output_item.done", "output_index": 0, "item": item}] if item else []),
                {"type": "response.incomplete" if limited else "response.completed", "response": response},
            ]
            for sequence, event in enumerate(events):
                event["sequence_number"] = sequence
            body = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    original_build = MuseCodeAgent.build_command

    def command(self, *args, **kwargs):
        modes.append(kwargs.get("resume", False))
        result = original_build(self, *args, **kwargs)
        if not modes[-1]:
            i = result.index("--")
            result[i:i] = ["--max-model-steps", "1"]
        return result

    monkeypatch.setattr(MuseCodeAgent, "build_command", command)
    monkeypatch.setitem(_TEST_CLI_RELEASES, "muse", ("muse-code", "1.0.3-R2198.1"))
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    agent = Agent(type="muse", model=Model(
        "muse-spark-1.3", api_url=f"http://127.0.0.1:{upstream.server_port}/v1", api_key="sandbox-placeholder",
        headers={"Authorization": "Bearer provider-key"}, reasoning="max", request_overrides={"reasoning": {"effort": "max"}, "max_output_tokens": 262144},
    ), env=_sandbox(docker_workspace, "muse"), dir=docker_workspace,
        executable="/usr/bin/true", environment=_environment(), minimal_context=True, subagents={},
        validate_version=False, auto_wait=False, auto_fallback=False, max_recovery_attempts=1)

    def deadline(*args):
        raise TimeoutError("Muse token-limit test exceeded 60 seconds")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    try:
        events = agent.run("Remember ORIGINAL_TASK_MARKER and answer the question.")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
    assert modes == [False, True]
    assert catalog_requests == ["/muse-code/models"]
    assert [e.content["reason"] for e in events if e.type == "recovery"] == ["token_limit_resume"]
    users = [i for i in requests[0][1]["input"] if i.get("role") == "user"]
    continued = [p for resumed, p in requests if resumed]
    assert continued
    for _, request in requests:
        assert request["max_output_tokens"] == 262144 and request["reasoning"]["effort"] == "max"
        assert [i for i in request["input"] if i.get("role") == "user"] == users
        assert "Continue the interrupted task" not in json.dumps(request["input"])
    assert "[Previous turn ended without an assistant reply.]" in json.dumps(continued[0]["input"])
    assert events[-1].type == "result"
    assert events[-1].content == ("" if second_cutoff else "CONTINUATION_OK")
    if second_cutoff:
        assert events[-1].raw["reason"] == "max_output_tokens"
