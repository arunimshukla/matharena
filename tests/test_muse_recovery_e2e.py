"""Real Muse request retries with a local provider; no Meta credentials or API calls."""

import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness_wrapper import Agent, Model
from test_harness_e2e import (
    _FakeCodexResponsesHandler,
    _TEST_CLI_RELEASES,
    _environment,
    _image_available,
    _sandbox,
    docker_workspace as _workspace_fixture,
)


docker_workspace = _workspace_fixture


@pytest.mark.skipif(not _image_available(), reason="harness Docker image is not built")
@pytest.mark.parametrize("failure", ["empty", "server_error", "output_budget"])
def test_muse_retries_failed_request_without_another_turn(docker_workspace, monkeypatch, failure):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            catalog = {"object": "list", "data": [{
                "id": "muse-spark-1.3", "object": "model", "metadata": {"muse-code": {
                    "name": "muse-spark-1.3", "release_date": "2026-09-02", "is_hidden": False,
                }},
            }]}
            body = json.dumps(catalog).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(payload)
            number = len(requests)
            if number == 2 and failure == "output_budget":
                body = json.dumps({"error": {
                    "type": "invalid_request_error",
                    "message": "The request contains invalid parameters. Check the request body for any errors or inconsistencies.",
                }}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if number in [1, 3]:
                item = {
                    "type": "function_call", "name": "muse.bash", "call_id": f"progress_{number}",
                    "id": f"fc_progress_{number}", "status": "completed",
                    "arguments": json.dumps({
                        "command": ("echo RETAINED_PROGRESS > progress.txt; cat progress.txt"
                                    if number == 1 else "cat progress.txt"),
                        "description": "Save progress",
                    }),
                }
            elif number == 2 and failure == "empty":
                item = {"type": "reasoning", "id": "rs_empty", "summary": [], "status": "completed", "encrypted_content": "retained-reasoning"}
            else:
                item = {
                    "type": "message", "id": f"msg_{number}", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                        "text": "Still working." if number == 2 else "RECOVERY_OK\n" + "x" * 21000, "annotations": []}],
                }
                if number == 2:
                    item["phase"] = "commentary"
            response = _FakeCodexResponsesHandler._response(f"resp_{number}", item)
            response["model"] = "muse-spark-1.3"
            terminal = "response.completed"
            if number == 2 and failure == "empty":
                response["usage"].update(output_tokens=19739, total_tokens=19747,
                                         output_tokens_details={"reasoning_tokens": 19735})
            if number == 2 and failure == "server_error":
                terminal = "response.failed"
                response.update(status="failed", error={
                    "code": "server_error", "message": "The model failed to generate a response.",
                })
            events = [
                {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": terminal, "response": response},
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

    monkeypatch.setitem(_TEST_CLI_RELEASES, "muse", ("muse-code", "1.0.3-R2198.1"))
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    agent = Agent(
        type="muse", model=Model(
            "muse-spark-1.3", api_url=f"http://127.0.0.1:{upstream.server_port}/v1",
            api_key="fake-key", reasoning="max",
            request_overrides={"reasoning": {"effort": "max"}, "max_output_tokens": 262144},
        ),
        env=_sandbox(docker_workspace, "muse"), dir=docker_workspace,
        executable="/usr/bin/true", environment=_environment(), subagents={},
        minimal_context=True, validate_version=False, auto_wait=False, auto_fallback=False,
        max_recovery_attempts=2,
    )

    def deadline(*args):
        raise TimeoutError("Muse local recovery test exceeded 60 seconds")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    try:
        events = agent.run("Remember CONVERSATION_MARKER, save progress using bash, then finish.")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert len(requests) == 4
    assert (docker_workspace / "progress.txt").read_text().strip() == "RETAINED_PROGRESS"
    assert "CONVERSATION_MARKER" in json.dumps(requests[-1]["input"])
    assert "RETAINED_PROGRESS" in json.dumps(requests[-1]["input"])
    if failure == "server_error":
        assert requests[1] == requests[2]
    assert all("Continue the interrupted task" not in json.dumps(p["input"]) for p in requests)
    assert all(p["reasoning"]["effort"] == "max" for p in requests)
    if failure == "output_budget":
        assert [p["max_output_tokens"] for p in requests] == [262144, 262144, 131072, 131072]
        assert {k: v for k, v in requests[1].items() if k != "max_output_tokens"} == {
            k: v for k, v in requests[2].items() if k != "max_output_tokens"
        }
        recorded = [json.loads(line)["event"]["content"] for line in
                    (docker_workspace / ".harness_wrapper/model_requests.jsonl").read_text().splitlines()]
        assert recorded == requests
    else:
        assert all(p["max_output_tokens"] == 262144 for p in requests)
    assert len({e.session_id for e in events if e.session_id}) == 1
    assert events[-1].content == "RECOVERY_OK\n" + "x" * 21000
    assert any(e.type == "tool_result" and e.tool_call_id == "progress_3"
               and "RETAINED_PROGRESS" in str(e.content) for e in events)
    assert agent.get_tokens().output_tokens == {"empty": 19745, "server_error": 8, "output_budget": 6}[failure]
    # Muse retries this small synthetic empty response inside its native turn;
    # terminal empty responses are covered by the completion unit tests.
    expected = {"empty": [], "server_error": ["provider_request_retry"], "output_budget": ["output_budget_reduced"]}[failure]
    assert [e.content["reason"] for e in events if e.type == "recovery"] == expected
