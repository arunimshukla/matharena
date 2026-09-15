"""Exercise Muse's real compaction protocol against a local fake provider."""

import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.muse_code import MuseCodeAgent
from test_harness_e2e import (
    _FakeCodexResponsesHandler,
    _TEST_CLI_RELEASES,
    _environment,
    _image_available,
    _sandbox,
    docker_workspace,
)


def _functions(tools):
    for tool in tools:
        if tool.get("type") == "namespace":
            yield from _functions(tool.get("tools", []))
        elif tool.get("type") == "function":
            yield tool


@pytest.mark.skipif(not _image_available(), reason="harness Docker image is not built")
@pytest.mark.parametrize("token_limit", [False, True])
def test_muse_native_compaction_preserves_its_tool_and_budget(docker_workspace, monkeypatch, token_limit):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)
    requests = []
    summaries = []
    main_calls = []
    limited_responses = []

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
            summary = next((t for t in _functions(payload.get("tools", [])) if t["name"] == "generate_summary"), None)
            if summary:
                summaries.append(payload)
                fields = summary["parameters"]["properties"]
                item = {
                    "type": "function_call", "name": "muse.generate_summary", "call_id": "summary1",
                    "id": "fc_summary1", "status": "completed",
                    "arguments": json.dumps({name: "- Preserve COMPACTION_CONTEXT_MARKER and finish the task." for name in fields}),
                }
                input_tokens = 500
            else:
                main_calls.append(payload)
                if not summaries and len(main_calls) < 10:
                    item = {
                        "type": "function_call", "name": "muse.bash", "call_id": f"bash{len(main_calls)}",
                        "id": f"fc_bash{len(main_calls)}", "status": "completed",
                        "arguments": json.dumps({"command": "python3 -c \"print('COMPACTION_CONTEXT_MARKER ' * 3000)\"", "description": "Save context marker", "max_output_tokens": 20000}),
                    }
                    # Force the native CLI across its hard compaction threshold.
                    input_tokens = 500000
                elif token_limit and not limited_responses:
                    item = None
                    limited_responses.append(len(requests))
                    input_tokens = 500
                else:
                    item = {
                        "type": "message", "id": "final", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "MUSE_COMPACTION_OK", "annotations": []}],
                    }
                    input_tokens = 500
            response = _FakeCodexResponsesHandler._response(f"resp_{len(requests)}", item)
            response["model"] = "muse-spark-1.3"
            response["usage"].update(input_tokens=input_tokens, total_tokens=input_tokens + 2)
            if item is None:
                response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[])
                response["usage"].update(output_tokens=262144, total_tokens=input_tokens + 262144,
                                         output_tokens_details={"reasoning_tokens": 262141})
            events = [
                {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                *([{"type": "response.output_item.done", "output_index": 0, "item": item}] if item else []),
                {"type": "response.incomplete" if item is None else "response.completed", "response": response},
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
    original_build = MuseCodeAgent.build_command

    def command(self, *args, **kwargs):
        result = original_build(self, *args, **kwargs)
        if "--" not in result:
            return result
        result[result.index("--"):result.index("--")] = ["--max-model-steps", "12", "--context-compaction-soft-threshold", "0.025", "--context-compaction-hard-threshold", "0.05"]
        return result

    monkeypatch.setattr(MuseCodeAgent, "build_command", command)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        agent = Agent(
            type="muse", model=Model(
                "muse-spark-1.3", api_url=f"http://127.0.0.1:{upstream.server_port}/v1",
                api_key="fake-key", reasoning="max",
                request_overrides={"reasoning": {"effort": "max"}, "max_output_tokens": 262144, "store": False},
            ),
            env=_sandbox(docker_workspace, "muse"), dir=docker_workspace,
            executable="/usr/bin/true", environment=_environment(),
            subagents={}, minimal_context=True, validate_version=False,
            auto_wait=False, auto_fallback=False, max_recovery_attempts=1,
        )
        def deadline(signum, frame):
            raise TimeoutError("Muse fake-provider compaction test exceeded 60 seconds")

        previous_handler = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(60)
        try:
            events = agent.run("Remember COMPACTION_CONTEXT_MARKER, use the shell to print it, and finish.")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert summaries, [(len(p.get("tools", [])), p.get("max_output_tokens")) for p in requests]
    assert all(p.get("max_output_tokens") != 262144 for p in summaries)
    assert all(p.get("reasoning", {}).get("effort") != "max" for p in summaries)
    assert all(p["max_output_tokens"] == 262144 and p["reasoning"]["effort"] == "max" for p in main_calls)
    assert any(e.type == "result" and e.content == "MUSE_COMPACTION_OK" for e in events)
    assert not any(e.type == "error" for e in events)
    recoveries = [e for e in events if e.type == "recovery"]
    # Depending on when asynchronous compaction installs, Muse may continue
    # internally or end the native turn and let the wrapper resume it.
    assert [e.content["reason"] for e in recoveries] in (
        [[], ["token_limit_resume"]] if token_limit else [[]]
    )
    if token_limit:
        assert len({e.session_id for e in events if e.session_id}) == 1
        if recoveries:
            assert all("Continue the interrupted task" not in json.dumps(p["input"]) for p in main_calls)
            assert any("[Previous turn ended without an assistant reply.]" in json.dumps(p["input"])
                       for p in main_calls)
    native_compactions = []
    for session_file in docker_workspace.glob(".harness-home/.local/share/muse/sessions/*/*/*/*/session.jsonl"):
        for line in session_file.read_text().splitlines():
            native = (json.loads(line).get("payload") or {}).get("event") or {}
            if str(native.get("kind", "")).startswith("context_compaction_"):
                native_compactions.append(native)
    assert any(e["kind"] == "context_compaction_installed" for e in native_compactions), native_compactions
    assert not any(e["kind"] == "context_compaction_fallback" and e.get("reason") == "summary_failed" for e in native_compactions), native_compactions
    assert not any(e.type == "tool_call" and e.tool_name == "generate_summary" for e in events)
    assert agent.get_tokens().output_tokens == 2 * (len(requests) - len(limited_responses)) + 262144 * len(limited_responses)
