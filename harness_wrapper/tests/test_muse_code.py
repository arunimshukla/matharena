import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.muse_code import MuseCodeAgent
from harness_wrapper.installation import ensure_cli, install_clis, resolve_cli_release
from harness_wrapper.models.request_capture import RequestCaptureProxy


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


def _agent(tmp_path, *, request_overrides=None, headers=None):
    executable = tmp_path / "muse"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    agent = Agent(
        type="muse",
        model=Model(
            "spark", api_url="https://api.meta.ai/v1", api_key="placeholder", reasoning="xhigh",
            request_overrides=request_overrides, headers=headers or {},
        ),
        dir=tmp_path,
        executable=executable,
        subagents={},
        minimal_context=True,
    )
    agent._resume_catalog = {"object": "list", "data": [{"id": "spark"}]}
    return agent


def _event(kind, **payload):
    return {"stream": {"kind": "session", "id": "s1"}, "payload_type": kind, "payload": payload}


def test_command_settings_and_session_resume(tmp_path):
    agent = _agent(tmp_path)
    assert isinstance(agent, MuseCodeAgent)
    agent._request_capture_url = "http://proxy/secret/v1"
    env = agent.model_environment()
    settings = json.loads((tmp_path / ".harness-home/.config/muse/settings.json").read_text())
    assert settings["endpoint_transport"] == {
        "base_url": "http://proxy/secret/v1",
        "auth": "bearer",
    }
    assert settings["run"]["reminder_roster"] == []
    assert env["META_API_KEY"] == "placeholder"
    assert env["MUSE_EXPERIMENTAL_FIRST_TURN_MINIMAL_EFFORT"] == "0"
    assert env["TBH_STREAM_FIRST_EVENT_TIMEOUT_SECS"] == "28800"
    assert env["TBH_STREAM_IDLE_TIMEOUT_SECS"] == "28800"
    command = agent.build_command("prompt")
    assert command[command.index("--reasoning-effort") + 1] == "xhigh"
    assert "--disable-web-tools" in command
    assert "--disable-sandbox" not in command
    agent.env = SimpleNamespace(enabled=True)
    assert "--disable-sandbox" in agent.build_command("prompt")
    resumed = agent.build_command("follow-up", resume=True, session_id="s1")
    assert resumed == [agent.executable, "serve", "--disable-sandbox"]
    with pytest.raises(ValueError):
        agent.build_command("follow-up", resume=True)


def test_only_local_tools_reach_provider_on_every_request(tmp_path):
    agent = _agent(tmp_path)
    payload = {
        "model": "wrong",
        "tools": [
            {
                "type": "namespace",
                "name": "muse",
                "tools": [
                    {"type": "function", "name": name}
                    for name in (
                        "bash",
                        "read_file",
                        "subagent_spawn",
                        "read_skill",
                        "web_search",
                        "workflow",
                    )
                ],
            },
            {"type": "web_search"},
        ],
    }
    agent._filter_request("/v1/responses", payload)
    assert payload["model"] == "spark"
    assert [t["name"] for t in payload["tools"][0]["tools"]] == ["bash", "read_file"]
    with pytest.raises(ValueError):
        agent._filter_request("/v1/other", payload)


def test_tool_calls_usage_and_final_answer_are_recorded_once(tmp_path):
    agent = _agent(tmp_path)
    agent._capture_model_response(
        "/v1/responses",
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": "muse.bash",
                "arguments": '{"command":"echo 323"}',
            },
        },
    )
    events = agent.normalize_event(
        _event(
            "tool.result", call_id="call1", text="323\n", correlation_facts={"tool_name": "bash"}
        )
    )
    assert [e.type for e in events] == ["tool_call", "tool_result"]
    assert events[0].content == {"command": "echo 323"}
    assert events[1].tool_call_id == "call1"
    completion = {
        "type": "response.completed",
        "response": {
            "id": "r1",
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 250,
                "total_tokens": 350,
                "output_tokens_details": {"reasoning_tokens": 220},
            },
        },
    }
    agent._capture_model_response("/v1/responses", completion)
    agent._capture_model_response("/v1/responses", completion)
    assert agent.get_tokens().input_tokens == 100
    assert agent.get_tokens().cache_read_tokens == 40
    assert agent.get_tokens().output_tokens == 250  # Includes reasoning already.
    assert agent.normalize_event(_event("run.output.delta", text="32"))[0].type == "event"
    final = agent.normalize_event(_event("run.terminal.completed", text="323"))[0]
    assert (final.type, final.content, final.session_id) == ("result", "323", "s1")
    assert (
        agent.normalize_event(_event("run.terminal.failed", reason="provider error"))[0].type
        == "error"
    )
    # A later failure must not discard already reported tokens.
    assert agent.get_tokens().output_tokens == 250


def test_static_catalog_does_not_open_an_unauthenticated_upstream_route():
    with RequestCaptureProxy(
        "http://127.0.0.1:1/v1",
        lambda *args: None,
        static_get_responses={"/muse-code/models": {"data": []}},
    ) as proxy:
        parts = urlsplit(proxy.base_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with urllib.request.urlopen(origin + "/muse-code/models") as response:
            assert json.load(response) == {"data": []}
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(origin + "/v1/responses")
        assert exc.value.code == 404


def test_muse_release_installation_and_full_build_validation(monkeypatch, tmp_path):
    version = "1.0.3-R2198.1"
    monkeypatch.setattr(
        "harness_wrapper.installation._fetch_json", lambda url: {"version": version}
    )
    assert resolve_cli_release("muse-code").version == version
    urls = install_clis(["muse-code"], versions={"muse-code": version}, dry_run=True)
    assert "version=1.0.3-R2198.1&file=muse-" in urls[0]
    prefix = tmp_path / "muse-code" / version / "bin"
    prefix.mkdir(parents=True)
    binary = prefix / "muse"
    binary.write_text(f'#!/bin/sh\nprintf "Muse Code 1.0.3 ({version})\\n"\n')
    binary.chmod(0o755)
    assert ensure_cli("muse-code", version, cache_root=tmp_path).executable == binary


def test_partial_text_and_zero_exit_do_not_count_as_completion(tmp_path, monkeypatch):
    from harness_wrapper import AgentEvent, CLIProcessError

    agent = _agent(tmp_path)

    def partial_stream(self, *args, **kwargs):
        yield AgentEvent(type="message", content="I will calculate it.", role="assistant")
        yield AgentEvent(type="error", content="stream interrupted")

    monkeypatch.setattr(Agent, "_stream_once", partial_stream)
    with pytest.raises(CLIProcessError, match="without a completed final answer"):
        list(agent._stream_once("solve", resume=False))


@pytest.mark.parametrize("namespaced", [False, True])
@pytest.mark.parametrize("minimal_context", [False, True])
def test_compaction_preserves_native_schema_prompt_and_budget(tmp_path, namespaced, minimal_context):
    agent = _agent(tmp_path, request_overrides={
        "reasoning": {"effort": "max"}, "max_output_tokens": 262144,
        "temperature": 0.9, "instructions": "Solve the problem.",
        "tools": [{"type": "web_search"}], "tool_choice": "auto",
        "store": False, "include": ["reasoning.encrypted_content"],
    })
    agent.minimal_context = minimal_context
    summary = {"type": "function", "name": "generate_summary", "parameters": {
        "type": "object", "properties": {"current_state": {"type": "string"}},
        "required": ["current_state"], "additionalProperties": False,
    }}
    tools = [summary, {"type": "function", "name": "bash"}, {"type": "web_search"}]
    if namespaced:
        tools = [{"type": "namespace", "name": "summary", "tools": tools}]
    payload = {
        "model": "spark", "tools": tools, "max_output_tokens": 4096,
        "reasoning": {"effort": "low"}, "instructions": "Call generate_summary to summarize.",
        "tool_choice": "required", "parallel_tool_calls": False,
        "input": [{"role": "user", "content": "<summary-request>context</summary-request>"}],
    }
    agent._filter_request("/v1/responses", payload)
    expected_tools = [{"type": "namespace", "name": "summary", "tools": [summary]}] if namespaced else [summary]
    assert payload["tools"] == expected_tools
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_output_tokens"] == 4096
    assert payload["instructions"] == "Call generate_summary to summarize."
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert "temperature" not in payload
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_solver_overrides_apply_without_inventing_missing_compaction_settings(tmp_path):
    agent = _agent(tmp_path, request_overrides={
        "reasoning": {"effort": "max"}, "max_output_tokens": 262144,
    })
    summary_payload = {"tools": [{"type": "function", "name": "generate_summary"}]}
    agent._filter_request("/v1/responses", summary_payload)
    assert "reasoning" not in summary_payload
    assert "max_output_tokens" not in summary_payload
    assert "instructions" not in summary_payload
    # A prompt mentioning compaction cannot bypass the solver tool filter.
    solver_payload = {"input": "<summary-request>", "tools": [
        {"type": "function", "name": "bash"}, {"type": "web_search"},
    ]}
    agent._filter_request("/v1/responses", solver_payload)
    assert solver_payload["tools"] == [{"type": "function", "name": "bash"}]
    assert solver_payload["reasoning"] == {"effort": "max"}
    assert solver_payload["max_output_tokens"] == 262144


def _response_event(response_id, *, limited=False, tokens=12):
    return {
        "type": "response.incomplete" if limited else "response.completed",
        "response": {"id": response_id, "usage": {"input_tokens": 10, "output_tokens": tokens},
                     "incomplete_details": {"reason": "max_output_tokens"} if limited else None},
    }


def test_compaction_usage_is_counted_without_exposing_internal_tool_calls(tmp_path):
    from threading import Thread

    agent = _agent(tmp_path)
    agent._capture_model_response("/v1/responses", _response_event("solve", limited=True))

    def summarize():
        agent._filter_request("/v1/responses", {"tools": [{"type": "function", "name": "generate_summary"}]})
        agent._capture_model_response("/v1/responses", {
            "type": "response.output_item.done", "item": {
                "type": "function_call", "name": "generate_summary", "id": "s1",
                "call_id": "summary-call", "arguments": '{"current_state":"in progress"}',
            },
        })
        agent._capture_model_response("/v1/responses", _response_event("summary", tokens=3))
    thread = Thread(target=summarize)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert agent._last_solver_response_limited is True
    assert agent.get_tokens().output_tokens == 15
    assert agent._drain_provider_events() == []
    # A successful main turn clears the token-limit condition.
    agent._capture_model_response("/v1/responses", _response_event("solve-2"))
    assert agent._last_solver_response_limited is False


@pytest.mark.parametrize("partial_text", ["", "A partial proof"])
def test_token_limit_resumes_same_session_and_preserves_usage(tmp_path, monkeypatch, partial_text):
    from harness_wrapper import AgentEvent

    agent = _agent(tmp_path)
    calls = []
    (tmp_path / "checkpoint.txt").write_text("saved progress")

    def stream(self, message, *, resume, session_id, last):
        calls.append((message, resume, session_id, last))
        session = AgentEvent(type="session", session_id="native-session", content={})
        self._observe(session)
        yield session
        if len(calls) == 1:
            self._capture_model_response("/v1/responses", _response_event("r1", limited=True, tokens=262144))
            result = AgentEvent(type="result", content=partial_text, session_id="native-session")
        else:
            assert (tmp_path / "checkpoint.txt").read_text() == "saved progress"
            self._capture_model_response("/v1/responses", _response_event("r2", tokens=100))
            result = AgentEvent(type="result", content="done", session_id="native-session")
        self._observe(result)
        yield result

    monkeypatch.setattr(Agent, "_stream_once", stream)
    monkeypatch.setattr(
        "harness_wrapper.harnesses.muse_code.msp.resume_session",
        lambda self, message, session_id: stream(
            self, message, resume=True, session_id=session_id, last=False,
        ),
    )
    events = agent.run("Solve the problem.")
    assert calls[0][1:] == (False, None, False)
    assert calls[1][1:] == (True, "native-session", False)
    assert calls[1][0] == ""
    assert agent.session_id == "native-session"
    assert agent.get_tokens().output_tokens == 262244
    assert [e.content["reason"] for e in events if e.type == "recovery"] == ["token_limit_resume"]
    assert events[-1].content == "done"


@pytest.mark.parametrize("max_recoveries", [0, 1, 3])
def test_token_limit_recovery_is_bounded_and_never_starts_fresh(tmp_path, monkeypatch, max_recoveries):
    from harness_wrapper import AgentEvent

    agent = _agent(tmp_path)
    agent.max_recovery_attempts = max_recoveries
    calls = []

    def stream(self, message, *, resume, session_id, last):
        calls.append((resume, session_id))
        session = AgentEvent(type="session", session_id="native-session", content={})
        self._observe(session)
        yield session
        self._capture_model_response("/v1/responses", _response_event(str(len(calls)), limited=True))
        result = AgentEvent(type="result", content="", session_id="native-session")
        self._observe(result)
        yield result

    monkeypatch.setattr(Agent, "_stream_once", stream)
    monkeypatch.setattr(
        "harness_wrapper.harnesses.muse_code.msp.resume_session",
        lambda self, message, session_id: stream(
            self, message, resume=True, session_id=session_id, last=False,
        ),
    )
    events = agent.run("Solve the problem.")
    expected = [(False, None)] + [(True, "native-session")] * max_recoveries
    assert calls == expected
    assert agent.get_tokens().output_tokens == 12 * len(expected)
    assert events[-1].type == "result" and events[-1].content == ""
    assert events[-1].raw["reason"] == "max_output_tokens"
    assert len([e for e in events if e.type == "recovery"]) == max_recoveries
    # A separate fresh attempt gets its own configured continuation allowance.
    agent.run("A new problem.")
    assert calls == expected + expected


@pytest.mark.parametrize("answer", ["", " ", "I cannot solve this problem."])
def test_completed_turn_is_not_reprompted(tmp_path, monkeypatch, answer):
    from harness_wrapper import AgentEvent

    agent = _agent(tmp_path)
    calls = []

    def stream(self, message, **kwargs):
        calls.append(message)
        self._capture_model_response("/v1/responses", _response_event("completed", tokens=29))
        result = AgentEvent(type="result", content=answer, session_id="native-session")
        self._observe(result)
        yield result

    monkeypatch.setattr(Agent, "_stream_once", stream)
    events = agent.run("Solve the problem.")
    assert calls == ["Solve the problem."]
    assert events[-1].content == answer
    assert agent.get_tokens().output_tokens == 29
    assert not any(e.type == "recovery" for e in events)


@pytest.mark.parametrize("detail", [
    "invalid API key", "HTTP 401 Unauthorized", "context window exceeded", "invalid model",
    "session logging unavailable: non-monotonic sequence", "tool failed",
    "server_error: The model failed to generate a response.",
])
def test_native_failures_never_trigger_server_error_turn_resume(tmp_path, detail):
    from harness_wrapper import AgentEvent, CLIProcessError

    agent = _agent(tmp_path)
    agent._observe(AgentEvent(type="session", session_id="native-session", content={}))
    assert agent._recover_cli_failure(CLIProcessError(["muse"], 1, detail), tried_models=set()) is None


@pytest.mark.parametrize("authorization_header", [None, "Authorization", "authorization", "AUTHORIZATION"])
def test_resume_catalog_is_cached_and_excludes_other_models(tmp_path, monkeypatch, authorization_header):
    import io

    headers = {"X-Client": "catalog-test"}
    if authorization_header is not None:
        headers[authorization_header] = "Bearer provider-key"
    agent = _agent(tmp_path, headers=headers)
    agent._resume_catalog = None
    requests = []

    def open_catalog(request, **kwargs):
        requests.append(request)
        return io.StringIO(json.dumps({"object": "list", "data": [
            {"id": "spark-contributor"},
            {"id": "spark", "metadata": {"muse-code": {"release_date": "2026-09-02"}}},
        ]}))

    monkeypatch.setattr("harness_wrapper.harnesses.muse_code.adapter.urllib.request.urlopen", open_catalog)
    first = agent._resume_model_catalog()
    assert [r["id"] for r in first["data"]] == ["spark"]
    assert agent._resume_model_catalog() is first
    assert len(requests) == 1
    assert requests[0].full_url == "https://api.meta.ai/muse-code/models"
    expected_auth = "Bearer provider-key" if authorization_header is not None else "Bearer placeholder"
    assert requests[0].get_header("Authorization") == expected_auth
    assert requests[0].get_header("X-client") == "catalog-test"
    assert agent.model_environment()["META_API_KEY"] == "placeholder"


@pytest.mark.parametrize("rows", [[], [{"id": "other-model"}], [{"id": "spark"}, {"id": "spark"}]])
def test_resume_catalog_refuses_missing_or_ambiguous_model(tmp_path, monkeypatch, rows):
    import io

    from harness_wrapper import CLIProcessError

    agent = _agent(tmp_path)
    agent._resume_catalog = None
    monkeypatch.setattr(
        "harness_wrapper.harnesses.muse_code.adapter.urllib.request.urlopen",
        lambda *args, **kwargs: io.StringIO(json.dumps({"data": rows})),
    )
    with pytest.raises(CLIProcessError, match="Could not load the configured model"):
        agent._resume_model_catalog()
    assert agent._resume_catalog is None


def test_token_cutoff_after_capacity_retry_is_saved_empty(tmp_path, monkeypatch):
    from harness_wrapper import AgentEvent
    from harness_wrapper.tools import CLIProcessError

    agent = _agent(tmp_path)
    agent.max_recovery_attempts = 1
    calls = []
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)

    def stream(self, message, *, resume, session_id, last):
        calls.append(resume)
        if len(calls) == 1:
            raise CLIProcessError(["muse"], 1, "Selected model is at capacity")
        session = AgentEvent(type="session", session_id="native-session", content={})
        self._observe(session)
        yield session
        self._capture_model_response("/v1/responses", _response_event("limited", limited=True))
        result = AgentEvent(type="result", content="partial", session_id="native-session")
        self._observe(result)
        yield result

    monkeypatch.setattr(Agent, "_stream_once", stream)
    events = agent.run("Solve")
    assert calls == [False, False]
    assert events[-1].type == "result" and events[-1].content == ""
    assert events[-1].raw["reason"] == "max_output_tokens"
    assert agent.get_tokens().output_tokens == 12
