"""Capacity recovery uses fake processes; no provider requests or real credentials."""

import pytest
from test_agents import FakeModel, FakeProcess, ProcessSequenceFactory, make_agent

from harness_wrapper.harnesses.codex_cli import CodexCLIAgent
from harness_wrapper.tools import CLIProcessError

CAPACITY_MESSAGE = (
    "Reading additional input from stdin...\n"
    "Reconnecting... 1/5 (stream disconnected before completion: "
    "stream closed before response.completed)\n"
    "Selected model is at capacity. Please try a different model."
)
SESSION_LINE = '{"type":"thread.started","thread_id":"capacity-session"}\n'
SUCCESS_LINE = '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'


@pytest.fixture
def delays(monkeypatch):
    waits = []
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", waits.append)
    return waits


@pytest.mark.parametrize("auth", ["api", "oauth"])
@pytest.mark.parametrize("message", [CAPACITY_MESSAGE, "server_overloaded", "server_is_overloaded"])
def test_capacity_retries_same_model_and_session(tmp_path, delays, auth, message):
    model = FakeModel("gpt-6-astra")
    model.auth_mode = auth
    model.provider = "openai"
    model.fallbacks = (FakeModel("must-not-use-fallback"),)

    def must_not_call(*args, **kwargs):
        pytest.fail("Server overload must not trigger quota/auth recovery")

    model.classify_cli_failure = must_not_call
    model.poll_rate_limits = must_not_call
    model.auto_wait = must_not_call
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=SESSION_LINE, stderr=message, returncode=1),
        FakeProcess(stderr=message, returncode=1),
        FakeProcess(stdout=SUCCESS_LINE),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, model=model, process_factory=factory)

    events = agent.run("Original problem")

    assert agent.model is model
    assert delays == [60, 60]
    assert events[-1].content == "done"
    recoveries = [event for event in events if event.type == "recovery"]
    assert [event.content["reason"] for event in recoveries] == ["capacity_retry"] * 2
    assert [event.content["retry_delay_seconds"] for event in recoveries] == delays
    for command, _ in factory.calls[1:]:
        assert command[:3] == [str(agent.executable), "exec", "resume"]
        assert "capacity-session" in command
        assert "gpt-6-astra" in command
        assert "Original problem" not in command
        assert command[-2:] == ["--", "-"]
    assert factory.stdin_inputs[1:] == [
        b"Continue the interrupted task from where it stopped."
    ] * 2


@pytest.mark.parametrize("budget", [0, 1, 3, 6])
def test_capacity_obeys_recovery_budget_and_minimum_delay(tmp_path, delays, budget):
    factory = ProcessSequenceFactory(
        *[
            FakeProcess(stdout=SESSION_LINE, stderr=CAPACITY_MESSAGE, returncode=1)
            for _ in range(budget + 1)
        ],
        FakeProcess(stdout=SUCCESS_LINE),
    )
    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=budget
    )

    with pytest.raises(CLIProcessError, match="at capacity"):
        agent.run("Original problem")

    assert len(factory.calls) == budget + 1
    assert delays == [60] * budget
    assert agent.trace.events[-1][0] == "error"


def test_capacity_before_session_starts_can_retry_original_request(tmp_path, delays):
    factory = ProcessSequenceFactory(
        FakeProcess(stderr=CAPACITY_MESSAGE, returncode=1),
        FakeProcess(stdout=SESSION_LINE + SUCCESS_LINE),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    assert agent.run("Original problem")[-1].content == "done"
    assert factory.calls[0][0] == factory.calls[1][0]
    assert delays == [60]


def test_capacity_with_progress_but_no_id_resumes_last_session(tmp_path, delays):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=SUCCESS_LINE, stderr=CAPACITY_MESSAGE, returncode=1),
        FakeProcess(stdout=SUCCESS_LINE),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    agent.run("Original problem")

    assert "--last" in factory.calls[1][0]
    assert "resume" in factory.calls[1][0]
    assert "Original problem" not in factory.calls[1][0]


def test_capacity_during_required_quota_resume_still_retries(tmp_path, delays, monkeypatch):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=SESSION_LINE, stderr="HTTP 429 rate limit", returncode=1),
        FakeProcess(stderr=CAPACITY_MESSAGE, returncode=1),
        FakeProcess(stderr=CAPACITY_MESSAGE, returncode=1),
        FakeProcess(stdout=SUCCESS_LINE),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    quota_recoveries = []

    def quota_recovery(error, **kwargs):
        quota_recoveries.append(error.stderr)
        assert "429" in error.stderr
        return "rate_limit_wait"

    monkeypatch.setattr(agent, "_recover_cli_failure", quota_recovery)

    assert agent.run("Original problem")[-1].content == "done"
    assert quota_recoveries == ["HTTP 429 rate limit"]
    assert delays == [60, 60, 60]  # Quota recovery also waits before retrying.
    assert all("capacity-session" in command for command, _ in factory.calls[1:])
    assert all("Original problem" not in command for command, _ in factory.calls[1:])


@pytest.mark.parametrize(
    "message",
    ["invalid API key", "context window exceeded", "disk capacity exceeded", "invalid model"],
)
def test_non_capacity_errors_also_retry(tmp_path, delays, message):
    factory = ProcessSequenceFactory(
        FakeProcess(stderr=message, returncode=1), FakeProcess(stdout=SUCCESS_LINE)
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    events = agent.run("Original problem")

    assert events[-1].content == "done"
    assert len(factory.calls) == 2
    assert delays == [60]
    assert [e.content["reason"] for e in events if e.type == "recovery"] == ["error_retry"]
