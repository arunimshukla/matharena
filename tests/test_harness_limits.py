"""Budget enforcement tests; no model queries, Docker, or long-running timers."""

import importlib.util
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from harness_wrapper import AgentEvent, TokenUsage
from matharena.json_zst import load_json_zst
from matharena.runs import Runs
from matharena.solvers import harness_solver
from matharena.solvers.run_budget import RunBudget


def budget(**limits):
    return RunBudget(limits, lambda u: u.output_tokens / 10, priced=True)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "12"])
def test_budget_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        budget(max_time_seconds=value)
    with pytest.raises(ValueError):
        budget(max_cost_usd=value)
    with pytest.raises(ValueError):
        budget(cost_limit_grace_seconds=value, cost_limit_grace_prompt="Answer now")


def test_anthropic_usage_arrives_before_cli_exit_without_double_counting():
    b = budget(max_cost_usd=10)
    b.capture_response(
        "/v1/messages",
        {
            "type": "message_start",
            "message": {
                "id": "msg1",
                "usage": {"input_tokens": 7, "cache_creation_input_tokens": 11},
            },
        },
    )
    b.capture_response(
        "/v1/messages", {"type": "message_delta", "usage": {"output_tokens": 60}}
    )
    b.capture_response(
        "/v1/messages", {"type": "message_delta", "usage": {"output_tokens": 100}}
    )
    assert b.exceeded() and b.reason == "cost_limit"
    u = b.usage(TokenUsage(input_tokens=7, output_tokens=100, cache_write_tokens=11))
    assert u == TokenUsage(input_tokens=7, output_tokens=100, cache_write_tokens=11)


def test_responses_usage_snapshots_and_internal_retries_accumulate():
    b = budget(max_cost_usd=20)
    for response_id, output in [("r1", 50), ("r1", 100), ("r1", 100), ("r2", 100)]:
        b.capture_response(
            "/v1/responses",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "usage": {"input_tokens": 5, "output_tokens": output},
                },
            },
        )
    assert b.usage() == TokenUsage(input_tokens=10, output_tokens=200)
    assert b.exceeded()


def test_google_usage_includes_reasoning():
    b = budget(max_cost_usd=10)
    b.capture_response(
        "/generateContent",
        {
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 30,
                "thoughtsTokenCount": 70,
            }
        },
    )
    assert b.usage().output_tokens == 100
    assert b.exceeded()


def test_unpriced_model_keeps_time_limit_but_disables_money_limit():
    b = RunBudget(
        {"max_time_seconds": 43200, "max_cost_usd": 100}, lambda u: 1000, priced=False
    )
    assert not b.exceeded()
    fields = b.prompt_fields()
    assert fields["run_time_limit_hours"] == "12"
    assert (
        datetime.fromisoformat(fields["run_deadline_at"])
        - datetime.fromisoformat(fields["run_started_at"])
    ).total_seconds() == 43200
    assert b.metadata()["max_cost_usd"] is None


def make_solver(tmp_path, monkeypatch, limit, *, ignore_interrupt=False):
    state = {
        "prompts": [],
        "stopped": threading.Event(),
        "interrupted": False,
        "removed": False,
    }
    config = {
        "model": "test",
        "api": "anthropic",
        "harness": "claude",
        "read_cost": 10,
        "write_cost": 50,
        "harness_config": {
            "managed_cli": False,
            "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            "time_limit_prompt": "Start: {run_started_at}. Deadline: {run_deadline_at}. {run_time_limit_hours} hours.",
            "max_time_seconds": 0.03 if limit == "time" else 43200,
            "max_cost_usd": 0.005 if limit == "cost" else 100,
        },
    }
    solver = harness_solver.HarnessSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None},
        "{problem}",
        config,
        "",
    )

    class FakeAgent:
        session_id = "session"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.usage = TokenUsage()
            state["agent"] = self

        def get_tokens(self):
            return self.usage

        def stream(self, prompt):
            state["prompts"].append(prompt)
            yield AgentEvent(type="reasoning", content="Partial reasoning")
            if limit == "cost":
                # The CLI has not emitted its final usage. API usage is enough.
                self.kwargs["on_model_response"](
                    "/v1/messages",
                    {
                        "id": "message1",
                        "usage": {"input_tokens": 3, "output_tokens": 100},
                    },
                )
            assert state["stopped"].wait(4), "watchdog failed to stop the harness"
            raise RuntimeError("CLI interrupted without a final response")

        def interrupt(self):
            state["interrupted"] = True
            if not ignore_interrupt:
                state["stopped"].set()

        def terminate(self):
            state["stopped"].set()

    def stop():
        state["removed"] = True
        state["stopped"].set()

    monkeypatch.setattr(harness_solver, "Agent", FakeAgent)
    monkeypatch.setattr(solver, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(solver, "_build_model", lambda: None)
    monkeypatch.setattr(
        solver, "_build_sandbox", lambda workspace: SimpleNamespace(stop=stop)
    )
    return solver, state


@pytest.mark.parametrize(
    ("limit", "ignore_interrupt"), [("cost", False), ("time", False), ("time", True)]
)
def test_limits_stop_and_save_partial_work_as_incorrect(
    tmp_path, monkeypatch, limit, ignore_interrupt
):
    solver, state = make_solver(
        tmp_path, monkeypatch, limit, ignore_interrupt=ignore_interrupt
    )
    started = time.monotonic()
    response = solver._solve_one(0, "Question", None, 1, 0)
    assert time.monotonic() - started < 4
    assert state["interrupted"]
    assert state["removed"] == ignore_interrupt
    assert response.detailed_cost["run_limits"]["exceeded"] == f"{limit}_limit"
    assert response.conversation[-1]["content"] == ""
    assert response.conversation[-2]["content"] == "Partial reasoning"
    assert "{run_" not in state["prompts"][0]
    assert "+00:00" in state["prompts"][0]
    if limit == "cost":
        assert response.detailed_cost["output_tokens"] == 100
        assert response.detailed_cost["cost"] == pytest.approx(0.00503)
    runs = Runs(
        "test",
        True,
        "model",
        "agent",
        {"problem_idx": 1, "problem": "Question", "answer": "42"},
        str(tmp_path),
    )
    runs.add_run(response, ("42", True, 0))
    runs.save_to_file()
    assert load_json_zst(runs.path)["correct"] == [False]
    runs.update_run_grading(0, ("42", True, 0))
    assert runs.correct == [False]
    # The limit also prevents last-chance prompts from purchasing a new budget.
    assert solver.last_chance(response) is response
    assert len(state["prompts"]) == 1


def test_judge_cannot_override_budget_failure():
    spec = importlib.util.spec_from_file_location(
        "limit_judge", Path("scripts/judge/judge.py")
    )
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    data = {
        "messages": [[], []],
        "detailed_costs": [{"run_limits": {"exceeded": "cost_limit"}}, {}],
        "judgment": [[{"points": 3, "max_points": 3}, {"points": 3, "max_points": 3}]],
    }
    judge.recompute_correct_and_pass_at_1(data)
    assert data["correct"] == [0, 1]
    assert data["pass_at_1"] == 0.5


def test_limits_are_enabled_in_both_august_competitions():
    for name in ("arxiv/august", "arxiv_false/august"):
        cfg = yaml.safe_load(Path(f"configs/competitions/{name}.yaml").read_text())
        limits = cfg["harness_config"]
        assert limits["max_time_seconds"] == 43200
        assert limits["max_cost_usd"] == 100
        assert limits["cost_limit_grace_seconds"] == 300
        assert "{cost_limit_deadline_at}" in limits["cost_limit_grace_prompt"]
        assert "{run_started_at}" in limits["time_limit_prompt"]
        assert "{run_deadline_at}" in limits["time_limit_prompt"]
        assert "{run_started_at}" not in cfg["instruction"]


def test_answer_returned_over_budget_is_preserved_but_incorrect(tmp_path, monkeypatch):
    solver, state = make_solver(tmp_path, monkeypatch, "cost")

    def run_immediately(agent, sandbox, budget, prompt):
        budget.capture_response(
            "/v1/messages",
            {
                "id": "last",
                "usage": {"input_tokens": 3, "output_tokens": 100},
            },
        )
        return [AgentEvent(type="result", content="42")]

    monkeypatch.setattr(solver, "_run_with_budget", run_immediately)
    response = solver._solve_one(0, "Question", None, 1, 0)
    assert response.conversation[-1]["content"] == "42"
    assert response.detailed_cost["run_limits"]["exceeded"] == "cost_limit"
    runs = Runs(
        "test",
        True,
        "model",
        "agent",
        {"problem_idx": 1, "problem": "Question", "answer": "42"},
        str(tmp_path),
    )
    runs.add_run(response, ("42", True, 0))
    assert runs.correct == [False]


def test_cost_grace_is_once_only_and_keeps_original_time_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("matharena.solvers.run_budget.time.monotonic", lambda: now[0])
    b = budget(
        max_cost_usd=10,
        max_time_seconds=43200,
        cost_limit_grace_seconds=300,
        cost_limit_grace_prompt="Answer in {cost_limit_grace_minutes} minutes by {cost_limit_deadline_at}",
    )
    b.usage(TokenUsage(output_tokens=100))
    assert b.exceeded()
    now[0] = 43100
    prompt = b.begin_cost_grace()
    assert "+00:00" in prompt
    assert b.cost_grace["seconds"] == 100
    assert not b.exceeded()
    assert b.begin_cost_grace() is None
    now[0] = 43200
    assert b.exceeded() and b.reason == "time_limit"


def test_time_limit_never_gets_a_cost_grace(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("matharena.solvers.run_budget.time.monotonic", lambda: now[0])
    b = budget(
        max_time_seconds=1,
        max_cost_usd=10,
        cost_limit_grace_seconds=300,
        cost_limit_grace_prompt="Answer now",
    )
    now[0] = 1
    assert b.exceeded() and b.reason == "time_limit"
    assert b.begin_cost_grace() is None


def test_cost_grace_needs_a_prompt():
    with pytest.raises(ValueError, match="requires cost_limit_grace_prompt"):
        budget(cost_limit_grace_seconds=300)


@pytest.mark.parametrize("final_kind", ["result", "message"])
def test_cost_grace_resumes_same_session_saves_prompt_and_remains_gradable(
    tmp_path, monkeypatch, final_kind
):
    solver, state = make_solver(tmp_path, monkeypatch, "cost")
    solver.harness_config.update(
        cost_limit_grace_seconds=300,
        cost_limit_grace_prompt="Budget hit: answer within {cost_limit_grace_minutes} minutes. Deadline {cost_limit_deadline_at}.",
    )
    calls = []

    def resume(self, prompt, *, session_id):
        calls.append(session_id)
        state["prompts"].append(prompt)
        assert session_id == "session"
        assert not self.kwargs["should_stop"]()
        self.kwargs["on_model_response"](
            "/v1/messages",
            {
                "id": "final",
                "usage": {"output_tokens": 20},
            },
        )
        yield AgentEvent(type=final_kind, role="assistant", content="42")

    # Install after constructing the fake agent but before starting its stream.
    original = harness_solver.Agent.__init__

    def initialize(self, **kwargs):
        original(self, **kwargs)
        workspace = kwargs["dir"] / "checkpoint.txt"
        workspace.write_text("saved proof")
        self.checkpoint = workspace

    def resume_with_files(self, prompt, *, session_id):
        assert self.checkpoint.read_text() == "saved proof"
        yield from resume(self, prompt, session_id=session_id)

    monkeypatch.setattr(harness_solver.Agent, "__init__", initialize)
    monkeypatch.setattr(
        harness_solver.Agent, "stream_resume", resume_with_files, raising=False
    )
    response = solver._solve_one(0, "Question", None, 1, 0)
    assert calls == ["session"]
    assert "within 5 minutes" in state["prompts"][1]
    assert response.conversation[-2] == {"role": "user", "content": state["prompts"][1]}
    assert response.conversation[-1]["content"] == "42"
    assert response.detailed_cost["output_tokens"] == 120
    limits = response.detailed_cost["run_limits"]
    assert limits["exceeded"] is None
    assert limits["cost_limit_grace"]["completed"] is True
    assert limits["usage_may_be_incomplete"] is True
    runs = Runs(
        "test",
        True,
        "model",
        "agent",
        {"problem_idx": 1, "problem": "Question", "answer": "42"},
        str(tmp_path),
    )
    runs.add_run(response, ("42", True, 0))
    assert runs.correct == [True]
    assert solver.last_chance(response) is response
    assert calls == ["session"]


@pytest.mark.parametrize("failure", ["timeout", "empty", "error", "no_session"])
def test_failed_final_budget_chance_is_saved_as_incorrect(
    tmp_path, monkeypatch, failure
):
    solver, state = make_solver(tmp_path, monkeypatch, "cost")
    solver.harness_config.update(
        cost_limit_grace_seconds=0.02 if failure == "timeout" else 300,
        cost_limit_grace_prompt="Answer now",
    )
    calls = []

    def resume(self, prompt, *, session_id):
        calls.append(session_id)
        if failure == "timeout":
            state["stopped"].clear()
            yield AgentEvent(type="reasoning", content="Final reasoning")
            assert state["stopped"].wait(3)
            raise RuntimeError("interrupted")
        if failure == "error":
            yield AgentEvent(type="reasoning", content="Final reasoning")
            raise RuntimeError("failed")
        yield AgentEvent(type="result", content="")

    monkeypatch.setattr(harness_solver.Agent, "stream_resume", resume, raising=False)
    if failure == "no_session":
        monkeypatch.setattr(harness_solver.Agent, "session_id", None)
    response = solver._solve_one(0, "Question", None, 1, 0)
    limits = response.detailed_cost["run_limits"]
    assert limits["exceeded"] == (
        "cost_grace_timeout" if failure == "timeout" else "cost_limit"
    )
    assert calls == ([] if failure == "no_session" else ["session"])
    runs = Runs(
        "test",
        True,
        "model",
        "agent",
        {"problem_idx": 1, "problem": "Question", "answer": "42"},
        str(tmp_path),
    )
    runs.add_run(response, ("42", True, 0))
    assert runs.correct == [False]


def test_cost_reported_at_cli_exit_also_gets_final_chance(tmp_path, monkeypatch):
    solver, state = make_solver(tmp_path, monkeypatch, "cost")
    solver.harness_config.update(
        cost_limit_grace_seconds=300, cost_limit_grace_prompt="Answer now"
    )

    def stream(self, prompt):
        self.usage = TokenUsage(output_tokens=200)
        yield AgentEvent(type="result", content="Initial answer")

    def resume(self, prompt, *, session_id):
        assert not self.kwargs["should_stop"]()
        yield AgentEvent(type="result", content="Final answer")

    monkeypatch.setattr(harness_solver.Agent, "stream", stream)
    monkeypatch.setattr(harness_solver.Agent, "stream_resume", resume, raising=False)
    response = solver._solve_one(0, "Question", None, 1, 0)
    assert response.detailed_cost["run_limits"]["cost_limit_grace"]["completed"]
    assert response.conversation[-1]["content"] == "Final answer"
