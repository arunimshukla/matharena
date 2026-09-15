"""Missing terminal responses remain pending; completed empty answers are retained."""

from types import SimpleNamespace

import pytest

from harness_wrapper import AgentEvent
from matharena.solvers.harness_solver import HarnessSolver


@pytest.mark.parametrize(
    "events",
    [
        [],
        [
            AgentEvent(
                type="tool_call", tool_name="run_command", content={"command": "ls"}
            )
        ],
        [AgentEvent(type="tool_result", content="Not a submitted answer")],
        [AgentEvent(type="reasoning", content="Still thinking")],
        [AgentEvent(type="message", role="assistant", content=" ")],
    ],
)
def test_solver_rejects_attempt_without_final_response(tmp_path, monkeypatch, events):
    solver = make_solver(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError, match="refusing to save this attempt"):
        solver._solve_one(0, "Problem", None, 4, 0)


def make_solver(tmp_path, monkeypatch, events, harness="gravity"):
    config = {
        "model": "gemini-test",
        "api": "google",
        "harness": harness,
        "harness_config": {
            "managed_cli": False,
            "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
        },
    }
    solver = HarnessSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None},
        "{problem}",
        config,
        "",
    )
    fake_agent = SimpleNamespace(run=lambda prompt: events, session_id="session-1")
    monkeypatch.setattr(
        "matharena.solvers.harness_solver.Agent", lambda **kwargs: fake_agent
    )
    monkeypatch.setattr(solver, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(solver, "_build_model", lambda: None)
    monkeypatch.setattr(solver, "_build_sandbox", lambda workspace: None)
    monkeypatch.setattr(solver, "_detailed_cost", lambda agent: {})
    return solver


def test_solver_keeps_complete_canonical_answer(tmp_path, monkeypatch):
    solver = make_solver(
        tmp_path,
        monkeypatch,
        [
            AgentEvent(type="message", role="assistant", content="Last chunk."),
            AgentEvent(type="result", content="First chunk. Last chunk."),
        ],
    )
    response = solver._solve_one(0, "Problem", None, 34, 1)
    assert response.conversation[-1]["content"] == "First chunk. Last chunk."


def test_solver_saves_completed_empty_response(tmp_path, monkeypatch):
    solver = make_solver(tmp_path, monkeypatch, [AgentEvent(type="result", content="")])
    response = solver._solve_one(0, "Problem", None, 55, 0)
    assert response.conversation[-1]["content"] == ""


HARNESS_NAMES = [
    "codex", "claude", "kimi", "qwen", "gravity", "opencode", "deepcode",
    "muse", "muse-code", "muse-cli",
]


@pytest.mark.parametrize("harness", HARNESS_NAMES)
@pytest.mark.parametrize("answer", ["", " \n", None])
def test_empty_completion_is_saved_without_reprompt(tmp_path, monkeypatch, harness, answer):
    solver = make_solver(tmp_path, monkeypatch, [
        AgentEvent(type="message", role="assistant", content="A partial proof"),
        AgentEvent(type="result", content=answer),
    ], harness=harness)
    monkeypatch.setattr(solver, "_detailed_cost", lambda agent: {"output_tokens": 123})

    response = solver._solve_one(0, "Problem", None, 17, 0)

    assert response.conversation[-1] == {"role": "assistant", "type": "response", "content": ""}
    assert response.history[0]["empty_completion"] is True
    assert response.detailed_cost["output_tokens"] == 123
    assert solver.last_chance(response) is response
    assert len(response.history) == 1


@pytest.mark.parametrize("harness", HARNESS_NAMES)
@pytest.mark.parametrize("solution", [None, "", "import Mathlib\n-- saved proof\n"])
def test_empty_completion_still_checks_and_submits_lean_file(
    tmp_path, monkeypatch, harness, solution
):
    solver = make_solver(tmp_path, monkeypatch, [
        AgentEvent(type="message", role="assistant", content="A partial proof"),
        AgentEvent(type="result", content=""),
    ], harness=harness)
    solver.is_lean_comp = True
    solver.run_check_after_harness = True

    def prepare(workspace, payload):
        workspace.mkdir(parents=True)
        if solution is not None:
            (workspace / "Solution.lean").write_text(solution)

    checks = []

    def check(command, **kwargs):
        checks.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(solver, "_prepare_workspace", prepare)
    monkeypatch.setattr(solver, "_build_sandbox", lambda workspace: SimpleNamespace(run=check))

    response = solver._solve_one(0, "Problem", None, 17, 0)

    expected = f"```lean\n{solution.rstrip()}\n```" if solution else ""
    assert response.conversation[-1]["content"] == expected
    assert checks == [["./check.sh"]]
    assert response.history[0]["check_returncode"] == 0
    assert solver.last_chance(response) is response
    assert len(response.history) == 1


@pytest.mark.parametrize("answer", [None, "", "42"])
@pytest.mark.parametrize("with_usage", [False, True])
def test_native_codex_completion_reaches_saved_response(tmp_path, monkeypatch, answer, with_usage):
    from harness_wrapper.harnesses.codex_cli import CodexCLIAgent

    # Exercise the real adapter's native-event conversion, rather than assuming
    # the CLI already emits the shared result event.
    adapter = CodexCLIAgent.__new__(CodexCLIAgent)
    native = [{"type": "thread.started", "thread_id": "codex-test"}]
    if answer is not None:
        native.append({"type": "item.completed", "item": {
            "type": "agent_message", "text": answer,
        }})
    native.append({"type": "turn.completed", **(
        {"usage": {"input_tokens": 20, "output_tokens": 100}} if with_usage else {}
    )})
    events = [event for raw in native for event in adapter.normalize_event(raw)]
    solver = make_solver(tmp_path, monkeypatch, events, harness="codex")

    response = solver._solve_one(0, "Problem", None, 1, 0)

    assert response.conversation[-1]["content"] == (answer or "")
    assert len(response.conversation) == 2  # No duplicated nonempty final answer.
    if not answer:
        assert response.history[0]["empty_completion"] is True
        assert solver.last_chance(response) is response
