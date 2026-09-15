"""MSP continuation must fail closed on a missing or mismatched native session."""

import io
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from test_muse_code import _agent

from harness_wrapper import CLIProcessError
from harness_wrapper.harnesses.muse_code.msp import _command_id, resume_session


def test_msp_command_ids_are_unique_uuid7():
    ids = {_command_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(UUID(value).version == 7 for value in ids)


@pytest.mark.parametrize(
    "problem", ["missing", "wrong_session", "wrong_model", "busy", "client_input"],
)
def test_msp_never_starts_a_fresh_or_mismatched_conversation(tmp_path, problem):
    agent = _agent(tmp_path)
    session = {"sessionId": "retained", "modelId": "spark", "providerId": "meta", "status": "idle"}
    if problem == "wrong_session":
        session["sessionId"] = "different"
    if problem == "wrong_model":
        session["modelId"] = "different-model"
    if problem == "busy":
        session["status"] = "running"
    replies = [
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 2, "result": {"session": session}},
    ]
    if problem == "missing":
        replies[1] = {"jsonrpc": "2.0", "id": 2, "error": {"message": "Session not found"}}
    if problem == "client_input":
        replies[1] = {
            "jsonrpc": "2.0", "id": "approval1", "method": "approval/request", "params": {},
        }

    class Input(io.StringIO):
        sent = ""

        def close(self):
            self.sent = self.getvalue()
            super().close()

    stdin = Input()
    process = SimpleNamespace(
        stdin=stdin,
        stdout=io.StringIO("".join(json.dumps(r) + "\n" for r in replies)),
        stderr=io.StringIO(""), poll=lambda: None, wait=lambda **kwargs: 0,
    )
    agent._process_factory = lambda *args, **kwargs: process
    with pytest.raises(CLIProcessError):
        list(resume_session(agent, "Continue", "retained"))
    commands = [json.loads(line) for line in stdin.sent.splitlines()]
    assert [r["method"] for r in commands] == ["initialize", "initialized", "session/resume"]
    assert agent._process is None


@pytest.mark.parametrize("message", ["Continue", ""])
def test_msp_requires_matching_turn_terminal_and_uses_full_provider_text(tmp_path, message):
    agent = _agent(tmp_path)
    agent._last_solver_final_text = "complete proof " * 3000
    replies = [
        {"id": 1, "result": {}},
        {"id": 2, "result": {"session": {
            "sessionId": "retained", "modelId": "spark", "providerId": "meta", "status": "idle",
        }}},
        {"id": 3, "result": {"disposition": "started", "turnId": "turn-1"}},
        {"method": "item/completed", "params": {"item": {
            "kind": "agentMessage", "text": "truncated view", "truncated": True,
        }}},
        {"method": "turn/completed", "params": {
            "sessionId": "retained", "turnId": "old-turn", "terminal": "completed",
        }},
        {"method": "turn/completed", "params": {
            "sessionId": "retained", "turnId": "turn-1", "terminal": "completed",
        }},
    ]
    class Input(io.StringIO):
        sent = ""

        def close(self):
            self.sent = self.getvalue()
            super().close()

    stdin = Input()
    process = SimpleNamespace(
        stdin=stdin,
        stdout=io.StringIO("".join(json.dumps({"jsonrpc": "2.0", **r}) + "\n" for r in replies)),
        stderr=io.StringIO(""), poll=lambda: None, wait=lambda **kwargs: 0,
    )
    agent._process_factory = lambda *args, **kwargs: process
    events = list(resume_session(agent, message, "retained"))
    commands = [json.loads(line) for line in stdin.sent.splitlines()]
    started = next(c for c in commands if c["method"] == "turn/start")
    assert started["params"]["input"] == [{"type": "text", "text": message}]
    assert [e.content for e in events if e.type == "result"] == [agent._last_solver_final_text]
    assert agent._process is None
