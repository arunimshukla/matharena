import json

import pytest

from harness_wrapper import Agent
from harness_wrapper.tools import CLIProcessError
from test_deepcode_harness import make_agent


def test_deepcode_interrupted_cli_resumes_saved_session(monkeypatch, tmp_path):
    agent = make_agent(tmp_path)
    session_id = "123e4567-e89b-42d3-a456-426614174000"
    directory = tmp_path / ".harness-home/.deepcode/projects/work"
    directory.mkdir(parents=True)
    index_path = directory / "sessions-index.json"
    transcript = directory / f"{session_id}.jsonl"
    calls = []

    def native_turn(self, message, *, resume, session_id=None, last=False):
        calls.append((message, resume, session_id, last))
        saved_id = "123e4567-e89b-42d3-a456-426614174000"
        entry = {"id": saved_id, "updateTime": "2026-09-06", "status": "interrupted"}
        saved = [
            {
                "id": "first",
                "sessionId": saved_id,
                "role": "assistant",
                "content": "Progress before interruption",
                "messageParams": None,
            }
        ]
        if len(calls) > 1:
            entry.update(status="completed", assistantReply="Finished after resume")
            saved.append(
                {
                    "id": "second",
                    "sessionId": saved_id,
                    "role": "assistant",
                    "content": "Finished after resume",
                    "messageParams": None,
                }
            )
        index_path.write_text(json.dumps({"entries": [entry]}))
        transcript.write_text("\n".join(json.dumps(item) for item in saved) + "\n")
        if len(calls) == 1:
            raise CLIProcessError(["deepcode"], 129, "interrupted")
        yield from ()

    monkeypatch.setattr(Agent, "_stream_once", native_turn)
    events = agent.run("Original task")
    assert len(calls) == 2
    assert calls[0] == ("Original task", False, None, False)
    assert calls[1] == (
        "Continue the interrupted task from where it stopped.",
        True,
        session_id,
        False,
    )
    assert agent.session_id == session_id
    assert [event.content for event in events if event.type == "message"] == [
        "Progress before interruption",
        "Finished after resume",
    ]
    assert events[-1].type == "result"


def test_deepcode_does_not_treat_missing_native_session_as_success(tmp_path):
    agent = make_agent(tmp_path)
    with pytest.raises(RuntimeError, match="without a saved native session"):
        list(agent._native_events(require_completed=True))
