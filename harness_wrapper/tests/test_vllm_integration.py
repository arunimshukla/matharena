"""Opt-in tests for a real vLLM server and pinned agent CLIs.

Start vLLM with ``meta-llama/Llama-3.2-1B-Instruct`` and set
``HARNESS_WRAPPER_VLLM_URL`` before running this module. These tests stay out of
the default suite because GPU allocation is cooperative on the target host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from harness_wrapper import Agent, Model

VLLM_URL = os.environ.get("HARNESS_WRAPPER_VLLM_URL")
MODEL = os.environ.get("HARNESS_WRAPPER_VLLM_MODEL", "llama-3.2-1b")
API_KEY = os.environ.get("HARNESS_WRAPPER_VLLM_API_KEY", "local-test-key")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not VLLM_URL, reason="HARNESS_WRAPPER_VLLM_URL is not set"),
]


def _post(path: str, payload: dict[str, object]) -> dict[str, object]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    request = Request(
        f"{VLLM_URL}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urlopen(request, timeout=120) as response:
        assert response.status == 200
        result = json.load(response)
    assert isinstance(result, dict)
    return result


def test_vllm_supports_both_wire_protocols() -> None:
    common = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply only OK"}],
        "max_tokens": 8,
    }
    assert _post("/v1/chat/completions", common)["choices"]
    assert _post("/v1/messages", common)["content"]


@pytest.mark.parametrize(
    ("harness", "executable_env", "api_type", "api_suffix"),
    [
        ("claude", "HARNESS_WRAPPER_CLAUDE", "anthropic", ""),
        ("codex", "HARNESS_WRAPPER_CODEX", "openai", "/v1"),
        ("kimi", "HARNESS_WRAPPER_KIMI", "openai", "/v1"),
    ],
)
def test_real_cli_round_trip(
    tmp_path: Path,
    harness: str,
    executable_env: str,
    api_type: str,
    api_suffix: str,
) -> None:
    executable = os.environ.get(executable_env)
    if not executable:
        pytest.skip(f"{executable_env} is not set")
    model = Model(
        MODEL,
        api_url=f"{VLLM_URL}{api_suffix}",
        api_key=API_KEY,
        api_type=api_type,
    )
    agent = Agent(
        type=harness,
        model=model,
        dir=tmp_path,
        executable=executable,
        environment={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
    )
    events = agent.run("Do not use tools. Reply with exactly OK and nothing else.")
    assert events
    assert agent.last_activity_at is not None
    assert agent.trace.path.exists()
    assert any(event.type in {"message", "result", "usage"} for event in events)
