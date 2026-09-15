"""Opt-in smoke tests for real plan-backed OAuth providers.

Set the provider-specific model environment variable to enable a case. Existing
credentials must already be valid; these tests never start an interactive login.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness_wrapper import Agent, Model
from harness_wrapper.models.oauth import (
    anthropic_oauth_config,
    kimi_oauth_config,
    openai_oauth_config,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("provider", "harness", "model_env"),
    [
        ("openai", "codex", "HARNESS_WRAPPER_OAUTH_OPENAI_MODEL"),
        ("anthropic", "claude", "HARNESS_WRAPPER_OAUTH_ANTHROPIC_MODEL"),
        ("kimi", "kimi", "HARNESS_WRAPPER_OAUTH_KIMI_MODEL"),
    ],
)
def test_live_oauth_poll_and_agent_round_trip(
    tmp_path: Path,
    provider: str,
    harness: str,
    model_env: str,
) -> None:
    model_name = os.environ.get(model_env)
    if not model_name:
        pytest.skip(f"{model_env} is not set")
    configs = {
        "openai": openai_oauth_config(auto_login=False, auto_relogin=False),
        "anthropic": anthropic_oauth_config(auto_login=False, auto_relogin=False),
        "kimi": kimi_oauth_config(auto_login=False, auto_relogin=False),
    }
    model = Model(
        model_name,
        oauth=provider,
        oauth_config=configs[provider],
    )

    snapshot = model.poll_rate_limits(timeout=30)
    assert snapshot.limits
    if snapshot.limited:
        pytest.skip(f"{provider} plan is currently rate limited")

    agent = Agent(
        type=harness,
        model=model,
        dir=tmp_path,
        subagents={},
        auto_wait=False,
        max_recovery_attempts=0,
    )
    events = agent.run("Do not use tools. Reply with exactly OK and nothing else.")

    assert any(event.type in {"message", "result"} for event in events)
