"""No model/network calls: verify competition policy and the Codex command boundary."""

from pathlib import Path

import pytest
import yaml

from harness_wrapper import Agent, Model
from matharena.runner import Runner
from matharena.solvers.harness_solver import HarnessSolver
from matharena.solvers.pure_model_solver import PureModelSolver


@pytest.mark.parametrize("competition", ("arxiv/june", "arxiv_false/june"))
@pytest.mark.parametrize("allow_harness", (False, True))
@pytest.mark.parametrize("prompt_variant", ("default", "harness", "custom"))
def test_astra_runner_enforces_competition_policy(
    tmp_path, monkeypatch, competition, allow_harness, prompt_variant
):
    root = Path.cwd()
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-placeholder")
    runner = object.__new__(Runner)
    runner.comp_name = competition
    runner.competition_config = yaml.safe_load(
        (root / "configs/competitions" / f"{competition}.yaml").read_text()
    )
    if allow_harness:
        runner.competition_config["allow_harness"] = True
    else:
        assert "allow_harness" not in runner.competition_config
    # Even explicit model/competition settings cannot bypass allow_harness.
    runner.competition_config["harness_config"] = {
        "tools_enabled": True,
        "minimal_context": False,
    }
    expected_template = runner.competition_config["instruction"]
    if prompt_variant != "default":
        runner.competition_config["harness_instruction"] = "Harness prompt: {problem}"
        if allow_harness:
            expected_template = runner.competition_config["harness_instruction"]
    if prompt_variant == "custom":
        expected_template = "Custom prompt: {problem}"
        load_config = runner.load_solver_config

        def load_custom_config(path):
            config = load_config(path)
            config["model_config"]["custom_instructions"] = {competition: expected_template}
            return config

        monkeypatch.setattr(runner, "load_solver_config", load_custom_config)
    if "{problem}" not in expected_template:
        expected_template += "\n\n{problem}"
    runner.solver_configs_dir = str(root / "configs/models")
    runner.base_output_dir = str(tmp_path / "outputs")
    runner.runs_per_problem = 1
    runner.redo_all = False
    runner.options = None
    runner.is_lean_comp = False
    runner.is_auto_graded_comp = competition.startswith("arxiv/")
    runner.problems = [{"problem_idx": 1, "problem": "Compute 6 * 7.", "answer": "42"}]
    monkeypatch.chdir(tmp_path)

    prepared = runner.prepare_run("openai/gpt-6-astra", set_request_metadata=False)

    solver = prepared["solver"]
    assert solver.default_prompt_template == expected_template
    if allow_harness:
        assert isinstance(solver, HarnessSolver)
        assert solver.tools_enabled is True
        assert solver.config["model"] == "gpt-6-astra"
        assert solver.config["reasoning_effort"] == "max"
        assert solver.harness_config["auth"] == "subscription"
        assert solver._harness_cli is None  # Preparation does not execute a CLI/model.
        assert solver.build_prompt("Compute 6 * 7.") == expected_template.format(
            problem="Compute 6 * 7."
        )
    else:
        assert isinstance(solver, PureModelSolver)
        assert solver.client.model == "gpt-6-astra"
        assert solver.client.api_key == "test-api-placeholder"
        assert solver.client.kwargs["reasoning_effort"] == "max"
        assert not {"harness", "harness_version", "harness_config"} & solver.client.kwargs.keys()


def test_astra_runner_prepares_regular_june_lean_with_tools(tmp_path, monkeypatch):
    root = Path.cwd()
    runner = object.__new__(Runner)
    runner.comp_name = "arxivlean/june"
    runner.competition_config = yaml.safe_load(
        (root / "configs/competitions/arxivlean/june.yaml").read_text()
    )
    runner.solver_configs_dir = str(root / "configs/models")
    runner.base_output_dir = str(tmp_path / "outputs")
    runner.runs_per_problem = 1
    runner.redo_all = False
    runner.options = None
    runner.is_lean_comp = True
    runner.is_auto_graded_comp = True
    runner.problems = [
        {
            "problem_idx": 1,
            "problem": "True is true.",
            "formal_statement": "theorem lean_smoke : True := by trivial",
            "answer": "",
        }
    ]
    monkeypatch.chdir(tmp_path)

    prepared = runner.prepare_run("openai/gpt-6-astra", set_request_metadata=False)

    solver = prepared["solver"]
    assert isinstance(solver, HarnessSolver)
    assert solver.tools_enabled is True
    assert solver.is_lean_comp is True
    assert solver.run_check_after_harness is True
    assert solver.lean_environment == "lean-4.31.0"
    assert solver.toolchain_dir == "leanprover--lean4---v4.31.0"
    assert solver.lean_runtime == "image"
    assert solver.harness_config["auth"] == "subscription"
    assert solver.config["model"] == "gpt-6-astra"
    assert solver.config["reasoning_effort"] == "max"
    assert solver._harness_cli is None
    payload, _ = prepared["batch"][0]
    prompt = solver.build_prompt(payload)
    assert runner.problems[0]["problem"] in prompt
    assert runner.problems[0]["formal_statement"] in prompt
    assert prompt == runner.competition_config["harness_instruction"].format(**payload)
    assert "verify_lean" not in prompt
    assert "add_to_file" not in prompt
    assert "./check.sh" in prompt
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = solver._build_sandbox(workspace)
    assert sandbox.image == "matharena-harness-arxivlean:lean4.31"
    assert any(
        "src=matharena-lean431-cache" in arg and arg.endswith(",readonly")
        for arg in sandbox.extra_args
    )

    # The same real competition must not silently switch existing API models.
    native_config = runner.load_solver_config(str(root / "configs/models/glm/glm-52.yaml"))
    assert runner._resolve_harness(native_config) is None
    native_args = runner._prepare_default_api_client_args(native_config["model_config"])
    assert {spec["function"]["name"] for _, spec in native_args["tools"]} == {
        "verify_lean",
        "verify_submission",
        "add_to_file",
        "loogle",
        "lean_explore_search",
    }
    assert all(callable(func) for func, _ in native_args["tools"])
    assert native_args["max_tool_calls"] == 200


@pytest.mark.parametrize("resume", (False, True))
def test_tool_free_codex_cannot_restore_tools_on_resume(tmp_path, resume):
    model = Model(
        "gpt-6-astra",
        api_url="http://127.0.0.1:1/v1",
        api_key="test-placeholder",
        reasoning="max",
        request_overrides={
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
        },
    )
    agent = Agent(
        type="codex",
        model=model,
        dir=tmp_path,
        executable="/usr/bin/true",
        minimal_context=False,
        tools_enabled=False,
    )
    command = agent.build_command("Solve it.", resume=resume)
    disabled = {
        command[i + 1] for i, flag in enumerate(command[:-1]) if flag == "--disable"
    }

    assert {
        "code_mode",
        "code_mode_only",
        "code_mode_host",
        "shell_tool",
        "unified_exec",
        "apps",
        "multi_agent",
        "plugins",
        "view_image",
        "browser_use",
    } <= disabled
    assert "--enable" not in command
    assert "--ignore-user-config" in command
    assert "skills.bundled.enabled=false" in command
    assert "project_doc_max_bytes=0" in command
    assert any("No tools are available" in argument for argument in command)
    assert agent.subagents == {}
    assert agent._effective_request_overrides() == {
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": False,
    }


def test_tool_free_mode_cannot_enable_lean_or_other_harnesses(tmp_path):
    config = {
        "model": "test",
        "harness": "codex",
        "allow_harness": False,
        "harness_config": {"workspace_root": str(tmp_path / "run")},
    }
    solver_config = {"type": "pure_model", "model_config": {}, "scaffold_config": None}
    with pytest.raises(ValueError, match="Lean harness"):
        HarnessSolver(solver_config, "{problem}", {**config, "is_lean_comp": True}, "")
    with pytest.raises(ValueError, match="only by Codex"):
        HarnessSolver(solver_config, "{problem}", {**config, "harness": "kimi"}, "")
    with pytest.raises(TypeError, match="booleans"):
        HarnessSolver(
            solver_config, "{problem}", {**config, "allow_harness": "false"}, ""
        )


@pytest.mark.parametrize("requested", (None, False, "codex"))
def test_legacy_codex_cannot_bypass_tool_free_policy(requested):
    runner = object.__new__(Runner)
    runner.competition_config = {}
    config = {"type": "codex_cli", "model_config": {"harness": requested}}
    with pytest.raises(ValueError, match="Legacy type: codex_cli"):
        runner._resolve_harness(config)
