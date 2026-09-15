"""Keep the other benchmarks usable without retaining the old ArXivMath route."""

import sys
from pathlib import Path

import pytest

from arxivmath.scripts.shared import create_queries, fulltext_review, verify_queries


MODULES = (create_queries, verify_queries, fulltext_review)


@pytest.mark.parametrize("module", MODULES)
def test_shared_cli_requires_explicit_benchmark_before_loading_model(monkeypatch, module):
    def unexpected_model_load(*args, **kwargs):
        pytest.fail("An unsupported generation route must not load a model.")

    monkeypatch.setattr(module, "load_model_config", unexpected_model_load)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--model-config", "unused"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("mode", ("--lean",))
def test_retained_modes_load_existing_prompts_without_requests(monkeypatch, module, mode):
    monkeypatch.setattr(module, "resolve_model_config_path", lambda _: "unused")
    monkeypatch.setattr(module, "load_model_config", lambda _: {"model": "unused"})
    monkeypatch.setattr(module, "APIClient", lambda **kwargs: object())
    monkeypatch.setattr(module, "list_paper_ids", lambda _: [])
    argv = [module.__name__, "--model-config", "unused", mode]
    if module is fulltext_review and mode == "--lean":
        argv += ["--prompt", "arxivmath/prompts/lean/hidden_condition.md"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_lean_fulltext_requires_its_own_prompt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fulltext_review", "--model-config", "unused", "--lean"])
    with pytest.raises(SystemExit) as exc:
        fulltext_review.main()
    assert exc.value.code == 2


def test_lean_semantic_judge_still_has_a_default_prompt(monkeypatch):
    module = verify_queries
    monkeypatch.setattr(module, "resolve_model_config_path", lambda _: "unused")
    monkeypatch.setattr(module, "load_model_config", lambda _: {"model": "unused"})
    monkeypatch.setattr(module, "APIClient", lambda **kwargs: object())
    monkeypatch.setattr(module, "list_paper_ids", lambda _: [])
    monkeypatch.setattr(sys, "argv", [module.__name__, "--model-config", "unused", "--semantic-judge"])
    module.main()


@pytest.mark.parametrize("mode", ("lean_mode",))
def test_retained_annotation_selection(mode):
    assert create_queries.needs_annotation({}, **{mode: True})
    assert not create_queries.needs_annotation({"keep": False}, **{mode: True})


@pytest.mark.parametrize("mode", ("lean_mode", "semantic_judge"))
def test_retained_verification_selection_and_templates(mode):
    annotation = {
        "keep": True,
        "question": "Question",
        "statement": "Statement",
        "formalized_statement": "Lean code",
        "_metadata": {"title": "Title", "abstract": "Abstract"},
    }
    prompt_paths = {
        "lean_mode": "lean/verify_lean_abstract.md",
        "semantic_judge": "lean/semantic_judge.md",
    }
    assert verify_queries.needs_verification(annotation, **{mode: True})
    template = (Path("arxivmath/prompts") / prompt_paths[mode]).read_text()
    assert verify_queries.render_prompt(template, annotation, **{mode: True})


def test_helpers_cannot_fall_back_to_old_arxivmath_mode():
    with pytest.raises(ValueError, match="Select Lean"):
        create_queries.needs_annotation({}, overwrite=True)
    with pytest.raises(ValueError, match="Select Lean"):
        verify_queries.needs_verification({})
    with pytest.raises(ValueError, match="Select Lean"):
        verify_queries.render_prompt("{question}", {})


@pytest.mark.parametrize("module", MODULES)
def test_old_false_generation_mode_is_not_available(monkeypatch, module):
    monkeypatch.setattr(sys, "argv", [module.__name__, "--model-config", "unused", "--false"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
