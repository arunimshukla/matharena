import json
import runpy
import sys

import pytest
from flask import Flask

import arxivmath.app as app_module
from matharena.arxivmath_source import SCHEMA_VERSION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source_annotation():
    return {
        "source_first_schema_version": SCHEMA_VERSION,
        "keep": True,
        "stage": "selected_for_human_review",
        "question": "Under the stated hypotheses, what is the invariant N?",
        "answer": "7",
        "answer_type": "exact_scalar",
        "declared_variables": [],
        "basis_summary": (
            "The item is based on the principal theorem in the main-results section. The cited "
            "calculation fixes the normalization and evaluates the requested invariant exactly."
        ),
        "source_first": {
            "investigation_sha256": "investigation",
            "basis_summary": "The item is based on the principal theorem.",
            "novelty_type": "new_exact_value",
            "importance": "main",
            "refutation_status": "not_applicable",
            "prior_claim": None,
            "new_result": None,
            "evidence": [
                {
                    "quote": "The principal calculation proves that the invariant N has exact value seven.",
                    "source_file": "main.tex",
                    "start_line": 10,
                    "end_line": 10,
                }
            ],
            "verification": {
                "status": "passed",
                "passed": True,
                "reason": "The evidence uniquely determines the value.",
            },
            "source_input": {"complete_source": True},
        },
    }


def configure_app(monkeypatch, tmp_path):
    paper_root = tmp_path / "paper"
    paper = paper_root / "2608.00001"
    write_json(
        paper / "metadata.json",
        {
            "title": "An exact invariant",
            "abstract": "We calculate an invariant.",
            "authors": [],
        },
    )
    write_json(paper / "llm_annotation.json", source_annotation())
    monkeypatch.setattr(app_module, "PAPER_ROOT", str(paper_root))
    monkeypatch.setattr(app_module, "ANNOTATION_FILENAME_OVERRIDE", None)
    monkeypatch.setattr(app_module, "CHECK_ONLY_KEPT", False)
    monkeypatch.setattr(app_module, "FALSE_MODE", False)
    monkeypatch.setattr(app_module, "LEAN_MODE", False)
    app_module.SKIPPED_BY_SESSION.clear()
    app_module.app.config.update(TESTING=True, SECRET_KEY="test")
    return paper


def test_source_review_displays_basis_evidence_and_verifier(monkeypatch, tmp_path):
    configure_app(monkeypatch, tmp_path)
    response = app_module.app.test_client().get("/paper/2608.00001")
    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Source grounding" in page
    assert "principal theorem" in page
    assert "main.tex, lines 10-10" in page
    assert "Independent verification" in page
    assert "passed" in page
    assert "Blind solve rate" not in page
    assert "All locally valid candidates" not in page
    assert "Counterexample audit" not in page


def test_source_review_is_accept_reject_only_and_does_not_rewrite_question(monkeypatch, tmp_path):
    paper = configure_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()
    response = client.post(
        "/paper/2608.00001/annotate",
        data={
            "status": "keep",
            "question": "A silently rewritten question?",
            "answer": "999",
        },
    )
    assert response.status_code == 302
    saved = json.loads((paper / "llm_annotation.json").read_text())
    assert saved["question"] == source_annotation()["question"]
    assert saved["answer"] == "7"
    assert saved["review"]["status"] == "keep"
    assert "question" not in saved["review"]
    assert saved["stage"] == "human_accepted"


def test_source_rejected_or_unverified_items_are_not_in_review_queue(monkeypatch, tmp_path):
    paper = configure_app(monkeypatch, tmp_path)
    annotation = source_annotation()
    annotation["keep"] = False
    annotation["stage"] = "automatic_verification_rejected"
    write_json(paper / "llm_annotation.json", annotation)
    response = app_module.app.test_client().get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "No ArXivMath items awaiting human review" in page
    assert str(paper.parent) in page
    assert "automatically" in page
    assert "--false" in page


def test_kept_review_navigation_can_skip_to_the_next_kept_item(monkeypatch, tmp_path):
    first = configure_app(monkeypatch, tmp_path)
    second = first.parent / "2608.00002"
    write_json(second / "metadata.json", {"title": "Another invariant", "authors": []})
    annotation = source_annotation()
    annotation.update(stage="human_accepted", review={"status": "keep"})
    for paper in (first, second):
        write_json(paper / "llm_annotation.json", annotation)
    monkeypatch.setattr(app_module, "CHECK_ONLY_KEPT", True)

    response = app_module.app.test_client().get("/paper/2608.00001/skip", follow_redirects=True)

    assert response.status_code == 200
    assert "Another invariant" in response.get_data(as_text=True)
    for paper in (first, second):
        assert json.loads((paper / "llm_annotation.json").read_text()) == annotation


@pytest.mark.parametrize("false_mode", [False, True])
def test_cli_opens_source_review_without_a_mode_flag(monkeypatch, tmp_path, false_mode):
    paper = configure_app(monkeypatch, tmp_path)
    filename = "llm_metadata_false_source.json" if false_mode else "llm_annotation.json"
    annotation = source_annotation()
    if false_mode:
        annotation.update(
            true_statement="A true assertion.",
            false_statement="A false assertion.",
            falsity_explanation="A refutation.",
        )
    write_json(paper / filename, annotation)
    pages = []

    def inspect_app(app, **kwargs):
        response = app.test_client().get("/", follow_redirects=True)
        assert response.status_code == 200
        pages.append(response.get_data(as_text=True))

    monkeypatch.setattr(Flask, "run", inspect_app)
    monkeypatch.setattr(
        sys, "argv", ["app.py", "--paper-root", str(paper.parent)] + (["--false"] if false_mode else [])
    )

    runpy.run_path("arxivmath/app.py", run_name="__main__")

    assert len(pages) == 1
    assert "Source grounding" in pages[0]
    assert "Independent verification" in pages[0]
    assert ("A false assertion." if false_mode else annotation["question"]) in pages[0]
    assert "<textarea" not in pages[0]


@pytest.mark.parametrize(
    "mode,filename,field",
    [
        ("LEAN_MODE", "metadata_lean_abstract.json", "statement"),
    ],
)
def test_other_review_modes_retain_editable_fields(monkeypatch, tmp_path, mode, filename, field):
    paper = configure_app(monkeypatch, tmp_path)
    write_json(paper / filename, {"keep": True, field: "Original problem."})
    monkeypatch.setattr(app_module, mode, True)
    client = app_module.app.test_client()
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert f'<textarea id="{field}"' in response.get_data(as_text=True)
    response = client.post("/paper/2608.00001/annotate", data={"status": "keep", field: "Reviewed problem."})
    assert response.status_code == 302
    saved = json.loads((paper / filename).read_text())
    assert saved[field] == "Original problem."
    assert saved["review"][field] == "Reviewed problem."
    assert saved["review"]["status"] == "keep"


def test_finished_queue_can_continue_when_new_items_arrive(monkeypatch, tmp_path):
    paper = configure_app(monkeypatch, tmp_path)
    annotation = source_annotation()
    annotation.update(keep=False, stage="awaiting_verification")
    write_json(paper / "llm_annotation.json", annotation)
    client = app_module.app.test_client()
    assert "Refresh this page" in client.get("/done").get_data(as_text=True)
    write_json(paper / "llm_annotation.json", source_annotation())
    response = client.get("/done", follow_redirects=True)
    assert response.status_code == 200
    assert source_annotation()["question"] in response.get_data(as_text=True)
    response = client.get("/paper/2608.00001/skip", follow_redirects=True)
    assert response.status_code == 200
    assert "All currently available items" in response.get_data(as_text=True)
