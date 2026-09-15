import json

import pytest

from matharena.arxivmath_source import (
    SCHEMA_VERSION,
    SourceStateError,
    abstract_screen_input_sha256,
    abstract_screen_record_is_current,
    abstract_screen_selection,
    answers_equivalent,
    build_investigation_source,
    clean_tex_for_model,
    investigation_payload_sha256,
    investigation_verification_is_current,
    locate_evidence_quote,
    metadata_arxiv_version,
    sha256_text,
    validate_investigation_payload,
    validate_normalized_investigation,
    validate_parser_safe_answer,
)


def basis_summary() -> str:
    return (
        "The question is based on the paper's principal theorem, stated in the main-results "
        "section and supported by the displayed calculation immediately following it. The cited "
        "source passage fixes the hypotheses, normalization, and exact invariant whose value is "
        "requested, so the benchmark item can stand independently of the article."
    )


def screen_record(metadata, *, decision="accept"):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "decision": decision,
        "identity": {
            "metadata_input_sha256": abstract_screen_input_sha256(metadata),
            "prompt_sha256": "prompt",
            "rendered_prompt_sha256": "rendered",
            "model_config_sha256": "config",
        },
    }


def accepted_payload(source):
    quote = "The principal calculation proves that the invariant N has exact value seven."
    assert quote in source
    return {
        "keep": True,
        "question": "Under the stated hypotheses, what is the exact value of the invariant N?",
        "answer": "7",
        "answer_type": "exact_scalar",
        "declared_variables": [],
        "basis_summary": basis_summary(),
        "novelty_type": "new_exact_value",
        "importance": "main",
        "refutation_status": "not_applicable",
        "prior_claim": None,
        "new_result": None,
        "evidence_quotes": [quote],
    }


def test_abstract_screen_freshness_and_selection(tmp_path):
    root = tmp_path / "paper"
    metadata = {"title": "A result", "abstract": "We determine an exact value."}
    paper = root / "2608.00001"
    paper.mkdir(parents=True)
    (paper / "metadata.json").write_text(json.dumps(metadata))
    record = screen_record(metadata)
    (paper / "abstract_screen.json").write_text(json.dumps(record))

    assert abstract_screen_record_is_current(record, metadata)
    old_record = {**record, "schema_version": SCHEMA_VERSION - 1}
    assert not abstract_screen_record_is_current(old_record, metadata)
    selection = abstract_screen_selection(root)
    assert selection["complete"] is True
    assert selection["accepted"] == ["2608.00001"]

    metadata["abstract"] = "Changed abstract."
    (paper / "metadata.json").write_text(json.dumps(metadata))
    assert abstract_screen_selection(root)["complete"] is False


def test_investigation_source_uses_complete_comment_cleaned_tex():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "% discarded author note\n"
        "\\section{Main result} % trailing note\n"
        "A theorem with 50\\% probability.\n"
        "% MATHARENA_SOURCE_END file=main.tex\n"
    )
    prepared = build_investigation_source(source)
    manifest = prepared.manifest_record()
    assert "discarded author note" not in prepared.text
    assert "trailing note" not in prepared.text
    assert "50\\% probability" in prepared.text
    assert "% MATHARENA_SOURCE_BEGIN file=main.tex" in prepared.text
    assert "% MATHARENA_SOURCE_END file=main.tex" in prepared.text
    assert manifest["complete_source"] is True
    assert manifest["source_sha256"] == sha256_text(prepared.text)
    assert manifest["raw_source_chars"] == len(source)
    assert manifest["source_chars"] == len(prepared.text)
    assert manifest["removed_comment_chars"] > 0


def test_comment_cleaner_preserves_literal_percent_and_removes_comment_environment():
    source = (
        "\\verb|literal % value| % remove this\n"
        "\\begin{verbatim}\ninside % verbatim\n\\end{verbatim}\n"
        "\\begin{comment}\nremove % all of this\n"
        "% \\end{comment}\nstill removed\n\\end{comment}\n"
        "% \\begin{comment}\n"
        "not inside a comment environment\n"
        "visible\n"
    )
    cleaned = clean_tex_for_model(source)
    assert "literal % value" in cleaned
    assert "inside % verbatim" in cleaned
    assert "remove this" not in cleaned
    assert "remove % all of this" not in cleaned
    assert "still removed" not in cleaned
    assert cleaned.count("\n") == source.count("\n")
    assert "visible" in cleaned
    assert "not inside a comment environment" in cleaned


def test_source_size_is_measured_after_comment_removal():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        + "% "
        + ("discarded " * 10_000)
        + "\nA short active theorem.\n"
        + "% MATHARENA_SOURCE_END file=main.tex\n"
    )
    manifest = build_investigation_source(source).manifest_record()
    assert manifest["raw_source_chars"] > 100_000
    assert manifest["source_chars"] < 1_000
    assert manifest["approximate_tokens"] < 500


def test_investigation_contract_accepts_exactly_one_grounded_question():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "The principal calculation proves that the invariant N has exact value seven.\n"
    )
    validation = validate_investigation_payload(accepted_payload(source), source)
    assert validation["valid"] is True
    investigation = validation["investigation"]
    assert investigation["answer"] == "7"
    assert investigation["evidence"][0]["source_file"] == "main.tex"
    assert investigation["investigation_sha256"] == investigation_payload_sha256(investigation)
    assert validate_normalized_investigation(investigation, source)["valid"] is True


def test_investigation_question_need_not_end_with_question_mark():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "The principal calculation proves that the invariant N has exact value seven.\n"
    )
    payload = accepted_payload(source)
    payload["question"] = "Determine the exact value of the invariant N under the stated hypotheses"

    assert validate_investigation_payload(payload, source)["valid"] is True

    payload["question"] = "   "
    validation = validate_investigation_payload(payload, source)
    assert validation["valid"] is False
    assert validation["reason"] == "question_must_be_nonempty_string"


def test_investigation_accepts_short_nonunique_evidence_quote():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "The principal calculation proves that the invariant N has exact value seven.\n"
        "The value is seven.\n"
    )
    payload = accepted_payload(source)
    payload["evidence_quotes"] = ["seven."]

    validation = validate_investigation_payload(payload, source)

    assert validation["valid"] is True
    assert validation["investigation"]["evidence"][0]["quote"] == "seven."
    assert validation["investigation"]["evidence"][0]["start_char"] == source.index("seven.")


@pytest.mark.parametrize("source_ending", ["\n", "\r\n"])
@pytest.mark.parametrize("quote_ending", ["\n", "\r\n"])
def test_evidence_accepts_line_endings_and_preserves_source_locations(source_ending, quote_ending):
    lines = ["The theorem has these hypotheses.", "Its conclusion follows from the construction."]
    source_quote = source_ending.join(lines)
    source = "% MATHARENA_SOURCE_BEGIN file=main.tex\nIntro: " + source_quote + "\n"

    evidence = locate_evidence_quote(source, quote_ending.join(lines))

    assert evidence["quote"] == source_quote
    assert evidence["quote_sha256"] == sha256_text(source_quote)
    assert evidence["start_char"] == source.index(source_quote)
    assert evidence["end_char"] == source.index(source_quote) + len(source_quote)
    assert evidence["start_line"] == 2
    assert evidence["end_line"] == 3
    assert evidence["source_file"] == "main.tex"


def test_evidence_line_ending_variants_still_require_uniqueness():
    quote = "The theorem has these hypotheses.\nIts conclusion follows."
    source = quote.replace("\n", "\r\n") + "\nOther text.\n" + quote
    with pytest.raises(SourceStateError, match="not unique"):
        locate_evidence_quote(source, quote)
    first = locate_evidence_quote(source, quote, require_unique=False)
    assert first["start_char"] == 0
    assert first["quote"] == quote.replace("\n", "\r\n")


@pytest.mark.parametrize(
    "quote",
    [
        "The theorem has these hypotheses.\n\nIts conclusion follows.",
        "The theorem has  these hypotheses.\nIts conclusion follows.",
        "The theorem has these hypotheses.\nIts conclusion fails.",
    ],
)
def test_evidence_does_not_normalize_content_or_other_whitespace(quote):
    source = "The theorem has these hypotheses.\r\nIts conclusion follows."
    with pytest.raises(SourceStateError, match="not a verbatim"):
        locate_evidence_quote(source, quote)


def test_evidence_still_detects_overlapping_matches():
    with pytest.raises(SourceStateError, match="not unique"):
        locate_evidence_quote("a" * 21, "a" * 20)


def test_normalized_investigation_reuses_full_generation_validator():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "The principal calculation proves that the invariant N has exact value seven.\n"
    )
    validation = validate_investigation_payload(accepted_payload(source), source)
    investigation = validation["investigation"]
    investigation["question"] = "According to the paper, what is the invariant N?"
    investigation["investigation_sha256"] = investigation_payload_sha256(investigation)
    revalidated = validate_normalized_investigation(investigation, source)
    assert revalidated["valid"] is False
    assert revalidated["reason"] == "question_references_paper"


def test_metadata_arxiv_version_requires_an_exact_consistent_pin():
    with pytest.raises(SourceStateError, match="exact versioned_id"):
        metadata_arxiv_version({"id": "2608.00001"})
    with pytest.raises(SourceStateError, match="does not match"):
        metadata_arxiv_version({"id": "2608.00001", "versioned_id": "2608.00002v1"})
    with pytest.raises(SourceStateError, match="does not match"):
        metadata_arxiv_version({"id": "2608.00001v2", "versioned_id": "2608.00001v3"})
    version = metadata_arxiv_version({"id": "2608.00001", "versioned_id": "2608.00001v4"})
    assert version.canonical_id == "2608.00001v4"


def test_investigation_contract_rejects_short_basis_and_paper_reference():
    source = "The principal calculation proves that the invariant N has exact value seven."
    payload = accepted_payload(source)
    payload["basis_summary"] = "Too short."
    assert validate_investigation_payload(payload, source)["reason"] == ("basis_summary_must_contain_40_to_120_words")

    payload = accepted_payload(source)
    payload["question"] = "According to the paper, what is the invariant N?"
    assert validate_investigation_payload(payload, source)["reason"] == "question_references_paper"

    payload = accepted_payload(source)
    payload["questions"] = [payload["question"]]
    assert validate_investigation_payload(payload, source)["reason"] == ("investigation_has_unexpected_fields")


def test_source_stage_classifies_refutations_independently():
    source = "The principal calculation proves that the invariant N has exact value seven."
    assert validate_investigation_payload(accepted_payload(source), source)["valid"] is True

    refutation = accepted_payload(source)
    refutation.update(
        {
            "novelty_type": "counterexample_to_prior_conjecture",
            "refutation_status": "question_targets_refutation",
            "prior_claim": "The invariant was conjectured to equal six.",
            "new_result": "The invariant equals seven.",
        }
    )
    assert validate_investigation_payload(refutation, source)["valid"] is True

    changed_prediction = {
        **refutation,
        "novelty_type": "different_from_prior_prediction",
    }
    assert validate_investigation_payload(changed_prediction, source)["valid"] is True

    inconsistent = {**refutation, "refutation_status": "not_applicable"}
    assert validate_investigation_payload(inconsistent, source)["reason"] == (
        "refutation_question_has_invalid_disposition"
    )

    rejection = validate_investigation_payload(
        {
            "keep": False,
            "basis_summary": basis_summary(),
            "refutation_status": "source_refutation_not_parser_gradable",
            "rejection_reason": "The result has no unique exact parser-safe answer.",
        },
        source,
    )
    assert rejection["valid"] is True


def test_parser_contract_and_answer_equivalence():
    assert validate_parser_safe_answer("7")["keep"] is True
    assert validate_parser_safe_answer(r"\left\lfloor x \right\rfloor")["keep"] is False
    assert answers_equivalent("7", "7.0")
    assert not answers_equivalent("7", "8")


def test_verification_currency_binds_question_and_inputs():
    investigation = {
        "keep": True,
        "question": "What is N?",
        "answer": "7",
        "answer_type": "exact_scalar",
        "declared_variables": [],
        "basis_summary": basis_summary(),
        "novelty_type": "new_exact_value",
        "importance": "main",
        "refutation_status": "not_applicable",
        "prior_claim": None,
        "new_result": None,
        "evidence": [],
    }
    state = {
        "investigation": investigation,
        "verification": {
            "status": "passed",
            "investigation_sha256": investigation_payload_sha256(investigation),
            "model_config_sha256": "config",
            "prompt_sha256": "prompt",
            "input_sha256": "input",
        },
    }
    assert investigation_verification_is_current(
        state,
        model_config_sha256="config",
        prompt_sha256="prompt",
        input_sha256="input",
    )
    state["investigation"]["answer"] = "8"
    assert not investigation_verification_is_current(
        state,
        model_config_sha256="config",
        prompt_sha256="prompt",
        input_sha256="input",
    )


def test_source_file_tracking_returns_to_parent_after_inlined_child():
    source = (
        "% MATHARENA_SOURCE_BEGIN file=main.tex\n"
        "parent before\n"
        "% MATHARENA_SOURCE_BEGIN file=child.tex via=main.tex:2\n"
        "child body\n"
        "% MATHARENA_SOURCE_END file=child.tex\n"
        "parent after\n"
        "% MATHARENA_SOURCE_END file=main.tex\n"
    )
    child = locate_evidence_quote(source, "child body\n", min_chars=1)
    parent = locate_evidence_quote(source, "parent after\n", min_chars=1)
    assert child["source_file"] == "child.tex"
    assert parent["source_file"] == "main.tex"
