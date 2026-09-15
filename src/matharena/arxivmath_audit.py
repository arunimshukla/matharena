"""Release audit for the abstract-screened ArXivMath source pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matharena.arxivbench_utils import load_model_config
from matharena.arxivmath_source import (
    FALSE_SCREEN_FILENAME,
    FALSE_INVESTIGATION_FILENAME,
    FALSE_FINAL_FILENAME,
    FALSE_FIELDS,
    validate_source_model,
    metadata_arxiv_version,
    validate_false_payload,
    ABSTRACT_SCREEN_FILENAME,
    FINAL_ANNOTATION_FILENAME,
    INVESTIGATION_FILENAME,
    MODEL_SOURCE_CLEANING_VERSION,
    SCHEMA_VERSION,
    VERIFICATION_CHECKS,
    abstract_screen_record_is_current,
    answers_equivalent,
    build_investigation_source,
    investigation_payload_sha256,
    load_json,
    paper_source_unavailable,
    read_source_artifacts,
    sha256_file,
    validate_normalized_investigation,
)


FINAL_QUESTION_FIELDS = (
    "question",
    "answer",
    "answer_type",
    "declared_variables",
    "basis_summary",
)


def final_question_payload_matches(final: dict[str, Any], investigation: dict[str, Any]) -> bool:
    fields = FALSE_FIELDS if investigation.get("benchmark") == "brokenarxiv" else FINAL_QUESTION_FIELDS
    return all(final.get(field) == investigation.get(field) for field in fields)


def _audit_source_input(
    raw_source_text: str,
    source_input: Any,
    identity: dict[str, Any],
    blockers: list[str],
) -> str:
    try:
        prepared = build_investigation_source(raw_source_text)
    except Exception as exc:
        blockers.append(f"investigation_source_cleaning_failed:{type(exc).__name__}")
        return ""
    expected = prepared.manifest_record()
    if source_input != expected:
        blockers.append("investigation_source_input_stale_or_malformed")
    if identity.get("source_input_sha256") != expected["source_sha256"]:
        blockers.append("investigation_identity_source_input_stale")
    return prepared.text


def _audit_identity(
    state: dict[str, Any],
    identity: dict[str, Any],
    blockers: list[str],
    *,
    require_generation: bool,
) -> None:
    for field in ("prompt_sha256", "model_config_sha256"):
        if not isinstance(identity.get(field), str) or not identity.get(field):
            blockers.append(f"investigation_identity_{field}_missing")
    max_source_tokens = identity.get("max_source_tokens")
    if isinstance(max_source_tokens, bool) or not isinstance(max_source_tokens, int) or max_source_tokens < 20_000:
        blockers.append("investigation_identity_max_source_tokens_invalid")
    if identity.get("cleaning_version") != MODEL_SOURCE_CLEANING_VERSION:
        blockers.append("investigation_identity_cleaning_version_stale")
    if require_generation:
        generation = state.get("generation")
        if not isinstance(generation, dict) or generation.get("status") != "completed":
            blockers.append("investigation_generation_record_missing")
        elif generation.get("model_config_sha256") != identity.get("model_config_sha256"):
            blockers.append("investigation_generation_config_stale")


def _audit_verification(
    verification: dict[str, Any],
    investigation: dict[str, Any],
    investigation_sha256: str,
    blockers: list[str],
    source_text: str = "",
) -> None:
    if verification.get("investigation_sha256") != investigation_sha256:
        blockers.append("question_verification_stale")
    for field in ("model_config_sha256", "prompt_sha256", "input_sha256"):
        if not isinstance(verification.get(field), str) or not verification.get(field):
            blockers.append(f"question_verification_{field}_missing")
    if investigation.get("benchmark") == "brokenarxiv":
        validation = validate_false_payload(verification.get("parsed"), source_text, investigation=investigation)
        if validation.get("valid") is not True or any(
            verification.get(key) != value for key, value in validation.items()
        ):
            blockers.append("false_verification_invalid_or_changed")
        elif verification.get("status") != ("passed" if validation["passed"] else "rejected"):
            blockers.append("false_verification_result_inconsistent")
        return
    checks = verification.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != set(VERIFICATION_CHECKS)
        or not all(isinstance(value, bool) for value in checks.values())
    ):
        blockers.append("question_verification_checks_malformed")
        checks = {}
    checks_pass = bool(checks) and all(checks.values())
    answer_matches = answers_equivalent(
        verification.get("derived_answer"),
        investigation.get("answer"),
    )
    passed = verification.get("status") == "passed"
    if (
        verification.get("checks_pass") is not checks_pass
        or verification.get("answer_matches") is not answer_matches
        or verification.get("passed") is not passed
        or passed is not (checks_pass and answer_matches)
    ):
        blockers.append("question_verification_result_inconsistent")


def audit_paper(
    paper_dir: Path,
    *,
    source_cache: str | Path | None = None,
    screen_filename: str = ABSTRACT_SCREEN_FILENAME,
    investigation_filename: str = INVESTIGATION_FILENAME,
    final_filename: str = FINAL_ANNOTATION_FILENAME,
    allow_source_unavailable: bool,
    false_mode: bool = False,
    require_human_review: bool = True,
) -> dict[str, Any]:
    if false_mode:
        allow_source_unavailable = True
        screen_filename = FALSE_SCREEN_FILENAME if screen_filename == ABSTRACT_SCREEN_FILENAME else screen_filename
        investigation_filename = (
            FALSE_INVESTIGATION_FILENAME if investigation_filename == INVESTIGATION_FILENAME else investigation_filename
        )
        final_filename = FALSE_FINAL_FILENAME if final_filename == FINAL_ANNOTATION_FILENAME else final_filename
    paper_id = paper_dir.name
    blockers: list[str] = []
    warnings: list[str] = []
    metadata = load_json(paper_dir / "metadata.json", {})
    screen_path = paper_dir / screen_filename
    screen = load_json(screen_path, {})
    if not isinstance(metadata, dict) or not abstract_screen_record_is_current(screen, metadata):
        blockers.append("abstract_screen_stale_or_malformed")
    elif screen.get("decision") != "accept":
        blockers.append("paper_not_accepted_by_abstract_screen")

    ingestion = load_json(paper_dir / "source_ingestion.json", {})
    if paper_source_unavailable(paper_dir):
        reason = f"source_unavailable:{ingestion.get('error_type') or ingestion.get('status') or 'failed'}"
        if allow_source_unavailable:
            warnings.append(reason)
            final = load_json(paper_dir / final_filename, {})
            if not isinstance(final, dict) or final.get("stage") != "source_unavailable":
                blockers.append("source_unavailable_final_tombstone_missing")
        else:
            blockers.append(reason)
        return {
            "paper_id": paper_id,
            "ready": not blockers,
            "source_unavailable": True,
            "question_generated": False,
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
        }

    try:
        raw_source_text, provenance, _ = read_source_artifacts(paper_dir, source_cache)
    except Exception as exc:
        blockers.append(f"source_artifacts_invalid:{type(exc).__name__}")
        return {
            "paper_id": paper_id,
            "ready": False,
            "source_unavailable": False,
            "question_generated": False,
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
        }

    if false_mode:
        try:
            if metadata_arxiv_version(metadata).canonical_id != provenance["arxiv_id"]:
                blockers.append("metadata_source_revision_mismatch")
        except ValueError:
            blockers.append("metadata_revision_missing_or_invalid")
    unresolved_count = provenance.get("unresolved_include_count") or 0
    if unresolved_count and false_mode:
        blockers.append("false_item_requires_complete_source")
    if unresolved_count:
        warnings.append(f"source_has_allowed_unresolved_includes:{unresolved_count}")

    state_path = paper_dir / investigation_filename
    state = load_json(state_path, {})
    if not isinstance(state, dict) or state.get("status") not in {"completed", "excluded"}:
        blockers.append("source_investigation_missing_or_incomplete")
        return {
            "paper_id": paper_id,
            "ready": False,
            "source_unavailable": False,
            "question_generated": False,
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
        }
    if state.get("schema_version") != SCHEMA_VERSION:
        blockers.append("source_investigation_schema_invalid")

    if false_mode:
        # Check model settings; edits to prompt files do not invalidate saved results.
        for stage, record in (
            ("screen", screen),
            ("investigate", state.get("generation") or {}),
            ("verify", state.get("verification") or {}),
        ):
            if not record:  # Missing required stages are checked below; exclusions need none.
                continue
            try:
                config_path = record["model_config_path"]
                if sha256_file(config_path) != record["model_config_sha256"]:
                    blockers.append(f"{stage}_config_changed")
                validate_source_model(load_model_config(config_path), false_mode=True)
            except (OSError, ValueError, TypeError, KeyError):
                blockers.append(f"{stage}_settings_missing_or_invalid")
    identity = state.get("identity") or {}
    if not isinstance(identity, dict):
        identity = {}
        blockers.append("investigation_identity_missing_or_malformed")
    if identity.get("source_sha256") != provenance.get("combined_source_sha256"):
        blockers.append("source_investigation_source_stale")
    if not screen_path.is_file() or identity.get("screen_sha256") != sha256_file(screen_path):
        blockers.append("source_investigation_screen_stale")
    excluded = state.get("status") == "excluded"
    _audit_identity(state, identity, blockers, require_generation=not excluded)
    model_source_text = _audit_source_input(raw_source_text, state.get("source_input"), identity, blockers)
    recorded_source_input = state.get("source_input") or {}
    recorded_tokens = (
        recorded_source_input.get("approximate_tokens") if isinstance(recorded_source_input, dict) else None
    )
    max_source_tokens = identity.get("max_source_tokens")
    if (
        not excluded
        and isinstance(recorded_tokens, int)
        and not isinstance(recorded_tokens, bool)
        and isinstance(max_source_tokens, int)
        and not isinstance(max_source_tokens, bool)
        and recorded_tokens > max_source_tokens
    ):
        blockers.append("source_investigation_exceeds_recorded_token_limit")
    source_record = state.get("source") or {}
    if not isinstance(source_record, dict) or any(
        source_record.get(field) != provenance.get(field)
        for field in (
            "arxiv_id",
            "base_id",
            "version",
            "cache_key",
            "combined_source_sha256",
            "unresolved_include_count",
            "unresolved_includes_allowed",
        )
    ):
        blockers.append("source_investigation_provenance_stale")

    if excluded:
        if state.get("reason") != "source_too_large":
            blockers.append("source_exclusion_reason_invalid")
        max_tokens = max_source_tokens
        approximate_token_count = recorded_tokens
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or isinstance(approximate_token_count, bool)
            or not isinstance(approximate_token_count, int)
            or approximate_token_count <= max_tokens
        ):
            blockers.append("source_oversized_exclusion_no_longer_valid")
        final = load_json(paper_dir / final_filename, {})
        source_first = final.get("source_first") if isinstance(final, dict) else None
        if (
            not isinstance(final, dict)
            or final.get("source_first_schema_version") != SCHEMA_VERSION
            or final.get("keep") is not False
            or final.get("stage") != "source_oversized"
            or not isinstance(source_first, dict)
            or source_first.get("source_sha256") != provenance.get("combined_source_sha256")
            or source_first.get("source_input") != state.get("source_input")
            or source_first.get("rejection_reason") != "source_too_large"
            or source_first.get("investigation_state_sha256") != sha256_file(state_path)
        ):
            blockers.append("source_oversized_final_tombstone_missing_or_stale")
        warnings.append(f"source_oversized:{approximate_token_count or 'unknown'}")
        return {
            "paper_id": paper_id,
            "ready": not blockers,
            "source_unavailable": False,
            "question_generated": False,
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
        }

    investigation = state.get("investigation")
    if not isinstance(investigation, dict):
        blockers.append("source_investigation_payload_missing")
        investigation = {}
    investigation_sha256 = investigation_payload_sha256(investigation)
    if investigation.get("investigation_sha256") != investigation_sha256:
        blockers.append("source_investigation_payload_hash_stale")
    investigation_validation = validate_normalized_investigation(
        investigation,
        model_source_text,
    )
    if investigation_validation.get("valid") is not True:
        blockers.append("source_investigation_invalid:" + str(investigation_validation.get("reason") or "unknown"))

    final = load_json(paper_dir / final_filename, {})
    if not isinstance(final, dict) or final.get("source_first_schema_version") != SCHEMA_VERSION:
        blockers.append("source_first_final_annotation_missing")
        final = {}
    source_first = final.get("source_first") or {}
    if not isinstance(source_first, dict):
        source_first = {}
        blockers.append("source_first_final_provenance_missing")
    if source_first.get("investigation_sha256") != investigation_sha256:
        blockers.append("source_first_final_investigation_stale")
    if state_path.is_file() and source_first.get("investigation_state_sha256") != sha256_file(state_path):
        blockers.append("source_first_final_state_stale")
    if source_first.get("source_sha256") != provenance.get("combined_source_sha256"):
        blockers.append("source_first_final_source_stale")
    if source_first.get("source_input") != state.get("source_input"):
        blockers.append("source_first_final_source_input_stale")
    if source_first.get("basis_summary") != investigation.get("basis_summary"):
        blockers.append("source_first_final_basis_summary_stale")
    if source_first.get("refutation_status") != investigation.get("refutation_status"):
        blockers.append("source_first_final_refutation_status_stale")

    question_generated = investigation.get("keep") is True
    if not question_generated:
        if source_first.get("rejection_reason") != investigation.get("rejection_reason"):
            blockers.append("source_first_final_rejection_reason_stale")
        if final.get("keep") is not False or final.get("stage") != "no_question":
            blockers.append("no_question_final_tombstone_missing")
    else:
        verification = state.get("verification")
        if not isinstance(verification, dict) or verification.get("status") not in {
            "passed",
            "rejected",
        }:
            blockers.append("question_verification_missing_or_incomplete")
            verification = {}
        _audit_verification(verification, investigation, investigation_sha256, blockers, model_source_text)
        expected_final_verification = {
            key: value for key, value in verification.items() if key not in {"raw", "parsed", "detailed_cost"}
        }
        if source_first.get("verification") != expected_final_verification:
            blockers.append("source_first_final_verification_stale")
        for field in ("novelty_type", "importance", "prior_claim", "new_result", "evidence"):
            if source_first.get(field) != investigation.get(field):
                blockers.append(f"source_first_final_{field}_stale")
        if verification.get("status") == "rejected":
            if final.get("keep") is not False or final.get("stage") != "automatic_verification_rejected":
                blockers.append("automatic_verification_rejection_not_applied")
        elif verification.get("status") == "passed":
            if not final_question_payload_matches(final, investigation):
                blockers.append("source_first_final_question_payload_mismatch")
            review = final.get("review") or {}
            review_status = review.get("status") if isinstance(review, dict) else None
            if false_mode and review_status in {"keep", "discard"} and review.get("binding") != sha256_file(state_path):
                blockers.append("human_review_is_stale")
            if require_human_review and review_status not in {"keep", "discard"}:
                blockers.append("selected_question_human_review_missing")
            elif review_status == "keep" and (final.get("keep") is not True or final.get("stage") != "human_accepted"):
                blockers.append("selected_question_review_state_inconsistent")
            elif review_status == "discard" and (
                final.get("keep") is not False or final.get("stage") != "human_rejected"
            ):
                blockers.append("selected_question_review_state_inconsistent")

    if investigation.get("refutation_status") == "source_refutation_not_parser_gradable":
        warnings.append(str(investigation["refutation_status"]))

    return {
        "paper_id": paper_id,
        "ready": not blockers,
        "source_unavailable": False,
        "question_generated": question_generated,
        "verification_status": (state.get("verification") or {}).get("status"),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }


__all__ = ["audit_paper", "final_question_payload_matches"]
