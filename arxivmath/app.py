#!/usr/bin/env python3
import argparse
import json
import os
import uuid
from datetime import datetime, timezone

from flask import Flask, abort, redirect, render_template, request, session, url_for

from matharena.arxivmath_source import paper_source_unavailable
from matharena.arxivmath_source import (
    FALSE_FINAL_FILENAME,
    FINAL_ANNOTATION_FILENAME,
    SCHEMA_VERSION,
    FALSE_INVESTIGATION_FILENAME,
    INVESTIGATION_FILENAME,
    sync_review_annotation,
    sha256_file,
)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_ROOT = os.path.join(APP_ROOT, "paper")
LEAN_ANNOTATION_FILENAME = "metadata_lean_abstract.json"
ANNOTATION_FILENAME_OVERRIDE = None
CHECK_ONLY_KEPT = False
FALSE_MODE = False
LEAN_MODE = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
SKIPPED_BY_SESSION = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_field_specs():
    if LEAN_MODE:
        return [
            {
                "name": "statement",
                "label": "Extracted problem statement",
                "placeholder": "No extracted statement available.",
                "empty_text": "No extracted statement available.",
                "height": 260,
            },
            {
                "name": "proof",
                "label": "Informal proof",
                "placeholder": "No informal proof available yet.",
                "empty_text": "No informal proof available yet.",
                "height": 260,
            },
            {
                "name": "formalized_statement",
                "label": "Formalized statement",
                "placeholder": "No formalized statement available yet.",
                "empty_text": "No formalized statement available yet.",
                "height": 220,
            },
        ]
    if FALSE_MODE:
        fields = (
            ("true_statement", "True statement"),
            ("false_statement", "False statement"),
            ("falsity_explanation", "Reference refutation"),
        )
    else:
        fields = (("question", "Question"), ("answer", "Answer"))
    return [
        {"name": name, "label": label, "empty_text": "No content available.", "editable": False}
        for name, label in fields
    ]


def list_paper_ids(check_only_kept=False):
    if not os.path.isdir(PAPER_ROOT):
        return []
    paper_ids = []
    for name in os.listdir(PAPER_ROOT):
        meta_path = os.path.join(PAPER_ROOT, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        annotation = load_annotation(name)
        if not (LEAN_MODE):
            if annotation.get("source_first_schema_version") != SCHEMA_VERSION:
                continue
            if paper_source_unavailable(os.path.join(PAPER_ROOT, name)):
                continue
        if annotation.get("keep") is not True:
            continue
        review = annotation.get("review")
        status = review.get("status") if isinstance(review, dict) else None
        if (check_only_kept and (status == "keep" or not status)) or (
            not check_only_kept and status not in {"keep", "discard"}
        ):
            paper_ids.append(name)
    return sorted(paper_ids)


def load_metadata(paper_id):
    meta_path = os.path.join(PAPER_ROOT, paper_id, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotation_filename():
    if ANNOTATION_FILENAME_OVERRIDE:
        return ANNOTATION_FILENAME_OVERRIDE
    if LEAN_MODE:
        return LEAN_ANNOTATION_FILENAME
    return FALSE_FINAL_FILENAME if FALSE_MODE else FINAL_ANNOTATION_FILENAME


def load_annotation(paper_id):
    if not (LEAN_MODE):
        return sync_review_annotation(
            os.path.join(PAPER_ROOT, paper_id),
            investigation_filename=FALSE_INVESTIGATION_FILENAME if FALSE_MODE else INVESTIGATION_FILENAME,
            final_filename=annotation_filename(),
        )
    path = os.path.join(PAPER_ROOT, paper_id, annotation_filename())
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotation(paper_id, data):
    path = os.path.join(PAPER_ROOT, paper_id, annotation_filename())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_session_id():
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id
    return session_id


def add_session_skip(paper_id):
    session_id = get_session_id()
    skipped = SKIPPED_BY_SESSION.setdefault(session_id, set())
    skipped.add(paper_id)


def get_session_skips():
    session_id = get_session_id()
    skipped = SKIPPED_BY_SESSION.get(session_id, set())
    if isinstance(skipped, set):
        return list(skipped)
    return []


@app.route("/")
def index():
    paper_ids = list_paper_ids(check_only_kept=CHECK_ONLY_KEPT)
    if not paper_ids:
        benchmark = (
            "BrokenArXiv"
            if FALSE_MODE
            else "ArXivLean" if LEAN_MODE else "ArXivMath"
        )
        message = f"No {benchmark} items awaiting human review in {PAPER_ROOT}.\n"
        if not (LEAN_MODE):
            message += "Verified items appear here automatically. Refresh this page as verification finishes.\n"
            if not FALSE_MODE:
                message += "For BrokenArXiv items, launch the app with --false.\n"
        message += "Use --check-kept to revisit items you already accepted."
        return app.response_class(message, mimetype="text/plain")
    return redirect(url_for("paper_view", paper_id=paper_ids[0]))


@app.route("/paper/<paper_id>")
def paper_view(paper_id):
    paper_ids = list_paper_ids(check_only_kept=CHECK_ONLY_KEPT)
    if paper_id not in paper_ids:
        abort(404)
    metadata = load_metadata(paper_id)
    if metadata is None:
        abort(404)
    annotation = load_annotation(paper_id)
    source_details = None
    if not (LEAN_MODE):
        source_first = annotation.get("source_first") or {}
        source_details = {
            "basis_summary": annotation.get("basis_summary") or source_first.get("basis_summary"),
            "novelty_type": source_first.get("novelty_type"),
            "importance": source_first.get("importance"),
            "refutation_status": source_first.get("refutation_status"),
            "prior_claim": source_first.get("prior_claim"),
            "new_result": source_first.get("new_result"),
            "evidence": source_first.get("evidence") or [],
            "verification": source_first.get("verification") or {},
            "difficulty_rationale": annotation.get("difficulty_rationale"),
            "plausibility_rationale": annotation.get("plausibility_rationale"),
            "easy_refutation_audit": annotation.get("easy_refutation_audit"),
            "review_binding": source_first.get("review_binding"),
        }
    if LEAN_MODE:
        review = annotation.get("review")
        if isinstance(review, dict):
            annotation = annotation.copy()
            for field in get_field_specs():
                name = field["name"]
                if name in review and review.get(name) not in {"", None}:
                    annotation[name] = review.get(name)
    index = paper_ids.index(paper_id)
    next_id = paper_ids[index + 1] if index + 1 < len(paper_ids) else None
    prev_id = paper_ids[index - 1] if index > 0 else None
    return render_template(
        "paper.html",
        paper_id=paper_id,
        metadata=metadata,
        annotation=annotation,
        next_id=next_id,
        prev_id=prev_id,
        position=index + 1,
        total=len(paper_ids),
        field_specs=get_field_specs(),
        source_details=source_details,
        false_mode=FALSE_MODE,
    )


@app.route("/paper/<paper_id>/annotate", methods=["POST"])
def annotate(paper_id):
    if load_metadata(paper_id) is None:
        abort(404)
    status = request.form.get("status", "").strip().lower()
    if status not in {"keep", "discard"}:
        abort(400, "Choose keep or discard.")
    paper_ids_before = list_paper_ids(check_only_kept=CHECK_ONLY_KEPT)
    index = paper_ids_before.index(paper_id) if paper_id in paper_ids_before else -1
    annotation = load_annotation(paper_id)
    if FALSE_MODE:
        binding = (annotation.get("source_first") or {}).get("review_binding")
        if not binding or request.form.get("review_binding") != binding:
            abort(409, "The item changed since it was displayed. Reload before reviewing.")
        try:
            if sha256_file(os.path.join(PAPER_ROOT, paper_id, FALSE_INVESTIGATION_FILENAME)) != binding:
                abort(409, "Generation or verification changed. Reload before reviewing.")
        except (OSError, ValueError):
            abort(409, "Source stage records are unavailable. Complete verification before reviewing.")
    review_fields = {}
    if LEAN_MODE:
        review_fields = {field["name"]: (request.form.get(field["name"]) or "").strip() for field in get_field_specs()}
    annotation["review"] = {"status": status, **review_fields, "updated_at": utc_now()}
    if FALSE_MODE:
        annotation["review"]["binding"] = binding
    if not (LEAN_MODE):
        annotation["keep"] = status == "keep"
        annotation["stage"] = "human_accepted" if annotation["keep"] else "human_rejected"
    save_annotation(paper_id, annotation)
    paper_ids_after = list_paper_ids(check_only_kept=CHECK_ONLY_KEPT)
    if paper_ids_after:
        if index >= 0:
            for candidate in paper_ids_before[index + 1 :]:
                if candidate in paper_ids_after:
                    return redirect(url_for("paper_view", paper_id=candidate))
        return redirect(url_for("paper_view", paper_id=paper_ids_after[0]))
    return redirect(url_for("done"))


@app.route("/paper/<paper_id>/skip")
def skip_paper(paper_id):
    if load_metadata(paper_id) is None:
        abort(404)
    paper_ids = list_paper_ids(check_only_kept=CHECK_ONLY_KEPT)
    if paper_id not in paper_ids:
        abort(404)
    index = paper_ids.index(paper_id)
    add_session_skip(paper_id)
    skip_ids = get_session_skips()
    remaining_ids = [pid for pid in paper_ids if pid not in skip_ids]
    if not remaining_ids:
        return redirect(url_for("done"))
    for candidate in paper_ids[index + 1 :]:
        if candidate in remaining_ids:
            return redirect(url_for("paper_view", paper_id=candidate))
    return redirect(url_for("paper_view", paper_id=remaining_ids[0]))


@app.route("/done")
def done():
    skipped = set(get_session_skips())
    pending = [paper_id for paper_id in list_paper_ids(check_only_kept=CHECK_ONLY_KEPT) if paper_id not in skipped]
    if pending:
        return redirect(url_for("paper_view", paper_id=pending[0]))
    return app.response_class(
        "All currently available items have been reviewed or skipped. Refresh this page to pick up newly verified items.",
        mimetype="text/plain",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review generated arXiv benchmark items.")
    parser.add_argument(
        "--check-kept",
        action="store_true",
        help="Include papers previously kept in human review.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--false", action="store_true", help="Use the false-statement pipeline metadata.")
    modes.add_argument("--lean", action="store_true", help="Review kept Lean extraction candidates.")
    parser.add_argument("--paper-root", default=None, help="Override the paper directory.")
    parser.add_argument(
        "--annotation-filename",
        default=None,
        help="Override the annotation filename used inside each paper directory.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on.")
    args = parser.parse_args()
    if args.paper_root:
        PAPER_ROOT = os.path.abspath(args.paper_root)
    if args.annotation_filename:
        if (
            args.annotation_filename in {".", ".."}
            or "\x00" in args.annotation_filename
            or os.path.basename(args.annotation_filename) != args.annotation_filename
        ):
            parser.error("--annotation-filename must be a filename, not a path")
        ANNOTATION_FILENAME_OVERRIDE = args.annotation_filename
    LEAN_MODE = args.lean
    CHECK_ONLY_KEPT = args.check_kept
    FALSE_MODE = args.false
    app.run(debug=True, port=args.port)
