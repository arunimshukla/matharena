#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

from matharena.arxivbench_utils import list_paper_ids, load_annotation, load_metadata
from matharena.arxivmath_audit import audit_paper
from matharena.arxivmath_source import (
    SCHEMA_VERSION,
    INVESTIGATION_FILENAME,
    SourceStateError,
    paper_source_unavailable,
    require_current_abstract_acceptances,
)


def is_accepted(annotation):
    if annotation.get("source_first_schema_version") != SCHEMA_VERSION:
        return False
    review = annotation.get("review") or {}
    verification = (annotation.get("source_first") or {}).get("verification") or {}
    return (
        annotation.get("keep") is True
        and review.get("status") == "keep"
        and verification.get("status") == "passed"
        and verification.get("passed") is True
    )


def get_reviewed_pair(annotation):
    question, answer = annotation.get("question"), annotation.get("answer")
    if not question or not answer:
        return None, None
    return question.strip(), answer.strip()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_export_tree(out_dir, accepted):
    problems_dir = os.path.join(out_dir, "problems")
    ensure_dir(problems_dir)

    answers_path = os.path.join(out_dir, "answers.csv")
    source_path = os.path.join(out_dir, "source.csv")
    source_meta_path = os.path.join(out_dir, "source_metadata.csv")
    with (
        open(answers_path, "w", encoding="utf-8", newline="") as answers_file,
        open(source_path, "w", encoding="utf-8", newline="") as source_file,
        open(source_meta_path, "w", encoding="utf-8", newline="") as source_meta_file,
    ):
        answers_writer = csv.writer(answers_file, lineterminator="\n")
        source_writer = csv.writer(source_file, lineterminator="\n")
        source_meta_writer = csv.writer(source_meta_file, lineterminator="\n")
        answers_writer.writerow(["id", "answer"])
        source_writer.writerow(["id", "source"])
        source_meta_writer.writerow(["id", "title", "authors"])
        for idx, (paper_id, question, answer, metadata) in enumerate(accepted, start=1):
            write_text(os.path.join(problems_dir, f"{idx}.tex"), question)
            answers_writer.writerow([idx, answer])
            title = metadata.get("title") or ""
            authors = metadata.get("authors") or []
            author_names = []
            for author in authors:
                if not isinstance(author, dict):
                    continue
                forenames = (author.get("forenames") or "").strip()
                keyname = (author.get("keyname") or "").strip()
                full_name = " ".join([part for part in [forenames, keyname] if part])
                if full_name:
                    author_names.append(full_name)
            source_writer.writerow([idx, paper_id])
            source_meta_writer.writerow([idx, title, "; ".join(author_names)])


def validate_export_tree(out_dir, expected_count):
    root = Path(out_dir)
    expected_problem_names = {f"{idx}.tex" for idx in range(1, expected_count + 1)}
    actual_problem_names = {path.name for path in (root / "problems").glob("*.tex") if path.is_file()}
    if actual_problem_names != expected_problem_names:
        raise RuntimeError("staged problem files do not match accepted question count")
    for filename in ("answers.csv", "source.csv", "source_metadata.csv"):
        with (root / filename).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) != expected_count + 1:
            raise RuntimeError(f"staged {filename} row count does not match accepted question count")
        if [row[0] for row in rows[1:]] != [str(idx) for idx in range(1, expected_count + 1)]:
            raise RuntimeError(f"staged {filename} ids are not contiguous")


def replace_export_tree(staged_dir, out_dir):
    staged = Path(staged_dir)
    destination = Path(out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def main():
    parser = argparse.ArgumentParser(description="Export accepted ArXivMath questions.")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--allow-source-unavailable", action="store_true")
    args = parser.parse_args()

    paper_ids = list_paper_ids(args.paper_root)
    try:
        selected, screen_summary = require_current_abstract_acceptances(args.paper_root, paper_ids=paper_ids)
    except SourceStateError as exc:
        print(f"Source release audit blocked: {exc}")
        return 1
    screen_accepted_ids = set(selected)
    print(f"Abstract screen: {len(selected)} accepted from {screen_summary['paper_count']} screened papers.")
    audit_reports = [
        audit_paper(
            Path(args.paper_root) / paper_id,
            source_cache=args.source_cache,
            investigation_filename=INVESTIGATION_FILENAME,
            final_filename="llm_annotation.json",
            allow_source_unavailable=args.allow_source_unavailable,
        )
        for paper_id in selected
    ]
    blocked = [report for report in audit_reports if not report["ready"]]
    if blocked:
        print(f"Source release audit blocked {len(blocked)} papers; refusing export.")
        for report in blocked[:20]:
            print(f"  {report['paper_id']}: {', '.join(report['blockers'])}")
        if len(blocked) > 20:
            print(f"  ... and {len(blocked) - 20} more")
        return 1
    accepted = []
    skipped_missing_review = []
    for paper_id in paper_ids:
        if args.allow_source_unavailable and paper_source_unavailable(Path(args.paper_root) / paper_id):
            continue
        annotation = load_annotation(args.paper_root, paper_id)
        if is_accepted(annotation):
            if paper_id not in screen_accepted_ids:
                print(
                    f"Source-first release audit blocked {paper_id}: "
                    "accepted annotation was rejected by the current abstract screen."
                )
                return 1
        if not is_accepted(annotation):
            continue
        question, answer = get_reviewed_pair(annotation)
        if not question or not answer:
            skipped_missing_review.append(paper_id)
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        if not isinstance(metadata, dict) or not metadata:
            skipped_missing_review.append(paper_id)
            continue
        accepted.append((paper_id, question, answer, metadata))

    if skipped_missing_review:
        print(
            "Accepted papers are missing reviewed question/answer or metadata: "
            + ", ".join(sorted(skipped_missing_review))
        )
        return 1
    if not accepted:
        print("No accepted papers found.")
        return 1

    destination = Path(args.out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staged-", dir=destination.parent))
    try:
        write_export_tree(staged_dir, accepted)
        validate_export_tree(staged_dir, len(accepted))
        replace_export_tree(staged_dir, destination)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)

    print(f"Exported {len(accepted)} questions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
