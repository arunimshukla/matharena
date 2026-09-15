#!/usr/bin/env python3
"""Audit the abstract-screened, single-question ArXivMath pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matharena.arxivbench_utils import list_paper_ids
from matharena.arxivmath_audit import audit_paper
from matharena.arxivmath_source import (
    add_source_mode_argument,
    configure_source_mode,
    ABSTRACT_SCREEN_FILENAME,
    FINAL_ANNOTATION_FILENAME,
    INVESTIGATION_FILENAME,
    SourceStateError,
    require_current_abstract_acceptances,
    utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source-first ArXivMath release gates.")
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--abstract-screen-filename", default=ABSTRACT_SCREEN_FILENAME)
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--investigation-filename", default=INVESTIGATION_FILENAME)
    parser.add_argument("--final-filename", default=FINAL_ANNOTATION_FILENAME)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-source-unavailable", action="store_true")
    add_source_mode_argument(parser)
    args = parser.parse_args()
    configure_source_mode(args)

    paper_root = Path(args.paper_root)
    paper_ids = list_paper_ids(str(paper_root))
    if args.max_papers is not None:
        paper_ids = paper_ids[: args.max_papers]
    try:
        paper_ids, screen_summary = require_current_abstract_acceptances(
            paper_root,
            paper_ids=paper_ids,
            screen_filename=args.abstract_screen_filename,
        )
    except SourceStateError as exc:
        print(f"Release audit blocked: {exc}")
        return 1
    reports = [
        audit_paper(
            paper_root / paper_id,
            source_cache=args.source_cache,
            screen_filename=args.abstract_screen_filename,
            investigation_filename=args.investigation_filename,
            final_filename=args.final_filename,
            allow_source_unavailable=args.allow_source_unavailable,
            false_mode=args.false,
        )
        for paper_id in paper_ids
    ]
    blocked = [report for report in reports if not report["ready"]]
    warnings = [report for report in reports if report["warnings"]]
    generated = [report for report in reports if report.get("question_generated")]
    payload = {
        "schema_version": 2,
        "created_at": utc_now(),
        "paper_root": str(paper_root),
        "summary": {
            "papers": len(reports),
            "ready": len(reports) - len(blocked),
            "blocked": len(blocked),
            "with_warnings": len(warnings),
            "questions_generated": len(generated),
            "abstract_screened_papers": screen_summary["paper_count"],
            "abstract_accepted_papers": len(screen_summary["accepted"]),
        },
        "papers": reports,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
