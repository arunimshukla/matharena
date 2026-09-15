#!/usr/bin/env python3
"""Export source-verified, human-reviewed BrokenArXiv items."""
import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import yaml

from matharena.arxivbench_utils import list_paper_ids
from matharena.arxivmath_audit import audit_paper
from matharena.arxivmath_source import (
    FALSE_INVESTIGATION_FILENAME,
    FALSE_FINAL_FILENAME,
    SourceStateError,
    atomic_write_json,
    load_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", required=True, help="Competition date YYYY-MM-DD.")
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()
    from datetime import date

    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error("--date must be YYYY-MM-DD")
    if args.max_papers is not None and args.max_papers < 1:
        parser.error("--max-papers must be positive")
    out = Path(args.out_dir)
    if out.exists():
        parser.error("export destination already exists; choose a new directory")
    root = Path(args.paper_root)
    try:
        ids = list_paper_ids(str(root))
        if args.max_papers is not None:
            ids = ids[: args.max_papers]
        accepted, reports, seen = [], [], set()
        for paper_id in ids:
            paper = root / paper_id
            annotation = load_json(paper / FALSE_FINAL_FILENAME, {})
            if annotation.get("stage") != "human_accepted":
                continue
            report = audit_paper(paper, source_cache=args.source_cache, false_mode=True, allow_source_unavailable=True)
            reports.append(report)
            if not report["ready"]:
                raise SourceStateError(f"{paper_id}: " + ", ".join(report["blockers"]))
            state = load_json(paper / FALSE_INVESTIGATION_FILENAME)
            statement = " ".join(state["investigation"]["false_statement"].split())
            if statement in seen:
                raise SourceStateError("duplicate false statements; reject duplicates before export")
            seen.add(statement)
            accepted.append((paper_id, load_json(paper / "metadata.json"), state, annotation["review"]))
        if not accepted:
            raise SourceStateError("no verified, human-accepted items to export")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Refusing export: {exc}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for directory in ("problems", "original", "refutations"):
            (temporary / directory).mkdir()
        source_rows, metadata_rows = [["id", "source"]], [["id", "title", "authors"]]
        grading, provenance = [], []
        for index, (paper_id, metadata, state, review) in enumerate(accepted, 1):
            item, verification = state["investigation"], state["verification"]
            (temporary / "problems" / f"{index}.tex").write_text(item["false_statement"] + "\n", encoding="utf-8")
            # The existing behavioral judge reads the true reference from original/.
            (temporary / "original" / f"{index}.tex").write_text(item["true_statement"] + "\n", encoding="utf-8")
            atomic_write_json(
                temporary / "refutations" / f"{index}.json",
                {
                    "reference_refutation": item["falsity_explanation"],
                    "item": item,
                    "verification": verification,
                },
            )
            source_rows.append([index, state["arxiv_id"]])
            authors = [
                " ".join(str(author.get(part) or "").strip() for part in ("forenames", "keyname")).strip()
                for author in (metadata.get("authors") or [])
            ]
            metadata_rows.append([index, metadata.get("title", ""), "; ".join(filter(None, authors))])
            grading.append({"id": str(index), "points": 3, "scheme": "", "ground_truth_proofs": []})
            provenance.append({"id": str(index), "paper_id": paper_id, "state": state, "review": review})
        for filename, rows in (("source.csv", source_rows), ("source_metadata.csv", metadata_rows)):
            with (temporary / filename).open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)
        atomic_write_json(temporary / "grading_scheme.json", grading)
        atomic_write_json(
            temporary / "generation_manifest.json",
            {
                "pipeline": "brokenarxiv",
                "max_papers": args.max_papers,
                "exported_papers": len(accepted),
                "audit": reports,
                "items": provenance,
            },
        )
        (temporary / "competition.yaml").write_text(
            yaml.safe_dump(
                {
                    "instruction": "Try to generate a proof for the following statement:\n",
                    "strict_parsing": False,
                    "n_problems": len(accepted),
                    "date": args.date,
                    "final_answer": False,
                    "dataset_path": str(out),
                    "judge_configs": ["judges/arxiv_judge_gemini_38_flash"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Exported {len(accepted)} BrokenArXiv items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
