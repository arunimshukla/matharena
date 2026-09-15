#!/usr/bin/env python3
import argparse

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    extract_json,
    list_paper_ids,
    load_annotation,
    load_metadata,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
    save_annotation,
)
from matharena.utils import normalize_conversation


LEAN_ANNOTATION_FILENAME = "metadata_lean_abstract.json"
LEAN_DEFAULT_PROMPT = "arxivmath/prompts/lean/extract_lean_abstract.md"
ARXIV_NONEXCLUSIVE_LICENSE_URL = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def needs_annotation(annotation, overwrite=False, lean_mode=False):
    if not (lean_mode):
        raise ValueError("Select Lean generation.")
    if overwrite:
        return True
    if lean_mode:
        keep = annotation.get("keep")
        if keep is None:
            return True
        if keep is True and not annotation.get("statement"):
            return True
        return False


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def should_skip_license(metadata, skip_arxiv_license=False):
    if not skip_arxiv_license:
        return False
    return (metadata.get("license") or "").strip() == ARXIV_NONEXCLUSIVE_LICENSE_URL


def main():
    parser = argparse.ArgumentParser(description="Generate Lean annotations.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to query.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--lean", action="store_true", help="Use the Lean abstract-candidate pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations.")
    parser.add_argument(
        "--skip-arxiv-license",
        action="store_true",
        help="Skip papers using arXiv's non-exclusive distribution-only license.",
    )
    args = parser.parse_args()

    if args.lean:
        prompt_path = args.prompt or LEAN_DEFAULT_PROMPT
        annotation_filename = args.annotation_filename or LEAN_ANNOTATION_FILENAME
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    client = APIClient(**model_config)

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[: args.max_papers]
    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if not needs_annotation(
            annotation,
            overwrite=args.overwrite,
            lean_mode=args.lean,
        ):
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        if should_skip_license(metadata, skip_arxiv_license=args.skip_arxiv_license):
            continue
        prompt = prompt_template.format(
            title=(metadata.get("title") or "").strip(),
            abstract=(metadata.get("abstract") or "").strip(),
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        return

    total_cost = 0.0
    kept_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        paper_id = query_paper_ids[idx]
        metadata = load_metadata(args.paper_root, paper_id)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
        }
        annotation.update(
            {
                "title": metadata.get("title") or "",
                "abstract": metadata.get("abstract") or "",
                "source_mode": "abstract",
            }
        )
        keep_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            annotation["parsed"] = parsed
            if args.lean:
                if parsed.get("statement"):
                    annotation["statement"] = str(parsed["statement"]).strip()
                if parsed.get("rationale"):
                    annotation["rationale"] = str(parsed["rationale"]).strip()
        if keep_value is not None:
            annotation["keep"] = keep_value
            if keep_value is True:
                kept_ids.append(paper_id)
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += annotation["cost"]

    print(f"Completed {len(queries)} queries. Total cost: ${total_cost:.6f}")
    print(f"Kept {len(kept_ids)} papers: {', '.join(kept_ids)}")


if __name__ == "__main__":
    main()
