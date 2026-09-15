#!/usr/bin/env python3
import argparse
from datetime import datetime

from tqdm import tqdm

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    ensure_ocr_batch,
    extract_json,
    get_latest_fields,
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


def should_review(annotation, overwrite=False, key="full_text_review"):
    if annotation.get("keep") is not True:
        return False
    if not overwrite and key in annotation:
        return False
    review = annotation.get("review") or {}
    return not review or review.get("status") == "keep"


def main():
    parser = argparse.ArgumentParser(description="Review Lean annotations using OCR.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to process.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--redo-ocr", action="store_true", help="Force OCR even if cached markdown exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full-text review results.")
    parser.add_argument("--key", default="full_text_review", help="Annotation key to store the review under.")
    parser.add_argument("--enable-web-search", action="store_true", help="Enable web search for additional context.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR and full text injection.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--lean", action="store_true", help="Use the Lean abstract-candidate pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    args = parser.parse_args()
    if args.lean and not args.prompt:
        parser.error("--lean requires an explicit --prompt; see arxivmath/scripts/create_lean_abstract.sh.")

    prompt_path = args.prompt
    annotation_filename = args.annotation_filename or (
        LEAN_ANNOTATION_FILENAME
    )
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    if args.enable_web_search:
        if model_config.get("api") == "google":
            model_config["tools"] = [(None, {"google_search": {}})]
            model_config["use_gdm_tools"] = True
            model_config["max_tool_calls"] = 50
        else:
            model_config["tools"] = [(None, {"type": "web_search"})]
    client = APIClient(**model_config)

    discarded = []
    updated = []
    kept = []
    total_cost = 0.0

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[: args.max_papers]
    review_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if should_review(
            annotation,
            overwrite=args.overwrite,
            key=args.key,
        ):
            review_ids.append(paper_id)

    query_inputs = []
    for paper_id in tqdm(review_ids):
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        question = answer = statement = formalized_statement = ""
        if args.key != "solid_authors":
            if args.lean:
                statement = (annotation.get("statement") or "").strip()
                formalized_statement = (annotation.get("formalized_statement") or "").strip()
                if not statement:
                    continue
                question = statement
        metadata = load_metadata(args.paper_root, paper_id)
        query_inputs.append(
            (
                paper_id,
                metadata,
                question,
                answer,
                formalized_statement,
                statement,
            )
        )

        if args.limit and len(query_inputs) >= args.limit:
            break

    if not query_inputs:
        print("No papers need review.")
        return

    full_texts = {}
    if not args.skip_ocr and args.key != "solid_authors":
        full_texts = ensure_ocr_batch(
            [paper_id for paper_id, *_ in query_inputs],
            redo=args.redo_ocr,
        )
        missing_text_ids = {paper_id for paper_id, *_ in query_inputs if paper_id not in full_texts}
        for paper_id in sorted(missing_text_ids):
            annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
            annotation["keep"] = False
            save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
            discarded.append(paper_id)
        query_inputs = [query_input for query_input in query_inputs if query_input[0] not in missing_text_ids]
        if not query_inputs:
            return

    queries = []
    query_paper_ids = []
    for (
        paper_id,
        metadata,
        question,
        answer,
        formalized_statement,
        statement,
    ) in query_inputs:
        prompt = prompt_template.format(
            question=question,
            answer=answer,
            original_statement=statement,
            formalized_statement=formalized_statement,
            statement=statement,
            full_text=full_texts.get(paper_id, ""),
            title=metadata.get("title") or "",
            authors=", ".join([f"{author['forenames']} {author['keyname']}" for author in metadata.get("authors", [])]),
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)

    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        if idx >= len(query_paper_ids):
            continue
        paper_id = query_paper_ids[idx]
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        action = None
        keep_value = None
        if isinstance(parsed, dict):
            action = parsed.get("action")
            keep_value = parsed.get("keep", True)

        review_record = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if isinstance(parsed, dict):
            review_record["parsed"] = parsed
            if "rationale" in parsed:
                review_record["rationale"] = parsed.get("rationale")
        if action:
            review_record["action"] = action

        review = annotation.get("review") or {}
        if action == "discard" or (action is None and '"action": "discard"' in response) or keep_value is False:
            review["status"] = "discard"
            review["updated_at"] = review_record["updated_at"]
            annotation["keep"] = False
            discarded.append(paper_id)
        elif action == "edit":
            if args.lean:
                edited_statement = parsed.get("statement") if isinstance(parsed, dict) else None
                if edited_statement and str(edited_statement).strip():
                    review["statement"] = str(edited_statement).strip()
                    annotation["statement"] = review["statement"]
                    review["updated_at"] = review_record["updated_at"]
                    review["status"] = "keep"
                    annotation["keep"] = True
                    updated.append(paper_id)
                else:
                    kept.append(paper_id)
        else:
            annotation["keep"] = True
            kept.append(paper_id)

        annotation["review"] = review
        annotation[args.key] = review_record
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += review_record["cost"]

    print(f"Full-text review complete. Total cost: ${total_cost:.6f}")
    print(f"Discarded ({len(discarded)}): {', '.join(discarded)}")
    print(f"Updated ({len(updated)}): {', '.join(updated)}")
    print(f"Kept ({len(kept)}): {', '.join(kept)}")


if __name__ == "__main__":
    main()
