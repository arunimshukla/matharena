<div align="center">
    <h1><img height="150px" src="../images/matharena_icon.png" alt="MathArena"><br>MathArena</h1>

  <a href="https://www.python.org/">
<img alt="Build" src="https://img.shields.io/badge/Python-3.12-1f425f.svg?color=blue">
  </a>
  <a href="https://opensource.org/licenses/MIT">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
  <a href="https://huggingface.co/MathArena">
<img alt="MathArena Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Matharena-ffc107?color=ffc107&logoColor=white">
  </a>
</div>

## Overview

This README covers the ArXivMath, BrokenArXiv, and ArXivLean curation workflow: downloading a new month of papers, running the automated extraction pipeline, manually reviewing candidate questions, exporting the accepted questions, and then evaluating models on the resulting competitions.

BrokenArXiv now has a separate [source-based Codex generation README](README_brokenarxiv.md), including the old-to-new comparison, difficult false-claim selection policy, review, and export instructions.

## Prerequisites

Use a separate paper workspace for each month and retain previous workspaces. Set `ARXIVMATH_PAPER_ROOT` or `ARXIV_FALSE_PAPER_ROOT` when using the shell helpers, or pass `--paper-root` to the Python stages and review app.

ArXivMath is abstract-first and source-verified. Codex first makes a binary accept/reject decision from the title and abstract only; it does not generate a question or classify the result at this stage. Exact-version TeX is downloaded only for accepted papers. One source investigation then either rejects the paper or creates exactly one question plus a short description of the part of the paper on which it is based. The source stages do not use PDF OCR. Set `ARXIV_SOURCE_CACHE` to storage with enough capacity; on Ada the helper uses `/userdata/$USER/matharena-arxiv-source` or `/scratch/userdata/$USER/matharena-arxiv-source`, whichever mount is available. Curation harness workspaces and API fallback request logs default to `<source-cache>/request-logs`, and can be redirected independently with `MATHARENA_REQUEST_LOG_DIR`.

All three ArXivMath model stages default to `openai/gpt-6-astra`: GPT-6 Astra at max reasoning effort through the normal `harness_wrapper` integration. This is the same model config used for evaluation, including its pinned Codex CLI version, context window and subscription authentication. The source Python commands have the same default as the shell helper.

Use the existing harness setup described in [README_harness.md](README_harness.md), including Docker and the shared harness image. Authenticate on the host before a future run:

```bash
codex login --device-auth
codex login status
```

`HarnessSolver` manages the configured CLI version and forwards requests through its host-side proxy. There is no separate curation Codex executable setting or authentication implementation. Each query has a fresh workspace and private home. The normal model config remains the place to choose harness version, authentication and reasoning effort.

ArXivLean uses OCR for their full-text stages. The new ArXivMath and BrokenArXiv source workflows do not require OCR:

```bash
uv run vllm serve zai-org/GLM-OCR \
  --port 8000 \
  --data-parallel-size 2 \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --performance-mode throughput \
  --optimization-level 3 \
  --renderer-num-workers 4 \
  --mm-processor-cache-gb 0 \
  --disable-uvicorn-access-log \
  --uvicorn-log-level warning
```

This is tuned for the local 2x RTX 2080 Ti box with vLLM 0.19.x. It is not needed for `arxivmath/scripts/create.sh` or the new `arxivmath/scripts/create_false.sh`.

## Adding a New Month

Download the metadata for the target month, choose a persistent source cache, and run the extraction pipelines. For example, for August 2026:

```bash
uv run python arxivmath/scripts/shared/download_arxiv_math.py \
  --from 2026-08-01 --until 2026-08-31
export ARXIV_SOURCE_CACHE=/scratch/userdata/$USER/matharena-arxiv-source
nice -n 7 bash arxivmath/scripts/create.sh screen

# Inspect the reported accepted-paper count before authorizing full-source usage.
nice -n 7 bash arxivmath/scripts/create.sh prepare

# These remain separate pipelines.
bash arxivmath/scripts/create_false.sh screen # BrokenArXiv: abstract triage only.
# Later, after reviewing the selection policy/count:
bash arxivmath/scripts/create_false.sh prepare
```

Both ArXivMath and BrokenArXiv abstract screening run at most 25 papers per batch, with a 31-second pause after each finished batch before starting the next. This pacing keeps normal screening traffic below the observed 60-new-WebSocket-connections-per-minute limit. It applies automatically to the helpers and direct screening commands. Investigation and verification have no added pacing.

`create.sh` is the abstract-screened, single-question ArXivMath pipeline. Its default `screen` mode performs only binary abstract triage and cannot start source downloads or source-level Codex calls. Explicit `prepare` reruns triage incrementally, processes only the currently accepted papers, verifies each generated question independently, and materializes the questions for human review. Verification writes each review item immediately; no separate finalization command is needed.

The old ArXivMath and BrokenArXiv abstract-to-item/OCR generation paths, BrokenArXiv perturbation prompts, training-generation scripts, and legacy annotation export fallbacks have been removed. The shared `create_queries.py`, `verify_queries.py`, and `fulltext_review.py` commands now serve only Lean generation. Their OCR helpers and prompts remain in use by those benchmarks. Training generation uses the same source pipeline with a separate `ARXIVMATH_PAPER_ROOT` or `ARXIV_FALSE_PAPER_ROOT`. Existing datasets, competition configs, evaluation code and downloaded paper workspaces are preserved.

The mutable OAI title and abstract fields are used only by triage, and their exact hash is recorded so a metadata change invalidates the old decision. Before downloading source, the pipeline resolves the latest explicit `vN` for every accepted paper through the arXiv Atom API and stores it in `metadata.json`; it never assumes an unversioned ID is `v1`. Authoritative question generation uses only the binary eligibility decision, pinned arXiv ID, and exact-version TeX artifacts.

The pipeline stages are:

1. screen every title and abstract with a resumable max-effort GPT-6 Astra Codex pass whose binary accept/reject decision is authoritative;
2. validate that Codex returned exactly one supported decision without applying any regex or local override;
3. resolve and persist the exact current arXiv revision, then download and safely extract that source version only for accepted papers, recording archive and TeX hashes;
4. recursively inline local TeX includes, remove non-rendered comments while retaining immutable raw-source hashes and line structure, and send the complete cleaned source to a max-effort GPT-6 Astra Codex invocation;
5. make one Codex investigation call per accepted paper, independently checking source-level novelty and producing either a documented rejection or exactly one self-contained, parser-safe question, answer, basis summary, and exact TeX evidence;
6. independently derive and verify the answer from the question and evidence with a fresh max-effort GPT-6 Astra harness invocation that does not see the gold answer;
7. materialize verified questions for one accept/reject human review pass.

There is no candidate ranking or difficulty measurement. The abstract model only accepts or rejects papers; it does not report novelty signals or generate questions. The full-source investigator independently checks first for counterexamples, disproofs, negative answers, competing-conjecture resolutions, and results differing from prior predictions. A supported relationship of this kind must either become the single question or be recorded as not exactly gradable.

Source downloading, investigation, verification and preparation for human review select the currently accepted papers without requiring the entire folder to finish screening. Other screening records are skipped, and matching completed work is reused on subsequent runs. Set these only when intentionally invalidating existing stages:

```bash
export ARXIVMATH_SOURCE_OVERWRITE=1 # Re-run source/model stages.
export ARXIVMATH_SCREEN_OVERWRITE=1 # Re-run abstract triage even when its inputs are unchanged.
export ARXIVMATH_ALLOW_SOURCE_UNAVAILABLE=1 # Explicitly skip papers whose source ingestion failed.
export ARXIVMATH_MAX_PAPERS=100 # Optional end-to-end pilot scope; unset for production.
export ARXIVMATH_INVESTIGATION_MAX_COST=200 # Optional API-equivalent dollar budget.
export ARXIVMATH_VERIFICATION_MAX_COST=50 # Optional API-equivalent dollar budget.
export ARXIVMATH_MAX_SOURCE_TOKENS=750000 # Optional conservative per-paper input ceiling.
```

The source stage never silently truncates an article. Its default ceiling is a conservative estimate of 750,000 input tokens after comment removal. A paper above the ceiling is recorded as `source_oversized`, receives no source-level model call, and is excluded from the benchmark. Increase the environment variable only after confirming that the investigator model has sufficient context capacity.

Unresolved TeX `\input`/`\include` directives fail by default because they invalidate the claim that Codex saw the complete article. For a known non-content include, `ARXIVMATH_ALLOW_UNRESOLVED_INCLUDES=1` is an explicit escape hatch; the release audit retains a warning that should be documented.

Model configs can be overridden without editing a script:

```bash
export ARXIVMATH_SCREEN_CONFIG=openai/gpt-6-astra
export ARXIVMATH_INVESTIGATOR_CONFIG=openai/gpt-6-astra
export ARXIVMATH_VERIFIER_CONFIG=openai/gpt-6-astra
```

All stages reuse `openai/gpt-6-astra`. Verification uses a separate prompt and fresh invocation, with the gold answer withheld; no difference in model, reasoning effort or config is required. There are no stage-specific model YAMLs.

The small adapter in `src/matharena/query_client.py` converts curation batches to the existing `HarnessSolver` interface. CLI management, Docker isolation, subscription/API authentication, retries, response extraction, exact request logs and token costs use the normal harness implementation. The retired `codex_exec_client.py` and its dedicated configs have been removed.

Curation sets `allow_harness: false` to select the harness's tool-free mode and disables automatic fallback to another model. Each stage requests its JSON contract in the prompt and validates the response locally before recording success. Invalid or missing outputs remain failed and are retried on a subsequent invocation. The old CLI-only output-schema files are no longer used.

The normal harness history is retained in `detailed_cost.history`, including the workspace and exact model-request log path. Workspaces default to `<source-cache>/request-logs/curation/<invocation>/p<index>_r0` through the helper's `MATHARENA_REQUEST_LOG_DIR`; direct Python calls default to `logs/harness_workspaces/curation/`. Query indices advance between batches so later batches do not overwrite earlier logs.

The stored `cost` uses the normal harness's token accounting and the model config's input, cached-input and output rates. With subscription authentication this is an API-equivalent estimate, not a subscription charge. Investigation and verification budget limits use this same value; there is no separate curation cost formula or service-tier override.

Source downloading selects papers with a completed, current abstract acceptance and skips other records. It can run while metadata downloads and abstract screening continue. Repeating `download_sources.py` picks up newly accepted papers and reuses cached sources; a completed screen for the entire folder is not required.

### Bulk arXiv source archives

For a large month, source bundles may be supplied as a local cache. Production import accepts a bundle member only when its member filename itself contains an explicit `vN` revision:

```bash
uv run python arxivmath/scripts/source/download_sources.py \
  --paper-root arxivmath/paper \
  --source-cache "$ARXIV_SOURCE_CACHE" \
  --bundle /scratch/userdata/$USER/arxiv-src/versioned-source-bundle.tar
```

The official S3 source bundles normally use unversioned member names, and their manifest does not prove a per-paper revision. Those members are therefore not labelled as `v1`; they fall back to the rate-limited exact-version arXiv `/src/<id>v<version>` endpoint. Repeat `--bundle` only for a source bundle whose members carry explicit revisions.

## Manual Review

ArXivMath has one mandatory human pass over independently verified questions. Verification puts each finished item into the review queue immediately. The GUI also refreshes review items directly from saved generation and verification records, so missing review files or interrupted runs require no preparation command. Open the GUI during verification and refresh it to see newly completed items:

```bash
uv run python arxivmath/app.py \
  --paper-root arxivmath/paper
```

Source-first questions are accept/reject only. If a question needs rewriting, regenerate the investigation and rerun independent verification; editing it after verification would invalidate its hashes.

Before export, run the release audit:

```bash
uv run python arxivmath/scripts/source/audit_pipeline.py \
  --paper-root arxivmath/paper \
  --source-cache "$ARXIV_SOURCE_CACHE" \
  --output arxivmath/source-audit.json
```

The audit blocks release for stale abstract screening, incomplete or changed source artifacts, stale cleaned-source manifests, stale source evidence, missing independent verification, mismatched final annotations, missing human decisions, and unavailable source. Oversized-source exclusions are audited as explicit final tombstones rather than partial reviews. An unavailable-source override is explicit and should be documented in the release notes.

The app always uses source-based, accept/reject review for ArXivMath and BrokenArXiv. Use `uv run python arxivmath/app.py --false` for BrokenArXiv; see [its README](README_brokenarxiv.md). Add `--check-kept` to revisit kept items. Lean review retain their editable fields:

```bash
uv run python arxivmath/app.py --false --check-kept # Revisit kept BrokenArXiv items.
```

The UI shows the statements and supporting material before you keep or discard each item. During the manual pass, remove:

1. guessable questions,
2. trivial questions,
3. questions with non-unique or context-dependent answers,
4. questions whose answers are too hard to parse robustly.

Do not commit or redistribute the downloaded source cache. The repository stores only source provenance and hashes needed to reproduce the extraction; generated benchmark records remain subject to the normal release review.

## Exporting the Accepted Questions

Once the review is done, export the accepted questions:

```bash
uv run python arxivmath/scripts/arxiv/export_accepted_questions.py \
  --source-cache "$ARXIV_SOURCE_CACHE" \
  --out-dir data/arxiv/august
uv run python arxivmath/scripts/broken/export_false_proofs.py \
  --source-cache "$ARXIV_SOURCE_CACHE" \
  --out-dir data/arxiv_false/august --date 2026-08-01
```

The ArXivMath exporter always reruns the release audit and refuses to export while any paper is blocked. If a release deliberately uses `--allow-source-unavailable`, pass the same flag to the exporter and record the exception in the release notes.

For BrokenArXiv, review/copy the emitted `competition.yaml` into `configs/competitions/arxiv_false/`. See [README_brokenarxiv.md](README_brokenarxiv.md).

For ArXivMath, copy the previous month's competition config to `configs/competitions/arxiv/<month>.yaml` and update it for the new month.

At minimum, update `n_problems`, `date`, and `dataset_path`. Also add the new month to `website/flaskr/static/data/competitions.json` if it should appear on the website.

## Running Models on the New Month

Run models as usual with the normal competition runner:

```bash
uv run python scripts/run.py --comp arxiv/august --models openai/gpt-5
```

For BrokenArXiv, remember that a separate judging pass is required:

```bash
uv run python scripts/judge/judge.py --comp arxiv_false/august
```

If you evaluate agents that rely on OCR or paper-reading tools, keep the DeepSeek-OCR server running while they execute.

## Reviewing Model Outputs and Cleaning the Dataset

After running models, inspect the outputs carefully. I usually:
- inspect all problems that every model got wrong for noise or extraction errors,
- inspect partially solved problems for parsing or ambiguity issues,
- inspect universally solved problems for hidden triviality.

If a problem should be removed, use `nuke_problems.py`, for example:

```bash
uv run python scripts/curation/nuke_problems.py arxiv/august 5
```

ArXivMath answers are often harder to parse than final answers from olympiad-style contests. After a run, open the local inspection app to review parser mistakes and manually override results where needed:

```bash
uv run python app/app.py --comp arxiv/august
```

If you patch the parser or grader, rerun:

```bash
uv run python scripts/regrade.py --comps arxiv/august
```
