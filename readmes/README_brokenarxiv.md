# BrokenArXiv generation

BrokenArXiv (`arxiv_false`) uses the existing source-based ArXivMath stages with a `--false` mode. It targets exceptionally difficult, plausible false claims that a paper refutes, especially documented conjectures and negative answers to open questions. **All three model stages use GPT-6 Astra through Codex, with medium reasoning for screening and high reasoning for source investigation and verification by default.**

The initial pilot screened 100 papers and investigated the five accepted sources. Generation and verification also have offline tests with mocked model responses and synthetic sources. Generated candidates still require verification and human review.

## What changed from the old pipelines

Before the ArXivMath overhaul (`5413d3081`), both pipelines generated items from titles and abstracts, verified those items, and used OCR full-text passes to repair them and apply author-record, prior-work and AI-generation filters. BrokenArXiv generated a true statement and an invented plausible false perturbation. Its verifier saw the proposed explanation of falsity, and later full-text passes could edit the pair.

The current ArXivMath pipeline screens abstracts first, downloads exact-version TeX only for eligible papers, generates at most one source-grounded question, verifies it in a fresh model invocation, and prepares it for human review. Its source manifests, evidence locations, hashes, incremental processing and release audit provide the infrastructure used here.

| Aspect | Old BrokenArXiv | New BrokenArXiv |
| --- | --- | --- |
| Abstract stage | Generate a statement pair | Binary eligibility decision only |
| Authoritative input | Abstract, followed by OCR repairs | Complete cleaned TeX from a pinned arXiv revision |
| Model | Gemini | GPT-6 Astra through the normal harness config |
| Target | Invented false perturbation | Documented prior mathematical claim refuted by a main contribution |
| Difficulty | Plausibility and informal avoidance of easy perturbations | Source evidence for a research barrier, plus explicit easy-refutation checks |
| Verification | Proposed refutation supplied to verifier | Fresh invocation derives its own refutation from source and the two statements |
| Editing | Full-text and human edits | Accept/reject review; revisions require regeneration and verification |
| Prior work | Separate OCR classifier | Required evidence and checks in both source passes |
| Author/AI classifiers | Extra model passes | Removed; selection uses mathematical evidence |
| Reference material | Exported statements | Both refutations, source evidence and generation records retained |

ArXivMath's exact-answer parser restrictions do not apply. Negative existence results, difficult counterexamples and false classifications can be suitable without a scalar answer, executable witness or formalization.

The solver receives:

> Try to generate a proof for the following statement:

The false mathematical statement is used to measure whether models accept a false premise. The true statement can separately be given with the same proof request to test whether a model can solve the original problem. Each statement stands alone, without the other statement or the reference refutation.

## Selection policy

There are exactly three claim kinds:

1. `disproved_conjecture`: a documented prior conjecture disproved by this paper;
2. `negative_answer`: the false affirmative statement corresponding to a documented prior open question answered negatively here;
3. `refuted_prediction`: a documented mathematical prediction or competing conjecture contradicted by the paper's main result.

The investigator prioritizes a genuine disproven conjecture when one is available. The paper must support the historical claim and its new refutation. An invented claim, arbitrary constant change, unmotivated inequality reversal, or a theorem with an obviously necessary hypothesis removed is not admissible.

There is no acceptance quota or automatic relaxation. Abstract screening can admit promising limitation or sharpness results for full investigation, but the source must establish one of the three relationships above. The abstract does not need complete definitions or an explicit use of the word conjecture.

| Field | Meaning |
| --- | --- |
| `true_statement` | The self-contained true resolution of the original open problem, logically negating the false claim without supplying the witness or proof |
| `false_statement` | A faithful, self-contained restatement of the documented prior claim |
| `falsity_explanation` | A reference refutation checking the witness's hypotheses and failed conclusion |
| `claim_kind`, `prior_claim` | The type of historical claim and its content/attribution |
| `basis_summary`, `importance` | Which main contribution establishes falsity |
| `plausibility_rationale` | Positive mathematical or historical reasons the claim was plausible |
| `difficulty_rationale` | The source-supported research barrier and necessary new insight |
| `easy_refutation_audit` | Concrete simple attacks considered and their outcomes |
| `evidence` | Supporting TeX excerpts with evidence roles and source locations when matched |

`true_statement` and `false_statement` are the new field names throughout generation, verification, review and export. There are no old-field aliases or conversion logic in the new source contract.

Both statements must define all needed objects, quantifiers, parameters and ambiguous conventions without access to the source or to each other. Neither should supply a counterexample, construction, proof hint, paper attribution or conjecture name as a lookup hint. Attribution belongs in the curation records.

The true statement must preserve the original proof challenge. For the integer domination root conjecture, this means asserting the existence of a graph with an integer domination root outside the conjectured set, leaving the graph and root to be found. Giving the 33-vertex graph, its edge list or the root −4 changes discovery into verification. Such details belong in `falsity_explanation` and the source evidence. Objects or parameters already fixed by the original problem remain in the statement. The verifier checks this requirement through its existing statement-support, hypothesis and difficulty checks; the JSON fields are unchanged.

A conjecture the authors merely pose, a failed proof, lack of a known proof, numerical evidence or independence from axioms does not establish falsity. The reference refutation must exhibit a witness or supply a rigorous existence argument, check every hypothesis and negate the conclusion.

## What extremely difficult means

Both generation and verification require positive source support for a substantial research barrier and the new construction or mechanism that overcomes it. A long statement or unfamiliar notation is insufficient.

The investigator explicitly considers small and degenerate examples, boundary regimes, parity, scaling, standard inequalities, elementary specializations and classical obstructions. The verifier separately checks source support for research difficulty. If a short independent refutation works, the item should be rejected. A historically difficult conjecture is insufficient if its extracted restatement accidentally becomes easier to disprove.

These are source-supported model assessments, not measured solve rates or proof certificates. No solver tournament, numerical difficulty score or ranking stage has been added. Real-paper quality and empirical difficulty need to be assessed on the generated items. A model failing to find a refutation does not establish difficulty.

The prior-work check uses the source's account. There is no external literature search or independent guarantee of historical priority.

## Shared implementation

There is no separate BrokenArXiv runner, source-state module or set of model YAMLs.

| Existing component (relative to `arxivmath/`) | BrokenArXiv use |
| --- | --- |
| `scripts/source/screen_abstracts.py --false` | Binary screening with the BrokenArXiv prompt |
| `scripts/source/download_sources.py --false` | Version resolution, safe TeX extraction, manifests and cache |
| `scripts/source/investigate_sources.py --false` | Shared harness-backed generation and resumable investigation records |
| `scripts/source/verify_questions.py --false` | Independent verification and immediate preparation for human review |
| `scripts/source/audit_pipeline.py --false` | Existing release audit with the false-statement contract |
| `scripts/broken/export_false_proofs.py` | Export of reviewed source items and refutations |

The false-statement validation branch is in the existing `src/matharena/arxivmath_source.py`. It validates the different mathematical fields and evidence roles; hashes, TeX preparation, model records and stage execution are shared.

The per-paper records are:

- `metadata.json`: shared paper metadata and pinned `versioned_id`;
- `abstract_screen_false.json`: BrokenArXiv's screening decision;
- `source_ref.json`, `source_ingestion.json`: shared source reference and ingestion status;
- `source_false_investigation.json`: generation and verification in the same record format used by ArXivMath;
- `llm_metadata_false_source.json`: the materialized review item.

The complete cleaned source is sent to generation. Unlike ArXivMath's evidence-window verifier, BrokenArXiv verification also receives the full source: assumptions or prior-work discussion elsewhere in the paper can invalidate a proposed refutation. It sees the two statements but not the generator's proposed refutation, selected evidence, classification or difficulty rationale.

Verification is a fresh invocation of the same GPT-6 configuration, using a separate prompt to derive its own refutation. No distinction in model, family or reasoning effort is required.

The verifier returns `keep`, eight boolean checks (`true_statement_supported`, `false_statement_refuted`, `hypotheses_match`, `self_contained`, `natural_claim`, `main_contribution`, `novelty_supported`, `research_difficult`), `reason`, and `evidence_quotes`. Its refutation is included in `reason`, which is displayed in the review UI and retained in the exported `verification` record. Claim classification and the explicit easy-refutation audit remain part of generation; they are not separate verifier output fields. The verdict must equal the conjunction of the eight checks.

Both accepted generation and verification outputs include excerpts for the result, proof, prior claim, prior work and difficulty. Source locations are optional: when an excerpt matches uniquely (allowing LF versus CRLF line endings), its stored text, hash and offsets refer to that source substring. Otherwise, the original quote and its hash are retained without a location, and the review GUI displays the quote with “Source location unavailable.” Formatting differences, repeated passages and quote-location failures do not reject a paper or block verification, human review or export. The verifier reads the complete source independently; mathematical support and correctness still require model assessment and human review.

Unavailable sources are recorded and skipped automatically. Corrupt or changed source artifacts remain errors. Unresolved includes are rejected for false items because the model needs the complete source. There is no OCR fallback. Oversized papers are excluded without truncation or a source-level model call.

## Running the pipeline

Screening defaults to the existing `openai/gpt-6-astra-medium` config; source investigation and verification default to `openai/gpt-6-astra-high`. Both select `harness: codex`, the pinned CLI version and subscription authentication. These defaults apply to the helper and direct Python stage commands. False mode requires a GPT-6 Codex harness model; automatic model fallback is disabled.

Curation batches go through the existing `HarnessSolver` and vendored `harness_wrapper`, using a small adapter in `src/matharena/query_client.py`. The separate Codex client and its stage-specific YAMLs have been removed. Both ArXivMath pipelines use this shared path for CLI management, Docker isolation, authentication, recovery, logs and costs.

The pipeline selects the existing tool-free mode: the model reasons over the supplied source without shell, browsing or other tools. Generation and verification start in separate workspaces. JSON output contracts are requested in the prompts and checked by the existing local validators; there is no curation-specific CLI schema mechanism.

Use the normal [harness setup](README_harness.md), including Docker, the shared harness image and a saved host Codex subscription login. The wrapper manages the version selected by the model config. No new YAML or separate Codex binary setting is needed.

Run from the repository root:

```bash
# Setup and metadata collection for a future run.
codex login --device-auth
codex login status

uv run python arxivmath/scripts/shared/download_arxiv_math.py \
  --from 2026-08-01 --until 2026-08-31 --outdir arxivmath/paper

export ARXIV_FALSE_PAPER_ROOT=arxivmath/paper
export ARXIV_SOURCE_CACHE=/path/to/persistent/arxiv-source-cache

# Default invocation performs only screening.
bash arxivmath/scripts/create_false.sh screen

# When ready for full-source processing:
bash arxivmath/scripts/create_false.sh prepare

# Open during or after verification; refresh to pick up completed items.
uv run python arxivmath/app.py --false --paper-root "$ARXIV_FALSE_PAPER_ROOT"

# Release audit after review.
bash arxivmath/scripts/create_false.sh audit
```

Both BrokenArXiv and ArXivMath abstract screening run at most 25 papers per batch and wait 31 seconds after each finished batch before starting the next. This keeps normal screening traffic below the observed 60-new-WebSocket-connections-per-minute limit. Pacing applies automatically to both helpers and direct `screen_abstracts.py` calls, with or without `--false`; investigation and verification have no added pacing. Smaller `--batch-size` values still apply. A failed screen leaves that paper pending and returns exit status 1, stopping `prepare`; rerun without `ARXIV_FALSE_SCREEN_OVERWRITE=1` to retry failed screens while reusing completed ones.

`screen` cannot start source downloading or investigation. `prepare` screens incrementally, downloads accepted sources, investigates and verifies. If source investigation returns status 1 for incomplete work, preparation continues by verifying the completed candidates; it still returns a nonzero status at the end to report unfinished work. Each completed verification immediately updates its review item, including when other papers are pending or fail. `audit` makes no model calls or downloads.

Source downloading, investigation, verification and preparation for human review use the currently accepted papers, even while metadata downloading or abstract screening continues for other papers. Papers with rejected, missing, failed, stale or unreadable screening records are skipped. Each invocation takes the current selection and reuses matching completed work; run it again to pick up later acceptances. No `--max-papers` limit or special opt-in is required:

```bash
uv run python arxivmath/scripts/source/download_sources.py --false \
  --paper-root arxivmath/paper \
  --source-cache arxivmath/source_cache
```

Verification and the GUI share the same review-record builder. The GUI also picks up saved verification results automatically when review files are missing or stale. There is no separate finalization script or helper mode. Refresh the page as verification finishes; unchanged human decisions are preserved, and changed generation or verification results require a fresh decision.

The review app automatically uses source-based, accept/reject review for ArXivMath and BrokenArXiv; `--false` selects BrokenArXiv. The obsolete manual-extraction and legacy statement-editing paths have been removed. The UI puts the true and false statements first, side by side on wide screens and stacked on smaller screens. The reference refutation, independent verification checks and rationale, source evidence, difficulty assessment and paper details appear in collapsible sections below. Keep/reject buttons stay visible while scrolling. Add `--check-kept` to the app command to revisit accepted items. Review binds to the investigation record, including verification; a new generation or verification run invalidates the old approval. Text revisions require regeneration and verification. Edits to prompt files do not invalidate saved results or block human review or export. The recorded prompt hashes remain provenance for the actual runs, and the displayed statements remain those actually generated and verified. Explicitly running a model stage after editing its prompt regenerates results under that prompt.

Audit checks source and item hashes, evidence, saved model settings, verification and human decisions. Missing or stale work blocks export. The exporter selects human-accepted items and audits each one. Other papers still being downloaded, screened, investigated, verified or reviewed do not block that export. A standalone audit can still be used to inspect completeness of the entire selected scope.

### Run controls

The helper exposes the shared stages' operational controls:

| Variable | Default / purpose |
| --- | --- |
| `ARXIV_FALSE_PAPER_ROOT` | `arxivmath/paper` |
| `ARXIV_SOURCE_CACHE` | `arxivmath/source_cache`; choose persistent storage as needed |
| `ARXIV_FALSE_MODEL_CONFIG` | Unset; use medium for screening and high for investigation/verification. If set, overrides all three model stages. |
| `MATHARENA_REQUEST_LOG_DIR` | Harness workspace/log root; defaults to `logs/harness_workspaces` for this helper |
| `ARXIV_FALSE_MAX_PAPERS` | Unset; optional sorted metadata-folder scope for a pilot |
| `ARXIV_FALSE_MAX_SOURCE_TOKENS` | `750000`; conservative cleaned-source ceiling |
| `ARXIV_FALSE_BATCH_SIZE` | `2` for investigation and verification |
| `ARXIV_FALSE_INVESTIGATION_MAX_COST` | Unset; soft API-equivalent budget for the generation invocation |
| `ARXIV_FALSE_VERIFICATION_MAX_COST` | Unset; same for verification |
| `ARXIV_FALSE_SCREEN_OVERWRITE` | `0`; set to `1` to rerun screening |
| `ARXIV_FALSE_MODEL_OVERWRITE` | `0`; set to `1` to rerun generation and verification |

The source estimate uses two characters per token. It is an input estimate, not a model-context guarantee; leave room for prompts and output. The shared stage accepts a ceiling of at least 20,000. Both source stages read complete papers, so verification can be expensive.

Costs use the normal harness accounting and rates from the model config; they are API-equivalent estimates, not subscription charges. The shared harness history is saved in `detailed_cost.history`, including exact request-log paths. Each invocation gets a unique workspace directory under `<log-root>/curation/`, and query indices advance across batches. Budgets are checked between batches, can overshoot by a batch and reset per invocation. Partial work or failures return nonzero, stopping the helper. Matching completed work is reused. Run one writer per paper workspace; shared atomic writes support resumption.

To run just one model stage:

```bash
uv run python arxivmath/scripts/source/investigate_sources.py --false \
  --paper-root "$ARXIV_FALSE_PAPER_ROOT" --source-cache "$ARXIV_SOURCE_CACHE" \
  --limit 5 --batch-size 1

uv run python arxivmath/scripts/source/verify_questions.py --false \
  --paper-root "$ARXIV_FALSE_PAPER_ROOT" --source-cache "$ARXIV_SOURCE_CACHE"
```

The helper uses each stage's default: `openai/gpt-6-astra-medium` for screening and `openai/gpt-6-astra-high` for investigation and verification. Use `ARXIV_FALSE_MODEL_CONFIG` to override all three in the helper or `--model-config` to override a direct stage. The shared ArXivMath defaults are unchanged. Shell variables configure the helper; direct Python calls use stage flags. For a separate training workspace, set `ARXIV_FALSE_PAPER_ROOT` to that workspace. No new training-specific licensing or sampling policy is imposed.

### Rerun the existing August batch at medium reasoning

The first 200 metadata folders are the previously screened August batch (15 papers reached source investigation, with 10 accepted final problems). To rerun all three model stages on that same batch from the repository root:

```bash
ARXIV_FALSE_MAX_PAPERS=200 \
ARXIV_FALSE_MODEL_CONFIG=openai/gpt-6-astra-medium \
ARXIV_FALSE_SCREEN_OVERWRITE=1 \
ARXIV_FALSE_MODEL_OVERWRITE=1 \
bash arxivmath/scripts/create_false.sh prepare
```

This explicitly uses medium reasoning for all three stages. Omit the model override to use the normal medium/high/high defaults. Screening runs again even though it already used medium. Accepted sources are reused from the cache, and newly accepted papers are downloaded if necessary. Investigation and verification replace the saved stage results; regenerated items need fresh human review. The exported `data/arxiv_false/august` dataset remains available for comparison until you explicitly export again. To review the new candidates:

```bash
uv run python arxivmath/app.py --false --paper-root arxivmath/paper
```

## Export

Export the currently human-accepted items to a new directory:

```bash
uv run python arxivmath/scripts/broken/export_false_proofs.py \
  --paper-root "$ARXIV_FALSE_PAPER_ROOT" --source-cache "$ARXIV_SOURCE_CACHE" \
  --out-dir data/arxiv_false/august --date 2026-08-01

# Review/copy the emitted competition config when ready.
cp data/arxiv_false/august/competition.yaml configs/competitions/arxiv_false/august.yaml
```

The exporter reruns the shared audit for each human-accepted item and refuses any stale or inconsistent acceptance and existing output directories. Pending and rejected items are skipped. `--max-papers` optionally restricts the sorted metadata-folder scope. Duplicate false statements after whitespace normalization are rejected; semantic deduplication is left to review.

The output is prepared in a temporary sibling directory and renamed into place:

- `problems/<id>.tex`: the false statement;
- `original/<id>.tex`: the true reference read by the benchmark judge;
- `source.csv`, `source_metadata.csv`: pinned arXiv IDs and attribution;
- `grading_scheme.json`: a maximum of 3 points per problem for the behavioral judge;
- `refutations/<id>.json`: both refutations and their evidence;
- `generation_manifest.json`: source/model records, review decisions and audit;
- `competition.yaml`: the competition definition to review before installing.

Reference files and manifests are curation material and must not be given to solvers. Refutations are not stored as proofs of the false statement. The existing judge assesses model behavior; this change does not alter its rubric or turn it into a counterexample-correctness judge.

The source pipeline is now the only generation path for ArXivMath and BrokenArXiv. The old BrokenArXiv branches in the shared abstract/OCR scripts, their four prompts, the old training-generation and extraction scripts, and the ArXivMath exporter’s legacy annotation fallback have been deleted. Shared OCR code remains for Lean generation. Existing datasets, competitions and their evaluation code are preserved; there are no migration aliases or legacy generation wrappers.

## Evaluate the accepted August examples

`arxiv_false/august` contains the 56 accepted examples. Its judge config, `judges/arxiv_judge_gemini_38_flash`, uses Gemini 3.8 Flash at high reasoning effort with the revised 0–3 behavioral rubric. Previous competition configs keep their existing judge. The judge reuses the normal Gemini model config through `HarnessSolver` and the Antigravity CLI, including its pinned version, high reasoning effort, Docker environment and native tools. It does not use the legacy API/Modal code-execution loop. Each judgment gets an isolated workspace under `logs/harness_workspaces/judges` (or `MATHARENA_REQUEST_LOG_DIR`), and its harness history and token costs are saved with the judgment. Concurrency follows `n_threads` in the judge config. Invalid or missing judge scores are retried automatically up to three total attempts, each with a fresh judge instance and workspace. The saved judgment includes all returned attempts' histories and token costs. After three failures it remains pending. Rerunning the same judge command retries pending judgments and preserves completed ones; use `--redo` only to regrade completed judgments too.

Run these commands in order from the repository root. The first exports the saved statements without model calls. The next two run one Astra attempt per problem and judge those attempts. The last prepares result summaries and traces for the local website.

```bash
uv run python arxivmath/scripts/broken/export_false_proofs.py \
  --paper-root arxivmath/paper --source-cache arxivmath/source_cache \
  --out-dir data/arxiv_false/august --date 2026-08-01

uv run python scripts/run.py --comp arxiv_false/august \
  --models openai/gpt-6-astra --n 1

uv run python scripts/judge/judge.py --comp arxiv_false/august \
  --models openai/gpt-6-astra
```

Astra uses its existing max-effort config and receives only the false statement with the proof request. The judge receives the true reference as well. Generation/refutation manifests are not solver inputs. Judging uses the existing Google API credentials. The 0–3 score measures response behavior under the revised rubric; it is not a proof-correctness score. Both the judge config and exported grading scheme use a maximum of 3. Results are normalized as points divided by 3, so 2 points corresponds to 66.7% and 3 points to 100%. If attempts have already been judged under the previous rubric, rerun the judge command with `--redo`.

## Where to review the mathematical policy

- [`source_screen.md`](../arxivmath/prompts/broken/source_screen.md): eligibility;
- [`source_investigate.md`](../arxivmath/prompts/broken/source_investigate.md): historical-claim selection, difficulty and refutation requirements;
- [`source_verify.md`](../arxivmath/prompts/broken/source_verify.md): fresh derivation and eight acceptance checks.

All three prompts start with the same explanation of the benchmark and its focus on recently open problems, then state the stage’s task. Prose paragraphs and list items have no hard word wrapping. Screening explains each eligible claim type and why an abstract can justify investigation without supplying all definitions or a disproof. Investigation starts with finding and faithfully restating a prior claim, followed by self-containedness, rigorous refutation, and difficulty checks. Verification explains its inputs and asks for a fresh derivation, evidence, and an assessment of correctness, novelty and research difficulty.

These prompts are the main places to adjust selection strictness. Model behavior and real-paper yield remain untested until a future authorized run.

## Offline checks

```bash
.venv/bin/python -m pytest -q \
  tests/test_query_client.py \
  tests/test_brokenarxiv_source.py \
  tests/test_arxivmath_source_scripts.py \
  tests/test_arxivmath_source_app.py \
  tests/test_arxivmath_source.py \
  tests/test_arxiv_source.py \
  tests/test_arxivmath_shared_modes.py
```

Tests run the shared stages against synthetic sources with mocked model clients. They check the actual prompt output examples against the validators and cover GPT-6 configuration, withheld generator rationale, new field names, evidence validation, same-config verification, resumption, stale records, review binding, automatic source skipping, export and screen-only behavior. They establish software behavior, not mathematical quality or empirical difficulty.
