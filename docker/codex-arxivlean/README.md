# Codex ArXivLean image

This image contains:

- Codex CLI 0.145.0
- Lean 4.31.0
- A read-only Docker volume containing Mathlib 4.31.0
- A system-enforced `arxivlean_benchmark` permission profile

The benchmark runner selects GPT-5.6 Sol with `max` reasoning through the
saved ChatGPT/Codex subscription authentication. Multi-agent mode remains
disabled, so each benchmark problem is handled by one isolated Codex agent.

Build it from the repository root:

```bash
scripts/build_codex_arxivlean_image.sh
```

The container itself needs outbound access so the Codex process can contact
OpenAI. Commands launched by Codex are restricted by the permission profile:
they can only write the current problem workspace, cannot read the mounted
Codex credentials, and cannot use the network.

The runtime relaxes Docker's seccomp and protected-system-path profiles because
Codex's Linux command sandbox needs an unprivileged user namespace with its own
`/proc`. Container capabilities are still fully dropped, privilege escalation
is disabled, the image root is read-only, and only the current problem
workspace is mounted from the host.

The build script creates the `matharena-lean431-cache` Docker volume and
downloads Mathlib's precompiled cache into it once. Benchmark containers mount
that volume read-only, so problems cannot communicate by modifying it.

The per-run `check.sh` is a Lean compilation check for the agent loop. It does
not replace MathArena's final Comparator evaluation; run the normal ArXivLean
judging pipeline on the saved outputs afterward.

Run all 48 June problems exactly once:

```bash
scripts/run_codex_arxivlean_june.sh
```

The wrapper uses the existing local MathArena Python environment and never
resolves or installs project dependencies. Set `MATHARENA_PYTHON` to override
the interpreter selection. It reads the committed `data/arxivlean/june` copy,
so preparing or running Codex does not download the benchmark from Hugging Face.

Check prerequisites without starting a model call:

```bash
scripts/run_codex_arxivlean_june.sh --check
```

Pass one or more problem numbers to run only those problems:

```bash
scripts/run_codex_arxivlean_june.sh 1 12 48
```
