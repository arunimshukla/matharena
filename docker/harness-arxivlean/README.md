# ArXivLean harness runtime image

This version-neutral image contains:

- Lean 4.31.0
- Node.js 22 and common shell utilities
- A scientific Python environment
- A separately populated, read-only Mathlib 4.31.0 cache volume

The Codex, Claude Code, Kimi Code, Antigravity CLI, Qwen Code, or OpenCode version
comes from the model config's `harness_version` field. MathArena installs that
exact release into a versioned host cache and mounts it read-only at
`/opt/harness-cli`. If the field is omitted, `latest` is resolved once at run
startup, and the resolved version is recorded in the output history.


Build the image and cache from the repository root:

```bash
scripts/build_harness_arxivlean_image.sh
```

The resulting image is `matharena-harness-arxivlean:lean4.31`. All six
harnesses use the same ephemeral read-only runtime. The current problem
workspace is the only writable bind mount, and Mathlib is mounted read-only.
The image entrypoint sets Linux's `no_new_privs` bit before launching the
selected CLI. This preserves the hardening guarantee on Snap-packaged Docker
versions that reject Docker's equivalent launch-time `--security-opt` flag.

Model traffic does not require general container internet access. MathArena
creates an internal Docker network for each invocation and exposes only a
host-side proxy. API keys and saved Codex/ChatGPT tokens remain on the host;
the container receives a random invocation-local proxy key.

The per-run `check.sh` gives the coding harness a Lean feedback loop. The
normal MathArena comparator remains responsible for final judging.
