"""Codex CLI solver for Lean benchmarks.

This runs the Codex CLI in a Docker container whose writable filesystem is a
single per-problem workspace. The recommended image contains a read-only Lean
environment and denies model-generated commands access to both the network and
the separately mounted Codex credentials.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from loguru import logger

from matharena.solvers import BaseSolver, SolverResponse


CONTAINER_WORKDIR = "/work"
CONTAINER_TMPDIR = "/tmp"
CONTAINER_ELAN_HOME = "/elan"
CONTAINER_LEAN_CACHE = "/mathlib-cache"
CONTAINER_CODEX_HOME = "/run/codex-home"
CONTAINER_BASE_PATH = (
    f"{CONTAINER_ELAN_HOME}/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
DEFAULT_DOCKER_IMAGE = "matharena-codex-arxivlean:lean4.31-codex0.145.0"
DEFAULT_WORKSPACE_ROOT = "logs/codex_workspaces/{competition}/{solver_name}/p{problem_idx}_r{run_idx}"
DEFAULT_PACKAGE_ORDER = [
    "Cli",
    "batteries",
    "Qq",
    "aesop",
    "proofwidgets",
    "importGraph",
    "LeanSearchClient",
    "plausible",
    "mathlib",
]


@dataclass
class CodexUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    n_turns: int = 0

    @property
    def fresh_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)


@dataclass
class DockerResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


class CodexCLISolver(BaseSolver):
    """Run Codex CLI on one isolated Lean workspace per problem/run."""

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        self.config = default_api_client_args.copy()
        self.competition = str(self.config.get("competition") or "unknown_competition")
        self.solver_name = str(self.config.get("solver_name") or "codex_cli")
        self.concurrent_requests = int(self.config.get("concurrent_requests") or 1)
        self.docker_image = str(self.config.get("docker_image") or DEFAULT_DOCKER_IMAGE)
        self.docker_network = str(self.config.get("docker_network") or "bridge")
        self.check_docker_network = str(self.config.get("check_docker_network") or "none")
        self.timeout_s = int(self.config.get("timeout_s") or self.config.get("max_seconds") or 28_800)
        self.cpu_limit = self.config.get("cpu_limit", 2)
        self.memory_gb = self.config.get("memory_gb", 8)
        self.pids_limit = int(self.config.get("pids_limit") or 512)
        self.tmpfs_size = str(self.config.get("tmpfs_size") or "1g")
        self.copy_codex_auth = bool(self.config.get("copy_codex_auth", True))
        self.codex_auth_path = Path(
            str(self.config.get("codex_auth_path") or Path.home() / ".codex" / "auth.json")
        ).expanduser().resolve()
        self.keep_workspaces = bool(self.config.get("keep_workspaces", True))
        self.lean_runtime = str(self.config.get("lean_runtime") or "host").lower()
        if self.lean_runtime not in {"host", "image"}:
            raise ValueError("lean_runtime must be either 'host' or 'image'.")
        self.lean_cache_volume = self.config.get("lean_cache_volume")
        self.cache_read_only = bool(self.config.get("cache_read_only", False))
        self.read_only_rootfs = bool(self.config.get("read_only_rootfs", False))
        self.emit_empty_on_failure = bool(self.config.get("emit_empty_on_failure", False))
        self.run_check_after_codex = bool(self.config.get("run_check_after_codex", True))
        self.cmd = _coerce_cmd(self.config.get("cmd") or ["codex", "exec", "--json"])
        self.cmd = _command_with_codex_options(
            self.cmd,
            model=self.config.get("model"),
            reasoning_effort=self.config.get("model_reasoning_effort"),
            codex_sandbox=str(self.config.get("codex_sandbox") or "auto"),
        )
        self.send_prompt_stdin = bool(self.config.get("send_prompt_stdin", _is_codex_exec_cmd(self.cmd)))
        self.host_lean_project = Path(
            str(self.config.get("lean_project_root") or "external/comparator_project")
        ).expanduser()
        if not self.host_lean_project.is_absolute():
            self.host_lean_project = Path.cwd() / self.host_lean_project
        self.host_lean_project = self.host_lean_project.resolve()
        self.host_lake_cache = Path(str(self.config.get("lake_cache_root") or self.host_lean_project / ".lake"))
        if not self.host_lake_cache.is_absolute():
            self.host_lake_cache = Path.cwd() / self.host_lake_cache
        self.host_lake_cache = self.host_lake_cache.resolve()
        self.host_elan_home = Path(str(self.config.get("elan_home") or Path.home() / ".elan")).expanduser().resolve()
        self.workspace_root_template = str(self.config.get("workspace_root") or DEFAULT_WORKSPACE_ROOT)
        self.package_names = _package_names(self.host_lean_project)
        self.toolchain_dir = str(self.config.get("lean_toolchain_dir") or _toolchain_dir(self.host_lean_project))

    def build_prompt(self, text: str | dict[str, Any] | None) -> str:
        if isinstance(text, dict):
            fields = {k: v for k, v in text.items() if v is not None}
            fields.setdefault("problem", "See Problem.md.")
        else:
            fields = {"problem": "See Problem.md." if text is None else text}
        return self.default_prompt_template.format_map(_PromptFields(fields))

    @override
    def solve_batch(
        self,
        stmt_batch: list[tuple[str, Any]],
        batch_idx_to_problem_idx: dict[int, int],
        batch_idx_to_run_idx: dict[int, int],
    ):
        max_workers = max(1, min(self.concurrent_requests, len(stmt_batch)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._solve_one,
                    batch_idx,
                    payload,
                    image_b64,
                    batch_idx_to_problem_idx[batch_idx],
                    batch_idx_to_run_idx[batch_idx],
                ): batch_idx
                for batch_idx, (payload, image_b64) in enumerate(stmt_batch)
            }
            for future in as_completed(futures):
                yield future.result()

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        return previous_response

    def _solve_one(
        self,
        batch_idx: int,
        payload: str | dict[str, Any] | None,
        image_b64: str | None,
        problem_idx: int,
        run_idx: int,
    ) -> SolverResponse:
        if image_b64 is not None:
            raise ValueError("CodexCLISolver currently supports text/Lean problems only.")

        prompt = self.build_prompt(payload)
        workspace = self._workspace_for(problem_idx, run_idx)
        self._prepare_workspace(workspace, payload, prompt)
        codex_home = None
        if self.copy_codex_auth and _is_codex_exec_cmd(self.cmd):
            codex_home = self._stage_codex_auth(workspace)

        logger.info(f"Starting Codex CLI run for P{problem_idx} r{run_idx} in {workspace}")
        try:
            result = self._run_in_docker(
                workspace,
                self.cmd,
                prompt if self.send_prompt_stdin else None,
                codex_home=codex_home,
            )
            self._write_run_logs(workspace, result)
            if self.run_check_after_codex:
                check = self._run_in_docker(
                    workspace,
                    ["sh", "-c", "./check.sh"],
                    None,
                    timeout_s=600,
                    docker_network=self.check_docker_network,
                )
                self._write_run_logs(workspace, check, stem="check")
        finally:
            if codex_home is not None:
                shutil.rmtree(codex_home, ignore_errors=True)

        solution_text = _read_text(workspace / "Solution.lean")
        if self.emit_empty_on_failure and result.returncode != 0:
            assistant_content = ""
        elif solution_text.strip():
            assistant_content = f"```lean\n{solution_text.rstrip()}\n```"
        else:
            assistant_content = _fallback_assistant_content(result.stdout)

        conversation = _codex_history_messages(prompt, result.stdout, assistant_content)
        detailed_cost = self._detailed_cost(result)
        codex_events = _parse_codex_events(result.stdout)
        history = [
            {
                "step": "codex_cli",
                "timestep": 0,
                "messages": conversation,
                "workspace": _display_workspace_path(workspace),
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "codex_events": codex_events,
                "codex_usage": _codex_usage_dict(_parse_codex_jsonl(result.stdout)),
                "check_returncode": check.returncode if self.run_check_after_codex else None,
                "check_timed_out": check.timed_out if self.run_check_after_codex else None,
            }
        ]
        response = SolverResponse(batch_idx, conversation, detailed_cost, history=history)
        if not self.keep_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        return response

    def _workspace_for(self, problem_idx: int, run_idx: int) -> Path:
        fields = {
            "competition": self.competition,
            "solver_name": self.solver_name,
            "problem_idx": problem_idx,
            "run_idx": run_idx,
        }
        raw = _format_template(self.workspace_root_template, fields)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _prepare_workspace(self, workspace: Path, payload: str | dict[str, Any] | None, prompt: str) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            natural = str(payload.get("problem") or "")
            formal = str(payload.get("formal_statement") or "")
        else:
            natural = "" if payload is None else str(payload)
            formal = ""

        _write_text(workspace / "Problem.md", _problem_markdown(natural, formal))
        _write_text(workspace / "prompt.txt", prompt)
        if formal.strip():
            initial_solution = f"import Mathlib\n\n{formal.rstrip()}\n"
        else:
            initial_solution = "import Mathlib\n\n"
        _write_text(workspace / "Problem.lean", initial_solution)
        solution_path = workspace / "Solution.lean"
        if not solution_path.exists() or self.config.get("overwrite_solution", True):
            _write_text(solution_path, initial_solution)
        _write_text(
            workspace / "README.md",
            _workspace_readme(self.docker_network, self.run_check_after_codex),
        )
        check = (
            "#!/usr/bin/env sh\n"
            "set -eu\n"
            f'export PATH="{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/bin:{CONTAINER_ELAN_HOME}/bin:$PATH"\n'
            'check_log="${TMPDIR:-/tmp}/lean-check-$$.log"\n'
            'if ! lean Solution.lean > "$check_log" 2>&1; then\n'
            '  cat "$check_log"\n'
            '  rm -f "$check_log"\n'
            "  exit 1\n"
            "fi\n"
            'cat "$check_log"\n'
            'if grep -q "warning:" "$check_log"; then\n'
            '  rm -f "$check_log"\n'
            "  exit 1\n"
            "fi\n"
            'rm -f "$check_log"\n'
        )
        _write_text(workspace / "check.sh", check)
        try:
            (workspace / "check.sh").chmod(0o755)
        except OSError:
            pass

    def _stage_codex_auth(self, workspace: Path) -> Path | None:
        if not self.codex_auth_path.exists():
            logger.warning(
                f"copy_codex_auth is enabled, but {self.codex_auth_path} does not exist."
            )
            return None
        codex_home = workspace.parent / f".codex-auth-{uuid.uuid4().hex}"
        codex_home.mkdir(mode=0o700, parents=True, exist_ok=False)
        shutil.copyfile(self.codex_auth_path, codex_home / "auth.json")
        try:
            (codex_home / "auth.json").chmod(0o600)
        except OSError:
            pass
        return codex_home

    def _run_in_docker(
        self,
        workspace: Path,
        inner_cmd: list[str],
        stdin_text: str | None,
        *,
        timeout_s: int | None = None,
        docker_network: str | None = None,
        codex_home: Path | None = None,
    ) -> DockerResult:
        container_name = f"matharena-codex-{uuid.uuid4().hex[:12]}"
        docker_cmd = self._docker_cmd(
            workspace,
            inner_cmd,
            container_name,
            interactive=stdin_text is not None,
            docker_network=docker_network or self.docker_network,
            codex_home=codex_home,
        )
        start = time.monotonic()
        proc = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(stdin_text, timeout=timeout_s or self.timeout_s)
            return DockerResult(proc.returncode or 0, stdout, stderr, time.monotonic() - start)
        except subprocess.TimeoutExpired:
            _docker_kill(container_name)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            elapsed = time.monotonic() - start
            stderr = (stderr or "") + f"\ntimeout after {timeout_s or self.timeout_s}s"
            return DockerResult(-9, stdout or "", stderr, elapsed, timed_out=True)

    def _docker_cmd(
        self,
        workspace: Path,
        inner_cmd: list[str],
        container_name: str,
        *,
        interactive: bool,
        docker_network: str,
        codex_home: Path | None,
    ) -> list[str]:
        lean_env = self._lean_env()
        args = ["docker", "run", "--rm", "--name", container_name]
        if interactive:
            args.append("-i")
        args += [
            "--memory",
            f"{self.memory_gb}g",
            "--cpus",
            str(self.cpu_limit),
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--network",
            docker_network,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{workspace}:{CONTAINER_WORKDIR}",
            "-w",
            CONTAINER_WORKDIR,
            "--tmpfs",
            f"{CONTAINER_TMPDIR}:rw,nosuid,nodev,size={self.tmpfs_size},mode=1777",
            "-e",
            f"HOME={CONTAINER_TMPDIR}",
            "-e",
            f"TMPDIR={CONTAINER_TMPDIR}",
            "-e",
            f"PATH={CONTAINER_BASE_PATH}",
            "-e",
            f"ELAN_HOME={CONTAINER_ELAN_HOME}",
            "-e",
            f"CODEX_HOME={CONTAINER_CODEX_HOME if codex_home is not None else CONTAINER_TMPDIR + '/codex-home'}",
        ]
        if self.read_only_rootfs:
            args.append("--read-only")
        if codex_home is not None:
            args += ["-v", f"{codex_home}:{CONTAINER_CODEX_HOME}"]
        if self.lean_runtime == "host":
            args += [
                "-v",
                f"{self.host_elan_home}:{CONTAINER_ELAN_HOME}:ro",
                "-v",
                f"{self.host_lake_cache}:{CONTAINER_LEAN_CACHE}/.lake{':ro' if self.cache_read_only else ''}",
            ]
        elif self.lean_cache_volume:
            args += [
                "-v",
                f"{self.lean_cache_volume}:{CONTAINER_LEAN_CACHE}{':ro' if self.cache_read_only else ''}",
            ]
        for key, value in lean_env.items():
            args += ["-e", f"{key}={value}"]
        for extra_arg in self.config.get("docker_extra_args") or []:
            args.append(str(extra_arg))
        args += [self.docker_image, *inner_cmd]
        return args

    def _lean_env(self) -> dict[str, str]:
        lib_paths = [
            f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}/.lake/build/lib/lean"
            for name in self.package_names
        ]
        lib_paths += [
            f"{CONTAINER_LEAN_CACHE}/.lake/build/lib/lean",
            f"{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/lib/lean",
        ]
        native_lib_paths = [
            f"{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/lib/lean",
            f"{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/lib",
            f"{CONTAINER_LEAN_CACHE}/.lake/build/lib",
        ]
        native_lib_paths += [
            f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}/.lake/build/lib"
            for name in reversed(self.package_names)
        ]
        src_paths = []
        for name in self.package_names:
            src_paths.extend([f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}"])
        src_paths.append(f"{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/src/lean/lake")
        return {
            "LEAN_PATH": ":".join(lib_paths),
            "LEAN_SRC_PATH": ":".join(src_paths),
            "LD_LIBRARY_PATH": ":".join(native_lib_paths),
        }

    def _write_run_logs(self, workspace: Path, result: DockerResult, *, stem: str = "codex") -> None:
        _write_text(workspace / f"{stem}_stdout.log", result.stdout)
        _write_text(workspace / f"{stem}_stderr.log", result.stderr)
        _write_text(
            workspace / f"{stem}_result.json",
            json.dumps(
                {
                    "returncode": result.returncode,
                    "elapsed_s": result.elapsed_s,
                    "timed_out": result.timed_out,
                },
                indent=2,
            )
            + "\n",
        )

    def _detailed_cost(self, result: DockerResult) -> dict[str, Any]:
        usage = _parse_codex_jsonl(result.stdout)
        cost = 0.0
        if self.config.get("accounting_costs_from_usage"):
            read_cost = float(self.config.get("read_cost") or 0)
            write_cost = float(self.config.get("write_cost") or 0)
            cache_read_cost = float(self.config.get("cache_read_cost", read_cost) or 0)
            cost = (
                usage.fresh_input_tokens * read_cost
                + usage.cached_input_tokens * cache_read_cost
                + usage.output_tokens * write_cost
            ) / 1_000_000.0
        return {
            "cost": cost,
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "fresh_input_tokens": usage.fresh_input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_output_tokens": usage.reasoning_output_tokens,
            "n_turns": usage.n_turns,
            "read_cost": self.config.get("read_cost"),
            "cache_read_cost": self.config.get("cache_read_cost"),
            "write_cost": self.config.get("write_cost"),
            "model": self.config.get("model"),
            "time": result.elapsed_s,
            "request_time": result.elapsed_s,
            "n_retries": 0,
        }


class _PromptFields(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _coerce_cmd(raw: Any) -> list[str]:
    if isinstance(raw, str):
        import shlex

        return shlex.split(raw)
    if isinstance(raw, (list, tuple)):
        return [str(part) for part in raw]
    return []


def _command_with_codex_options(
    cmd: list[str],
    *,
    model: Any,
    reasoning_effort: Any,
    codex_sandbox: str,
) -> list[str]:
    if not _is_codex_exec_cmd(cmd):
        return cmd
    if model:
        cmd = _with_codex_model(cmd, str(model))
    if reasoning_effort:
        cmd = _with_codex_reasoning_effort(cmd, str(reasoning_effort))
    sandbox = codex_sandbox.lower()
    if sandbox in {"docker-bypass", "bypass"} and not _has_codex_sandbox_flag(cmd):
        cmd = _insert_before_prompt(cmd, ["--dangerously-bypass-approvals-and-sandbox"])
    elif sandbox in {"auto", "workspace", "workspace-write"} and not _has_codex_sandbox_flag(cmd):
        cmd = _insert_before_prompt(cmd, ["--sandbox", "workspace-write"])
    elif sandbox not in {"permissions", "permission-profile", "managed"}:
        raise ValueError(f"Unsupported codex_sandbox value: {codex_sandbox}")
    if _codex_prompt_arg_index(cmd) is None:
        cmd.append("-")
    return cmd


def _with_codex_model(cmd: list[str], model: str) -> list[str]:
    return _insert_codex_exec_options(_without_codex_option(cmd, {"-m", "--model"}, "--model="), ["-m", model])


def _with_codex_reasoning_effort(cmd: list[str], effort: str) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] in {"-c", "--config"} and i + 1 < len(cmd) and cmd[i + 1].startswith("model_reasoning_effort"):
            i += 2
            continue
        if cmd[i].startswith("--config=model_reasoning_effort"):
            i += 1
            continue
        cleaned.append(cmd[i])
        i += 1
    return _insert_codex_exec_options(cleaned, ["-c", f'model_reasoning_effort="{effort}"'])


def _without_codex_option(cmd: list[str], value_options: set[str], equals_prefix: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] in value_options:
            i += 2
            continue
        if cmd[i].startswith(equals_prefix):
            i += 1
            continue
        out.append(cmd[i])
        i += 1
    return out


def _insert_codex_exec_options(cmd: list[str], options: list[str]) -> list[str]:
    try:
        idx = cmd.index("exec") + 1
    except ValueError:
        return cmd
    return [*cmd[:idx], *options, *cmd[idx:]]


def _insert_before_prompt(cmd: list[str], options: list[str]) -> list[str]:
    prompt_idx = _codex_prompt_arg_index(cmd)
    if prompt_idx is None:
        return [*cmd, *options]
    return [*cmd[:prompt_idx], *options, *cmd[prompt_idx:]]


def _is_codex_exec_cmd(cmd: list[str]) -> bool:
    return bool(cmd and Path(cmd[0]).name == "codex" and "exec" in cmd)


def _codex_prompt_arg_index(cmd: list[str]) -> int | None:
    if not _is_codex_exec_cmd(cmd):
        return None
    try:
        i = cmd.index("exec") + 1
    except ValueError:
        return None
    value_options = {
        "-c",
        "--config",
        "-i",
        "--image",
        "-m",
        "--model",
        "--local-provider",
        "-p",
        "--profile",
        "-s",
        "--sandbox",
        "-C",
        "--cd",
        "--add-dir",
        "--output-schema",
        "--color",
        "-a",
        "--ask-for-approval",
        "-o",
        "--output-last-message",
    }
    value_options_with_equals = {
        "--config",
        "--image",
        "--model",
        "--local-provider",
        "--profile",
        "--sandbox",
        "--cd",
        "--add-dir",
        "--output-schema",
        "--color",
        "--ask-for-approval",
        "--output-last-message",
    }
    while i < len(cmd):
        part = cmd[i]
        if part == "--":
            return i + 1 if i + 1 < len(cmd) else None
        if part in value_options:
            i += 2
            continue
        if any(part.startswith(f"{opt}=") for opt in value_options_with_equals):
            i += 1
            continue
        if part == "-" or not part.startswith("-"):
            return i
        i += 1
    return None


def _has_codex_sandbox_flag(cmd: list[str]) -> bool:
    return any(part in {"--dangerously-bypass-approvals-and-sandbox", "--sandbox", "--full-auto"} for part in cmd)


def _format_template(template: str, fields: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(fields.get(key, match.group(0)))

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)


def _package_names(project_root: Path) -> list[str]:
    manifest_path = project_root / "lake-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_PACKAGE_ORDER
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        return DEFAULT_PACKAGE_ORDER
    names = [str(pkg["name"]) for pkg in packages if isinstance(pkg, dict) and pkg.get("name")]
    return names or DEFAULT_PACKAGE_ORDER


def _toolchain_dir(project_root: Path) -> str:
    toolchain_path = project_root / "lean-toolchain"
    raw = _read_text(toolchain_path).strip()
    if raw.startswith("leanprover/lean4:"):
        return "leanprover--lean4---" + raw.split(":", 1)[1]
    if raw:
        return raw.replace("/", "--").replace(":", "---")
    return "leanprover--lean4---v4.31.0"


def _parse_codex_jsonl(text: str) -> CodexUsage:
    usage = CodexUsage()
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        raw_usage = event.get("usage")
        if not isinstance(raw_usage, dict):
            continue
        usage.input_tokens += int(raw_usage.get("input_tokens") or 0)
        usage.cached_input_tokens += int(raw_usage.get("cached_input_tokens") or 0)
        usage.output_tokens += int(raw_usage.get("output_tokens") or 0)
        usage.reasoning_output_tokens += int(raw_usage.get("reasoning_output_tokens") or 0)
        usage.n_turns += 1
    return usage


def _parse_codex_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _codex_usage_dict(usage: CodexUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "fresh_input_tokens": usage.fresh_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "n_turns": usage.n_turns,
    }


def _codex_history_messages(prompt: str, stdout: str, final_content: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for event in _parse_codex_events(stdout):
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or f"codex_{len(messages)}")
        if item_type == "agent_message":
            text = str(item.get("text") or "")
            if text:
                messages.append({"role": "assistant", "type": "response", "content": text})
        elif item_type == "command_execution":
            command = str(item.get("command") or "")
            messages.append(
                {
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": "shell",
                    "tool_call_id": item_id,
                    "arguments": {
                        "command": command,
                        "status": item.get("status"),
                    },
                }
            )
            messages.append(
                {
                    "role": "tool_response",
                    "tool_name": "shell",
                    "tool_call_id": item_id,
                    "content": _codex_command_output(item),
                }
            )
        elif item_type == "file_change":
            messages.append(
                {
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": "file_change",
                    "tool_call_id": item_id,
                    "arguments": {
                        "changes": item.get("changes", []),
                        "status": item.get("status"),
                    },
                }
            )
            messages.append(
                {
                    "role": "tool_response",
                    "tool_name": "file_change",
                    "tool_call_id": item_id,
                    "content": json.dumps(item.get("changes", []), ensure_ascii=False),
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "type": "cot",
                    "content": json.dumps(item, ensure_ascii=False),
                }
            )

    if final_content and (
        not messages
        or messages[-1].get("role") != "assistant"
        or messages[-1].get("type", "response") != "response"
        or messages[-1].get("content") != final_content
    ):
        messages.append({"role": "assistant", "type": "response", "content": final_content})
    return messages


def _codex_command_output(item: dict[str, Any]) -> str:
    parts = []
    if item.get("aggregated_output"):
        parts.append(str(item["aggregated_output"]))
    if "exit_code" in item:
        parts.append(f"exit_code={item.get('exit_code')}")
    if item.get("status"):
        parts.append(f"status={item.get('status')}")
    return "\n".join(parts)


def _fallback_assistant_content(stdout: str) -> str:
    blocks = re.findall(r"```(?:lean)?\s*\n(.*?)```", stdout, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return f"```lean\n{blocks[-1].rstrip()}\n```"
    return stdout.strip()


def _problem_markdown(natural: str, formal: str) -> str:
    return (
        "# Problem\n\n"
        "## Natural language statement\n\n"
        f"{natural.rstrip()}\n\n"
        "## Formal statement\n\n"
        "```lean\n"
        f"{formal.rstrip()}\n"
        "```\n"
    )


def _workspace_readme(docker_network: str, run_check_after_codex: bool) -> str:
    check_note = "The runner will also execute `./check.sh` after Codex exits.\n" if run_check_after_codex else ""
    return (
        "# Isolated Codex Lean workspace\n\n"
        "Only this directory is mounted as the writable workspace for the agent.\n"
        "Edit `Solution.lean` and run `./check.sh` to typecheck it with Lean; warnings fail the check.\n"
        "The compiled Mathlib cache is mounted outside this directory and exposed through `LEAN_PATH`.\n"
        f"Docker network mode for the Codex process: `{docker_network}`.\n"
        f"{check_note}"
    )


def _docker_kill(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        pass


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _display_workspace_path(workspace: Path) -> str:
    try:
        return str(workspace.relative_to(Path.cwd()))
    except ValueError:
        return workspace.name


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
