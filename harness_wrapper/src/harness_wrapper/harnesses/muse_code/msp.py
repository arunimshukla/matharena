"""Continue a retained Muse conversation through its native session protocol.

Muse 1.0.3's exec --session-id path can restore the wrong journal sequence after
compaction. MSP session/resume uses the durable session loader instead. This
module never reconstructs, renumbers, or replaces conversation history.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from ...agent import AgentEvent
from ...tools import CLIProcessError, merge_environment

if TYPE_CHECKING:
    from .adapter import MuseCodeAgent


def _command_id() -> str:
    """MSP requires UUIDv7 command IDs (also on Python versions before 3.14)."""
    value = (int(time.time() * 1000) << 80) | (uuid4().int & ((1 << 76) - 1))
    value = (value & ~(3 << 62)) | (2 << 62) | (7 << 76)
    return str(UUID(int=value))


def resume_session(
    agent: MuseCodeAgent, message: str | None, session_id: str,
) -> Iterator[AgentEvent]:
    command = agent.build_command(message, resume=True, session_id=session_id)
    explicit_env = agent.model_environment()
    explicit_env.update(agent._environment_override)
    host_cwd = agent.root
    if agent.env is not None:
        command, host_cwd, process_env = agent.env.prepare_command(
            command, cwd=agent.root, env=explicit_env,
        )
    else:
        process_env = merge_environment(explicit_env)
    bound_logger = logger.bind(harness=agent.harness_name, session_id=session_id)
    bound_logger.info("Continuing native Muse session through MSP")
    with agent._process_lock:
        if agent._process is not None and agent._process.poll() is None:
            raise RuntimeError("this Agent already has a running process")
        process = agent._process_factory(
            command, cwd=host_cwd, env=process_env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, **({"start_new_session": True} if os.name == "posix" else {}),
        )
        agent._process = process
        agent._interrupted_process = None
    stderr: list[str] = []

    def drain_stderr() -> None:
        if process.stderr is not None:
            for line in process.stderr:
                agent._touch()
                stderr.append(line)

    reader = threading.Thread(target=drain_stderr, name="muse-msp-stderr", daemon=True)
    reader.start()
    request_id = 0
    observed_items: set[str] = set()
    completed_turn = False

    def send(method: str, params: dict[str, Any], *, notification: bool = False) -> int | None:
        nonlocal request_id
        wire: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            request_id += 1
            wire["id"] = request_id
        if process.stdin is None:
            raise CLIProcessError(command, 1, "Muse MSP stdin is unavailable")
        try:
            process.stdin.write(json.dumps(wire) + "\n")
            process.stdin.flush()
        except OSError as error:
            raise CLIProcessError(command, 1, "Muse MSP input closed unexpectedly") from error
        return wire.get("id")

    def receive() -> dict[str, Any]:
        line = process.stdout.readline() if process.stdout is not None else ""
        if not line:
            reader.join(timeout=2)
            raise CLIProcessError(command, process.poll() or 1,
                                  "Muse MSP closed before completing the turn\n" + "".join(stderr))
        agent._touch()
        try:
            wire = json.loads(line)
            if not isinstance(wire, dict) or wire.get("jsonrpc") != "2.0":
                raise ValueError("expected an MSP JSON-RPC message")
        except ValueError as error:
            raise CLIProcessError(command, 1, "Invalid Muse MSP response") from error
        # Do not automatically approve new requests or fabricate tool results.
        if "method" in wire and "id" in wire:
            raise CLIProcessError(command, 1, f"Muse MSP requires client input: {wire['method']}")
        return wire

    def notify(wire: dict[str, Any]) -> Iterator[AgentEvent]:
        yield from agent._drain_provider_events()
        params = wire.get("params", {})
        yield AgentEvent(type="event", content=params, session_id=session_id, raw=wire)
        if wire.get("method") == "item/completed":
            item = params.get("item", {})
            item_id = item.get("itemId")
            if item.get("kind") == "toolCall" and item_id not in observed_items:
                observed_items.add(item_id)
                yield AgentEvent(
                    type="tool_result", role="tool", content=item.get("visibleOutput", ""),
                    tool_name=item.get("tool"), tool_call_id=item.get("callId"),
                    session_id=session_id, raw=wire,
                )

    def rpc(
        method: str, params: dict[str, Any],
    ) -> Generator[AgentEvent, None, dict[str, Any]]:
        wanted = send(method, params)
        while True:
            wire = receive()
            if wire.get("id") == wanted:
                if "error" in wire:
                    error = wire["error"]
                    detail = error.get("message", error)
                    raise CLIProcessError(command, 1, f"Muse MSP {method}: {detail}")
                return wire.get("result", {})
            yield from notify(wire)

    try:
        yield from rpc("initialize", {
            "clientInfo": {"name": "harness_wrapper", "version": "0.1.0"},
        })
        send("initialized", {}, notification=True)
        resumed = yield from rpc("session/resume", {
            "sessionId": session_id, "commandId": _command_id(), "excludeItems": True,
        })
        session = resumed.get("session", {})
        if session.get("sessionId") != session_id or session.get("status") != "idle":
            raise CLIProcessError(command, 1, "Muse did not resume the requested idle session")
        if session.get("modelId") != agent.model_name or session.get("providerId") != "meta":
            raise CLIProcessError(command, 1, "Muse resumed with a different model or provider")
        yield AgentEvent(type="session", content={"resumed": True, "transport": "msp"},
                         session_id=session_id)
        if agent.env is not None and getattr(agent.env, "enabled", False):
            # Match exec's --disable-approval within the existing outer sandbox.
            yield from rpc("session/setApprovalMode", {
                "sessionId": session_id, "commandId": _command_id(), "mode": "allowAll",
            })
        params: dict[str, Any] = {
            "sessionId": session_id, "commandId": _command_id(),
            # Preserve an explicit empty string: Muse supplies its own
            # placeholder, without an extra model-visible user instruction.
            "input": [{"type": "text", "text": message if message is not None else "Continue the interrupted task."}],
        }
        effort = getattr(agent.model, "reasoning", None)
        if effort is not None:
            # The MSP enum spells the CLI's max alias ultra. Provider overrides
            # still enforce the configured API effort (e.g. reasoning.effort=max).
            params["reasoningEffort"] = "ultra" if effort == "max" else str(effort)
        started = yield from rpc("turn/start", params)
        if started.get("disposition") != "started":
            raise CLIProcessError(command, 1, "Muse did not start the continuation turn")
        turn_id = started["turnId"]
        while True:
            wire = receive()
            yield from notify(wire)
            params = wire.get("params", {})
            if (wire.get("method") != "turn/completed"
                    or params.get("sessionId") != session_id or params.get("turnId") != turn_id):
                continue
            if params.get("terminal") != "completed":
                detail = params.get("error", {}).get("message") or params.get("reason")
                raise CLIProcessError(command, 1, str(detail or "Muse continuation failed"))
            # MSP's view items may truncate long answers. Use the complete text
            # captured from the provider only after native terminal completion.
            completed_turn = True
            yield AgentEvent(type="result", content=agent._last_solver_final_text,
                             session_id=session_id, raw=wire)
            break
    finally:
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        interrupted = agent._interrupted_process is process
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        reader.join(timeout=2)
        with agent._process_lock:
            if agent._process is process:
                agent._process = None
            if agent._interrupted_process is process:
                agent._interrupted_process = None
        if completed_turn and returncode and not interrupted:
            raise CLIProcessError(command, returncode, "".join(stderr))
