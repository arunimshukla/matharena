from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness_wrapper.traces import TokenUsage, Trace, TraceEvent
from harness_wrapper.visualization import render_trace_html


def test_trace_persists_all_universal_event_kinds(tmp_path: Path) -> None:
    trace = Trace("session-1", root=tmp_path, harness="codex", model="tiny", metadata={"run": 2})
    trace.add_message("user", "hello")
    trace.add_reasoning("thinking", private=True)
    trace.add_tool_call("read", {"path": "README.md"}, tool_call_id="call-1")
    trace.add_tool_result("read", {"text": "ok"}, tool_call_id="call-1")
    trace.add_answer("done", finish_reason="stop")
    trace.add_usage(input_tokens=10, output_tokens=3, cache_read_tokens=2, cost_usd=0.01)

    assert trace.path == tmp_path / ".harness_wrapper" / "traces" / "session-1.jsonl"
    assert len(trace) == 6
    records = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"]["type"] for record in records] == [
        "message",
        "reasoning",
        "tool_call",
        "tool_result",
        "answer",
        "usage",
    ]
    assert records[2]["event"]["tool_call_id"] == "call-1"
    assert records[-1]["event"]["usage"]["total_tokens"] == 13
    assert records[-1]["harness"] == "codex"
    assert records[-1]["model"] == "tiny"
    assert records[-1]["trace_metadata"] == {"run": 2}


def test_trace_can_be_rendered_as_self_contained_html(tmp_path: Path) -> None:
    trace = Trace(
        "visualized",
        root=tmp_path,
        harness="codex",
        model="tiny",
        metadata={"run": 2},
        persist=False,
    )
    trace.add_message("user", "hello </script><script>alert('no')</script>")
    trace.add_usage(input_tokens=10, output_tokens=3, cost_usd=0.01)

    html = render_trace_html(trace)

    assert html.startswith("<!doctype html>")
    assert '"session_id":"visualized"' in html
    assert '"harness":"codex"' in html
    assert '"total_tokens":13' in html
    assert "hello </script>" not in html
    assert "hello \\u003c/script>" in html
    assert 'id="top-open"' in html


def test_trace_reopens_and_loads_jsonl(tmp_path: Path) -> None:
    original = Trace("resume", root=tmp_path)
    original.add_answer("first")
    reopened = Trace("resume", root=tmp_path)
    reopened.add_answer("second")

    assert [event.content for event in reopened] == ["first", "second"]
    loaded = Trace.load(reopened.path)
    assert loaded.session_id == "resume"
    assert [event.content for event in loaded] == ["first", "second"]


def test_activity_time_uses_latest_event_timestamp(tmp_path: Path) -> None:
    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    trace = Trace("activity", root=tmp_path, persist=False)
    trace.append(TraceEvent("answer", "ok", timestamp=when))

    assert trace.last_activity_at == when
    assert trace.last_activity == when
    assert trace.seconds_since_activity(now=when + timedelta(seconds=12.5)) == 12.5


def test_nonpersistent_trace_can_be_atomically_saved(tmp_path: Path) -> None:
    trace = Trace("manual", root=tmp_path, persist=False)
    trace.add("custom", {"value": True}, metadata={"extension": "test"})
    assert not trace.path.exists()
    assert trace.save() == trace.path
    assert trace.path.exists()
    assert Trace("manual", root=tmp_path).events[0].metadata == {"extension": "test"}


def test_trace_values_are_json_safe_and_ids_are_path_safe(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Trace("../escape", root=tmp_path)
    with pytest.raises(ValueError, match="inside"):
        Trace("escape", root=tmp_path, storage_dir=tmp_path.parent / "global-cache")
    with pytest.raises(TypeError):
        TraceEvent("answer", object())
    with pytest.raises(ValueError):
        TraceEvent("answer", float("nan"))


def test_usage_validation_and_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=-1)
    usage = TokenUsage(input_tokens=5, output_tokens=4, cache_write_tokens=2)
    assert usage.total_tokens == 9
    trace = Trace("usage", root=tmp_path, persist=False)
    event = trace.add_usage(usage)
    assert event.usage == usage
    compatible = trace.add_usage(prompt_tokens=3, completion_tokens=2)
    assert compatible.usage is not None
    assert compatible.usage.input_tokens == 3


def test_record_event_accepts_provider_adapter_mapping(tmp_path: Path) -> None:
    trace = Trace("mapping", root=tmp_path, persist=False)
    event = trace.record_event(
        {"type": "answer", "content": "ok", "role": "assistant", "metadata": {"raw": 1}}
    )
    assert event == trace.events[0]
    assert event.metadata == {"raw": 1}


def test_trace_rejects_storage_directory_symlink_replacement(tmp_path: Path) -> None:
    trace = Trace("known", root=tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    if trace.storage_dir.exists():
        shutil.rmtree(trace.storage_dir)
    trace.storage_dir.parent.mkdir(parents=True, exist_ok=True)
    trace.storage_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        trace.add_answer("must stay local")
    assert not (outside / "known.jsonl").exists()
    assert trace.events == ()


def test_trace_rejects_trace_file_symlink(tmp_path: Path) -> None:
    trace = Trace("linked", root=tmp_path)
    trace.storage_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    trace.path.symlink_to(victim)

    with pytest.raises(OSError):
        trace.add_answer("must not append")
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert trace.events == ()


def test_trace_rejects_trace_file_hardlink(tmp_path: Path) -> None:
    trace = Trace("linked", root=tmp_path)
    trace.storage_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    trace.path.hardlink_to(victim)

    with pytest.raises(OSError, match="hard links"):
        trace.add_answer("must not append")
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert trace.events == ()


def test_concurrent_trace_instances_do_not_lose_events(tmp_path: Path) -> None:
    first = Trace("shared", root=tmp_path)
    first.add_answer("one")
    second = Trace("shared", root=tmp_path)

    first.add_answer("two")
    second.add_answer("three")
    second.flush()

    assert [event.content for event in Trace("shared", root=tmp_path)] == [
        "one",
        "two",
        "three",
    ]


def test_trace_enforces_private_permissions_independent_of_umask(tmp_path: Path) -> None:
    original_umask = os.umask(0)
    try:
        trace = Trace("private", root=tmp_path)
        trace.add_reasoning("sensitive")
    finally:
        os.umask(original_umask)

    assert trace.storage_dir.stat().st_mode & 0o777 == 0o700
    assert trace.path.stat().st_mode & 0o777 == 0o600


def test_failed_persistence_does_not_mutate_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = Trace("failure", root=tmp_path)

    def fail(_event: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(trace, "_write_event", fail)
    with pytest.raises(OSError, match="disk full"):
        trace.add_answer("not durable")
    assert trace.events == ()


def test_incomplete_final_record_is_ignored_and_repaired(tmp_path: Path) -> None:
    trace = Trace("crash", root=tmp_path)
    trace.add_answer("complete")
    with trace.path.open("ab") as handle:
        handle.write(b'{"schema_version":1,"event":')

    recovered = Trace("crash", root=tmp_path)
    assert [event.content for event in recovered] == ["complete"]
    recovered.add_answer("after crash")
    assert [event.content for event in Trace("crash", root=tmp_path)] == [
        "complete",
        "after crash",
    ]


@pytest.mark.parametrize("warmup", ["append", "reopen", "flush"])
def test_repeated_appends_do_not_reread_history(tmp_path, monkeypatch, warmup):
    trace = Trace("fast", root=tmp_path)
    trace.add_answer("first")
    if warmup == "reopen":
        trace = Trace("fast", root=tmp_path)
    elif warmup == "flush":
        trace.flush()

    def unexpected_read(_storage_fd):
        pytest.fail("Appending to an unchanged trace reread its history")

    monkeypatch.setattr(trace, "_read_events_fd", unexpected_read)
    for i in range(32):
        trace.add_answer(f"event {i}")
    assert Trace("fast", root=tmp_path).events == trace.events
    assert len(trace) == 33


@pytest.mark.parametrize("replace_file", [False, True])
def test_cached_trace_rejects_same_size_history_changes(tmp_path, replace_file):
    trace = Trace("changed", root=tmp_path)
    trace.add_answer("first")
    previous = trace.path.stat()
    rewritten = trace.path.read_bytes().replace(b'"first"', b'"other"')
    assert len(rewritten) == previous.st_size
    if replace_file:
        replacement = trace.path.with_suffix(".replacement")
        replacement.write_bytes(rewritten)
        replacement.replace(trace.path)
    else:
        trace.path.write_bytes(rewritten)
    os.utime(trace.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))

    with pytest.raises(RuntimeError, match="trace changed concurrently"):
        trace.add_answer("must not append")
    assert trace.path.read_bytes() == rewritten


def test_append_reconciles_after_fsync_failure(tmp_path, monkeypatch):
    trace = Trace("fsync", root=tmp_path)
    trace.add_answer("first")

    def fail(_fd):
        raise OSError("fsync failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="fsync failed"):
            trace.add_answer("written before failure")
    assert [event.content for event in trace] == ["first"]
    trace.add_answer("after failure")
    assert [event.content for event in trace] == [
        "first",
        "written before failure",
        "after failure",
    ]
    assert Trace("fsync", root=tmp_path).events == trace.events


@pytest.mark.parametrize("request_log", [False, True])
def test_appending_to_unchanged_history_does_not_decode_old_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_log: bool
) -> None:
    kwargs = (
        {"session_id": "model_requests", "storage_dir": tmp_path / ".harness_wrapper"}
        if request_log
        else {"session_id": "stream"}
    )
    trace = Trace(root=tmp_path, **kwargs)
    for _ in range(20):
        trace.add_message("user", "historical context " * 500)
    # Initial loading is expected; appending our own events must not repeatedly
    # decode this history as it grows. This guards throughput without a flaky
    # wall-clock assertion on CI machines.
    reopened = Trace(root=tmp_path, **kwargs)
    with monkeypatch.context() as patch:

        def unexpected_read(*_args: object) -> None:
            pytest.fail("unchanged historical records were reread during append")

        patch.setattr(reopened, "_read_events_fd", unexpected_read)
        for i in range(50):
            reopened.add_answer(f"chunk {i}")
    assert len(Trace.load(reopened.path)) == 70
    assert reopened.events[-1].content == "chunk 49"


def test_parallel_trace_instances_preserve_every_writer(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    traces = [Trace("shared", root=tmp_path) for _ in range(4)]
    barrier = Barrier(len(traces))

    def write(index: int) -> None:
        barrier.wait(timeout=5)
        for i in range(15):
            traces[index].add_answer(f"{index}:{i}")

    with ThreadPoolExecutor(max_workers=len(traces)) as pool:
        list(pool.map(write, range(len(traces))))
    events = Trace("shared", root=tmp_path).events
    assert len(events) == 60
    assert {event.content for event in events} == {
        f"{index}:{i}" for index in range(4) for i in range(15)
    }


@pytest.mark.parametrize("change", ["rewrite", "truncate", "replace", "delete"])
def test_cached_trace_rejects_changed_history(tmp_path: Path, change: str) -> None:
    trace = Trace("changed", root=tmp_path)
    trace.add_answer("old")
    initial = trace.path.stat()
    changed = trace.path.read_bytes().replace(b'"old"', b'"bad"')
    if change == "rewrite":
        trace.path.write_bytes(changed)
        # Same size and restored mtime must still be detected through ctime.
        os.utime(trace.path, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        assert trace.path.stat().st_mtime_ns == initial.st_mtime_ns
    elif change == "truncate":
        trace.path.write_bytes(b"")
    elif change == "replace":
        replacement = trace.path.with_suffix(".replacement")
        replacement.write_bytes(changed)
        os.replace(replacement, trace.path)
    else:
        trace.path.unlink()
    with pytest.raises(RuntimeError, match="changed concurrently"):
        trace.add_answer("must not hide the edit")
    assert [event.content for event in trace] == ["old"]


@pytest.mark.parametrize("complete_tail", [False, True])
@pytest.mark.parametrize("reopen", [False, True])
def test_cached_trace_recovers_a_tail_without_a_newline(
    tmp_path: Path, complete_tail: bool, reopen: bool
) -> None:
    trace = Trace("tail", root=tmp_path)
    trace.add_answer("before")
    tail = (
        json.dumps({"event": TraceEvent("answer", "recovered").to_dict()}).encode()
        if complete_tail
        else b'{"event":'
    )
    with trace.path.open("ab") as handle:
        handle.write(tail)
    if reopen:
        trace = Trace("tail", root=tmp_path)
    trace.add_answer("after")
    expected = ["before", "recovered", "after"] if complete_tail else ["before", "after"]
    assert [event.content for event in Trace.load(trace.path)] == expected
    assert trace.path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("failure", ["partial_write", "fsync"])
def test_failed_append_invalidates_cached_disk_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    trace = Trace("retry", root=tmp_path)
    trace.add_answer("before")
    with monkeypatch.context() as patch:
        if failure == "fsync":

            def fail_sync(_fd: int) -> None:
                raise OSError("simulated fsync failure")

            patch.setattr(os, "fsync", fail_sync)
        else:
            original_write = os.write
            wrote_partial = False

            def fail_write(fd: int, data: bytes) -> int:
                nonlocal wrote_partial
                if wrote_partial:
                    raise OSError("simulated write failure")
                wrote_partial = True
                return original_write(fd, data[: len(data) // 2])

            patch.setattr(os, "write", fail_write)
        with pytest.raises(OSError, match="simulated"):
            trace.add_answer("uncertain")
    assert [event.content for event in trace] == ["before"]
    trace.add_answer("after")
    # A complete record whose fsync failed may still be present. A partial
    # record must be removed before retrying. Neither case may corrupt history.
    expected = ["before", "uncertain", "after"] if failure == "fsync" else ["before", "after"]
    assert [event.content for event in Trace.load(trace.path)] == expected
    assert [event.content for event in trace] == expected


def test_cached_trace_requires_flushing_unpersisted_local_events(tmp_path: Path) -> None:
    trace = Trace("pending", root=tmp_path)
    trace.add_answer("one")
    trace.persist = False
    trace.add_answer("two")
    trace.persist = True
    with pytest.raises(RuntimeError, match="changed concurrently"):
        trace.add_answer("three")
    trace.flush()
    trace.add_answer("three")
    assert [event.content for event in Trace.load(trace.path)] == ["one", "two", "three"]
