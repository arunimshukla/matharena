"""Universal, append-only traces shared by every agent harness."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | str | None) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_safe(value: Any) -> JsonValue:
    """Validate and normalize values before they reach the trace file."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("trace values cannot contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"trace value is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-neutral token and cost statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cost_usd is not None and (self.cost_usd < 0 or not math.isfinite(self.cost_usd)):
            raise ValueError("cost_usd must be finite and non-negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def prompt_tokens(self) -> int:
        """OpenAI-compatible name for ``input_tokens``."""

        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """OpenAI-compatible name for ``output_tokens``."""

        return self.output_tokens

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = asdict(self)
        result["total_tokens"] = self.total_tokens
        return result


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A single event in the universal trace schema."""

    type: str
    content: JsonValue = None
    timestamp: datetime = field(default_factory=utc_now)
    role: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    usage: TokenUsage | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type or not self.type.strip():
            raise ValueError("trace event type cannot be empty")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "content", _json_safe(self.content))
        object.__setattr__(self, "metadata", _json_safe(dict(self.metadata)))

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "metadata": dict(self.metadata),
        }
        for key in ("role", "tool_name", "tool_call_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TraceEvent:
        usage_value = value.get("usage")
        usage = None
        if usage_value is not None:
            keys = (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost_usd",
            )
            usage = TokenUsage(**{key: usage_value[key] for key in keys if key in usage_value})
        return cls(
            type=str(value["type"]),
            timestamp=_timestamp(value.get("timestamp")),
            content=value.get("content"),
            role=value.get("role"),
            tool_name=value.get("tool_name"),
            tool_call_id=value.get("tool_call_id"),
            usage=usage,
            metadata=value.get("metadata", {}),
        )


class Trace:
    """In-memory trace with durable JSONL persistence.

    By default each append is immediately flushed beneath the invocation root,
    never a user's global home/cache directory. Existing trace files are loaded
    when the same session id is reopened.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        storage_dir: str | os.PathLike[str] | None = None,
        harness: str | None = None,
        model: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        persist: bool = True,
        load_existing: bool = True,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        if not _SESSION_ID.fullmatch(self.session_id):
            raise ValueError("session_id must be path-safe and at most 128 characters")
        self.root = Path(root or Path.cwd()).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"trace root is not a directory: {self.root}")
        if storage_dir is None:
            self.storage_dir = self.root / ".harness_wrapper" / "traces"
        else:
            candidate = Path(storage_dir).expanduser()
            self.storage_dir = (
                (self.root / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
        if self.storage_dir != self.root and self.root not in self.storage_dir.parents:
            raise ValueError("trace storage directory must be inside the invocation root")
        self.harness = harness
        self.model = model
        self.metadata = _json_safe(dict(metadata or {}))
        self.persist = persist
        self._events: list[TraceEvent] = []
        # A successful append already leaves memory in sync with the file. Use
        # file identity/metadata to avoid decoding the entire history again.
        # Changed files still take the full validation path (e.g. another writer).
        self._known_file_state: tuple[int, int, int, int, int] | None = None
        self._known_event_count: int | None = None
        self._lock = threading.RLock()
        if load_existing and self.storage_dir.exists():
            with self._locked_storage() as storage_fd:
                self._events.extend(self._read_events_fd(storage_fd))
                self._known_file_state = self._trace_file_state(storage_fd)
                self._known_event_count = len(self._events)

    @property
    def path(self) -> Path:
        return self.storage_dir / f"{self.session_id}.jsonl"

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    @property
    def last_activity_at(self) -> datetime | None:
        with self._lock:
            return self._events[-1].timestamp if self._events else None

    @property
    def last_activity(self) -> datetime | None:
        """Alias useful to harness timeout implementations."""

        return self.last_activity_at

    def seconds_since_activity(self, *, now: datetime | None = None) -> float | None:
        last = self.last_activity_at
        if last is None:
            return None
        return max(0.0, (_timestamp(now) - last).total_seconds())

    def append(
        self,
        event: TraceEvent | Mapping[str, Any] | str,
        content: Any = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """Append a fully formed event, or construct one from a type/content."""

        if isinstance(event, str):
            event = TraceEvent(type=event, content=content, **kwargs)
        elif isinstance(event, Mapping):
            if content is not None or kwargs:
                raise TypeError("content/keyword fields cannot accompany an event mapping")
            event = TraceEvent.from_dict(event)
        elif content is not None or kwargs:
            raise TypeError("content/keyword fields cannot accompany a TraceEvent")
        with self._lock:
            if self.persist:
                self._write_event(event)
            self._events.append(event)
        return event

    add = append
    record_event = append

    def add_message(self, role: str, content: Any, **metadata: JsonValue) -> TraceEvent:
        return self.append("message", content, role=role, metadata=metadata)

    def add_answer(self, content: Any, **metadata: JsonValue) -> TraceEvent:
        return self.append("answer", content, role="assistant", metadata=metadata)

    def add_reasoning(self, content: Any, **metadata: JsonValue) -> TraceEvent:
        return self.append("reasoning", content, role="assistant", metadata=metadata)

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Any,
        *,
        tool_call_id: str | None = None,
        **metadata: JsonValue,
    ) -> TraceEvent:
        return self.append(
            "tool_call",
            arguments,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        tool_call_id: str | None = None,
        **metadata: JsonValue,
    ) -> TraceEvent:
        return self.append(
            "tool_result", result, tool_name=tool_name, tool_call_id=tool_call_id, metadata=metadata
        )

    def add_usage(self, usage: TokenUsage | None = None, **values: Any) -> TraceEvent:
        if usage is not None and values:
            raise TypeError("pass either TokenUsage or token keyword values")
        if "prompt_tokens" in values:
            values["input_tokens"] = values.pop("prompt_tokens")
        if "completion_tokens" in values:
            values["output_tokens"] = values.pop("completion_tokens")
        return self.append("usage", usage=usage or TokenUsage(**values))

    def flush(self) -> Path:
        """Durably append local events without overwriting concurrent writers."""

        with self._lock, self._locked_storage() as storage_fd:
            disk_events = list(self._read_events_fd(storage_fd))
            local_events = list(self._events)
            self._known_file_state = None
            self._known_event_count = None
            if local_events == disk_events:
                self._fsync_trace(storage_fd)
            elif local_events == disk_events[: len(local_events)]:
                # Another writer appended since this instance last observed the
                # file. Adopt those durable events rather than replacing them.
                self._events[:] = disk_events
                self._fsync_trace(storage_fd)
            elif disk_events == local_events[: len(disk_events)]:
                for event in local_events[len(disk_events) :]:
                    self._append_event_locked(storage_fd, event)
            else:
                raise RuntimeError("trace changed concurrently; refusing to discard events")
            self._known_file_state = self._trace_file_state(storage_fd)
            self._known_event_count = len(self._events)
        return self.path

    save = flush

    def _record(self, event: TraceEvent) -> str:
        record: dict[str, JsonValue] = {
            "schema_version": 1,
            "session_id": self.session_id,
            "event": event.to_dict(),
        }
        if self.harness is not None:
            record["harness"] = self.harness
        if self.model is not None:
            record["model"] = self.model
        if self.metadata:
            record["trace_metadata"] = self.metadata
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def _write_event(self, event: TraceEvent) -> None:
        with self._locked_storage() as storage_fd:
            current_state = self._trace_file_state(storage_fd)
            if (
                current_state != self._known_file_state
                or len(self._events) != self._known_event_count
            ):
                disk_events = list(self._read_events_fd(storage_fd))
                if self._events == disk_events[: len(self._events)]:
                    self._events[:] = disk_events
                elif self._events != disk_events:
                    raise RuntimeError(
                        "trace changed concurrently; refusing an inconsistent append"
                    )
            # An interrupted write/fsync can leave bytes on disk even though
            # append() raises. Force reconciliation before the next attempt.
            self._known_file_state = None
            self._known_event_count = None
            self._known_file_state = self._append_event_locked(storage_fd, event)
            # append() adds this event to memory after persistence succeeds.
            self._known_event_count = len(self._events) + 1

    @staticmethod
    def _file_state(fd: int) -> tuple[int, int, int, int, int]:
        value = os.fstat(fd)
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def _trace_file_state(self, storage_fd: int) -> tuple[int, int, int, int, int] | None:
        try:
            fd = self._open_trace(storage_fd, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            return self._file_state(fd)
        finally:
            os.close(fd)

    @property
    def _filename(self) -> str:
        return f"{self.session_id}.jsonl"

    @contextmanager
    def _locked_storage(self) -> Iterator[int]:
        storage_fd = self._open_storage_dir()
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        lock_fd = -1
        try:
            lock_fd = os.open(
                f".{self.session_id}.lock",
                os.O_RDWR | os.O_CREAT | no_follow,
                0o600,
                dir_fd=storage_fd,
            )
            self._validate_private_file(lock_fd, "trace lock")
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield storage_fd
        finally:
            if lock_fd >= 0:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            os.close(storage_fd)

    def _open_storage_dir(self) -> int:
        """Open/create storage beneath root without following directory links."""

        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        flags = os.O_RDONLY | directory | close_on_exec | no_follow
        current_fd = os.open(self.root, flags)
        relative = self.storage_dir.relative_to(self.root)
        try:
            for part in relative.parts:
                try:
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                except FileNotFoundError:
                    with suppress(FileExistsError):
                        os.mkdir(part, mode=0o700, dir_fd=current_fd)
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                os.fchmod(next_fd, 0o700)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    def _open_trace(self, storage_fd: int, flags: int, *, create: bool = False) -> int:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        if create:
            flags |= os.O_CREAT
        fd = os.open(
            self._filename,
            flags | no_follow | close_on_exec,
            0o600,
            dir_fd=storage_fd,
        )
        try:
            self._validate_private_file(fd, "trace")
            # fchmod updates ctime even when the mode is already correct. Avoid
            # invalidating the append cache merely by opening the trace.
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                os.fchmod(fd, 0o600)
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _validate_private_file(fd: int, description: str) -> None:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError(f"{description} path is not a regular file")
        if file_stat.st_nlink != 1:
            raise OSError(f"{description} file has unexpected hard links")
        if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
            raise PermissionError(f"{description} file is owned by another user")

    def _append_event_locked(
        self, storage_fd: int, event: TraceEvent
    ) -> tuple[int, int, int, int, int]:
        fd = self._open_trace(storage_fd, os.O_RDWR | os.O_APPEND, create=True)
        try:
            self._repair_incomplete_tail(fd)
            data = (self._record(event) + "\n").encode("utf-8")
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written == 0:
                    raise OSError("short write while persisting trace")
                view = view[written:]
            os.fsync(fd)
            return self._file_state(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _repair_incomplete_tail(fd: int) -> None:
        size = os.lseek(fd, 0, os.SEEK_END)
        if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
            return
        position = size
        suffix = b""
        cutoff = 0
        while position:
            count = min(position, 64 * 1024)
            position -= count
            chunk = os.pread(fd, count, position)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                cutoff = position + newline + 1
                suffix = chunk[newline + 1 :] + suffix
                break
            suffix = chunk + suffix
        try:
            json.loads(suffix)
        except (UnicodeDecodeError, json.JSONDecodeError):
            os.ftruncate(fd, cutoff)
        else:
            os.write(fd, b"\n")

    def _fsync_trace(self, storage_fd: int) -> None:
        try:
            fd = self._open_trace(storage_fd, os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _read_events_fd(self, storage_fd: int) -> Iterable[TraceEvent]:
        try:
            fd = self._open_trace(storage_fd, os.O_RDONLY)
        except FileNotFoundError:
            return ()
        try:
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
                return tuple(self._parse_event_lines(handle, self.path))
        finally:
            os.close(fd)

    @staticmethod
    def _read_events(path: Path) -> Iterable[TraceEvent]:
        with path.open("r", encoding="utf-8") as handle:
            yield from Trace._parse_event_lines(handle, path)

    @staticmethod
    def _parse_event_lines(handle: Iterable[str], path: Path) -> Iterable[TraceEvent]:
        previous: tuple[int, str] | None = None
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if previous is not None:
                event = Trace._parse_event_line(*previous, path=path, final=False)
                if event is not None:
                    yield event
            previous = (line_number, line)
        if previous is not None:
            event = Trace._parse_event_line(*previous, path=path, final=True)
            if event is not None:
                yield event

    @staticmethod
    def _parse_event_line(
        line_number: int,
        line: str,
        *,
        path: Path,
        final: bool,
    ) -> TraceEvent | None:
        try:
            record = json.loads(line)
            return TraceEvent.from_dict(record.get("event", record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if final and not line.endswith(("\n", "\r")):
                # A process can die between writes. The next append repairs the
                # incomplete tail while all prior complete records stay usable.
                return None
            raise ValueError(f"invalid trace record at {path}:{line_number}") from error

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        root: str | os.PathLike[str] | None = None,
    ) -> Trace:
        trace_path = Path(path).expanduser().resolve()
        invocation_root = (
            Path(root).expanduser().resolve() if root is not None else trace_path.parent
        )
        if trace_path != invocation_root and invocation_root not in trace_path.parents:
            raise ValueError("trace path must be inside the invocation root")
        with trace_path.open("r", encoding="utf-8") as handle:
            first_record = next((json.loads(line) for line in handle if line.strip()), None)
        session_id = (
            str(first_record.get("session_id", trace_path.stem))
            if first_record
            else trace_path.stem
        )
        return cls(
            session_id,
            root=invocation_root,
            storage_dir=trace_path.parent,
            harness=first_record.get("harness") if first_record else None,
            model=first_record.get("model") if first_record else None,
            metadata=first_record.get("trace_metadata", {}) if first_record else None,
            persist=True,
            load_existing=True,
        )
