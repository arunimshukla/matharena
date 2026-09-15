"""Version-pinned, source-first ingestion for arXiv papers.

This module deliberately does not compile TeX.  arXiv source archives are
untrusted input: extraction accepts only regular files and directories, and
``\\input``/``\\include`` expansion is constrained to the extracted tree.

Downloaded source is intended for local benchmark curation.  A caller must
check the paper's licence before redistributing either the source or excerpts.
The default arXiv-friendly request interval is enforced across calls in the
same process; independently running processes must coordinate their pacing.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable
from urllib.parse import quote, urlparse

import requests


DEFAULT_USER_AGENT = (
    "MathArena-SourceFirst/1.0 "
    "(source-first benchmark research; https://github.com/eth-sri/matharena)"
)

_RATE_LIMIT_LOCK = threading.Lock()
_LAST_REQUEST_BY_ORIGIN: dict[str, float] = {}


class ArxivSourceError(RuntimeError):
    """Base class for source-ingestion errors."""


class DownloadError(ArxivSourceError):
    """An artifact could not be downloaded within the configured limits."""


class UnsafeArchiveError(ArxivSourceError):
    """An archive or TeX reference attempted an unsafe filesystem operation."""


class SourceFormatError(ArxivSourceError):
    """The downloaded source could not be interpreted deterministically."""


@dataclass(frozen=True)
class ArxivVersion:
    """A normalized arXiv identifier pinned to one immutable version."""

    base_id: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.base_id, str) or not self.base_id:
            raise ValueError("base_id must be a non-empty arXiv identifier")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version must be a positive integer")
        modern = re.fullmatch(r"(?P<year>\d{2})(?P<month>\d{2})\.\d{4,5}", self.base_id)
        legacy = re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]*/\d{7}", self.base_id)
        if modern:
            if not 1 <= int(modern.group("month")) <= 12:
                raise ValueError("base_id contains an invalid month")
        elif legacy is None:
            raise ValueError(f"invalid arXiv base identifier: {self.base_id!r}")

    @property
    def canonical_id(self) -> str:
        return f"{self.base_id}v{self.version}"

    @property
    def storage_key(self) -> str:
        """A slash-free name suitable for caller-created paper directories."""

        return self.canonical_id.replace("/", "__")

    def __str__(self) -> str:
        return self.canonical_id


@dataclass(frozen=True)
class DownloadLimits:
    """Network and artifact limits for one source/PDF pair."""

    max_source_bytes: int = 100 * 1024 * 1024
    max_pdf_bytes: int = 100 * 1024 * 1024
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    total_timeout_seconds: float = 180.0
    min_request_interval_seconds: float = 3.0
    chunk_size: int = 64 * 1024

    def __post_init__(self) -> None:
        positive = {
            "max_source_bytes": self.max_source_bytes,
            "max_pdf_bytes": self.max_pdf_bytes,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "chunk_size": self.chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")


@dataclass(frozen=True)
class ExtractionLimits:
    """Limits applied to expanded archives and combined TeX."""

    max_members: int = 5_000
    max_source_archive_bytes: int = 100 * 1024 * 1024
    max_archive_stream_bytes: int = 320 * 1024 * 1024
    max_compression_header_bytes: int = 64 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_file_bytes: int = 64 * 1024 * 1024
    max_path_length: int = 1_024
    max_path_depth: int = 64
    max_include_depth: int = 64
    max_include_directives: int = 10_000
    max_unresolved_includes: int = 1_000
    max_combined_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CombinedSource:
    """Result of recursively expanding local TeX includes."""

    text: str
    included_files: tuple[str, ...]
    unresolved_includes: tuple[dict[str, object], ...]


_MODERN_ID_RE = re.compile(r"(?P<year>\d{2})(?P<month>\d{2})\.(?P<number>\d{4,5})v(?P<version>[1-9]\d*)")
_LEGACY_ID_RE = re.compile(
    r"(?P<archive>[A-Za-z][A-Za-z0-9.-]*)/(?P<number>\d{7})v(?P<version>[1-9]\d*)"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def parse_arxiv_version(value: str) -> ArxivVersion:
    """Parse an arXiv ID or arxiv.org URL and require an explicit ``vN``.

    Both modern IDs (``2608.12345v1``) and legacy IDs
    (``hep-th/9901001v2``) are supported.  Unversioned identifiers are rejected
    so a later paper revision cannot silently change a benchmark item.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("arXiv identifier must be a non-empty string")
    candidate = value.strip()
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:].strip()

    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError("only HTTPS arxiv.org URLs are accepted")
        if parsed.query or parsed.fragment:
            raise ValueError("arXiv URL must not contain a query or fragment")
        path = parsed.path.strip("/")
        for prefix in ("abs/", "pdf/", "src/"):
            if path.startswith(prefix):
                path = path[len(prefix) :]
                break
        else:
            raise ValueError("arXiv URL must use an /abs/, /pdf/, or /src/ path")
        candidate = path

    if candidate.endswith(".pdf"):
        candidate = candidate[:-4]

    modern = _MODERN_ID_RE.fullmatch(candidate)
    if modern:
        month = int(modern.group("month"))
        if not 1 <= month <= 12:
            raise ValueError(f"invalid month in arXiv identifier: {value!r}")
        base_id = f"{modern.group('year')}{modern.group('month')}.{modern.group('number')}"
        return ArxivVersion(base_id=base_id, version=int(modern.group("version")))

    legacy = _LEGACY_ID_RE.fullmatch(candidate)
    if legacy:
        return ArxivVersion(
            base_id=f"{legacy.group('archive')}/{legacy.group('number')}",
            version=int(legacy.group("version")),
        )

    if re.search(r"v\d+$", candidate) is None:
        raise ValueError(f"arXiv identifier must pin an explicit version (for example {candidate}v1)")
    raise ValueError(f"invalid versioned arXiv identifier: {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_metadata(
    path: Path,
    *,
    relative_to: Path,
    url: str,
    content_type: str | None,
    reused: bool,
) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "url": url,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "content_type": content_type,
        "reused": reused,
    }


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _validate_artifact(path: Path, kind: str) -> None:
    if path.stat().st_size == 0:
        raise DownloadError(f"downloaded {kind} artifact is empty")
    if _looks_like_html(path):
        raise DownloadError(f"downloaded {kind} artifact is an HTML response, not a paper artifact")
    if kind == "pdf":
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise DownloadError("downloaded PDF does not have a PDF header")


def _download_file(
    session: requests.Session,
    url: str,
    target: Path,
    *,
    kind: str,
    max_bytes: int,
    limits: DownloadLimits,
    user_agent: str,
    overwrite: bool,
    min_request_interval_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> tuple[dict[str, object], bool]:
    if target.exists() and not overwrite:
        if not target.is_file():
            raise DownloadError(f"artifact path is not a regular file: {target}")
        if target.stat().st_size > max_bytes:
            raise DownloadError(f"cached {kind} exceeds the configured size limit")
        _validate_artifact(target, kind)
        return (
            _artifact_metadata(
                target,
                relative_to=target.parent,
                url=url,
                content_type=None,
                reused=True,
            ),
            False,
        )

    part = target.with_name(f".{target.name}.part")
    part.unlink(missing_ok=True)
    content_type: str | None = None
    try:
        if min_request_interval_seconds:
            origin = urlparse(url).netloc.lower()
            with _RATE_LIMIT_LOCK:
                now = monotonic()
                previous = _LAST_REQUEST_BY_ORIGIN.get(origin)
                if previous is not None:
                    remaining = min_request_interval_seconds - (now - previous)
                    if remaining > 0:
                        sleep(remaining)
                        now = monotonic()
                # Record request start while holding the lock so concurrent
                # pilot workers cannot start requests to arXiv simultaneously.
                _LAST_REQUEST_BY_ORIGIN[origin] = max(now, (previous or now))
        # Deliberate rate-limit waiting is not part of the artifact's transfer
        # timeout.  Connect and read timeouts still apply inside requests.get.
        started = monotonic()
        response = session.get(
            url,
            stream=True,
            timeout=(limits.connect_timeout_seconds, limits.read_timeout_seconds),
            headers={"User-Agent": user_agent, "Accept-Encoding": "identity"},
        )
        try:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type")
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    content_length = None
                if content_length is not None and content_length > max_bytes:
                    raise DownloadError(
                        f"{kind} Content-Length {content_length} exceeds limit {max_bytes}"
                    )

            received = 0
            with part.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=limits.chunk_size):
                    if monotonic() - started > limits.total_timeout_seconds:
                        raise DownloadError(
                            f"{kind} download exceeded {limits.total_timeout_seconds:g}s total timeout"
                        )
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise DownloadError(f"{kind} download exceeds limit {max_bytes} bytes")
                    handle.write(chunk)
        finally:
            response.close()
        _validate_artifact(part, kind)
        os.replace(part, target)
    except DownloadError:
        raise
    except requests.RequestException as exc:
        raise DownloadError(f"failed to download {kind} from {url}: {exc}") from exc
    except OSError as exc:
        raise DownloadError(f"failed to store {kind} artifact at {target}: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)

    return (
        _artifact_metadata(
            target,
            relative_to=target.parent,
            url=url,
            content_type=content_type,
            reused=False,
        ),
        True,
    )


def download_arxiv_artifacts(
    arxiv_id: str | ArxivVersion,
    paper_dir: str | Path,
    *,
    session: requests.Session | None = None,
    base_url: str = "https://arxiv.org",
    limits: DownloadLimits | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    include_pdf: bool = True,
    overwrite: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, dict[str, object]]:
    """Download exact-version source and, optionally, PDF artifacts.

    ``paper_dir/source.raw`` and, when requested, ``paper_dir/paper.pdf`` are
    written atomically.  Source-first benchmark generation should set
    ``include_pdf=False``; the PDF option remains for manual-review workflows.
    A process-wide, per-origin limiter applies the default three-second interval
    both within and across invocations.  Multiple processes must provide their
    own shared limiter.
    """

    version = arxiv_id if isinstance(arxiv_id, ArxivVersion) else parse_arxiv_version(arxiv_id)
    limits = limits or DownloadLimits()
    if not user_agent.strip():
        raise ValueError("user_agent must identify the client")
    base_url = base_url.rstrip("/")
    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")

    directory = Path(paper_dir)
    directory.mkdir(parents=True, exist_ok=True)
    encoded_id = quote(version.canonical_id, safe="/")
    source_url = f"{base_url}/src/{encoded_id}"

    owns_session = session is None
    active_session = session or requests.Session()
    try:
        source_meta, _ = _download_file(
            active_session,
            source_url,
            directory / "source.raw",
            kind="source",
            max_bytes=limits.max_source_bytes,
            limits=limits,
            user_agent=user_agent,
            overwrite=overwrite,
            min_request_interval_seconds=limits.min_request_interval_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        artifacts = {"source": source_meta}
        if include_pdf:
            pdf_url = f"{base_url}/pdf/{encoded_id}.pdf"
            pdf_meta, _ = _download_file(
                active_session,
                pdf_url,
                directory / "paper.pdf",
                kind="pdf",
                max_bytes=limits.max_pdf_bytes,
                limits=limits,
                user_agent=user_agent,
                overwrite=overwrite,
                min_request_interval_seconds=limits.min_request_interval_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            artifacts["pdf"] = pdf_meta
    finally:
        if owns_session:
            active_session.close()

    # _download_file reports paths relative to each artifact's parent, which is
    # paper_dir for every downloaded artifact.
    return artifacts


def _safe_archive_path(raw_name: str, limits: ExtractionLimits) -> PurePosixPath | None:
    if not raw_name or "\x00" in raw_name:
        raise UnsafeArchiveError("archive contains an empty or NUL-containing path")
    if len(raw_name) > limits.max_path_length:
        raise UnsafeArchiveError(f"archive path exceeds {limits.max_path_length} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_name):
        raise UnsafeArchiveError(f"archive path contains a control character: {raw_name!r}")
    if "\\" in raw_name:
        raise UnsafeArchiveError(f"archive path uses a backslash: {raw_name!r}")
    if raw_name.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(raw_name):
        raise UnsafeArchiveError(f"archive contains an absolute path: {raw_name!r}")
    raw_parts = raw_name.split("/")
    if ".." in raw_parts:
        raise UnsafeArchiveError(f"archive path traverses outside its root: {raw_name!r}")
    normalized = PurePosixPath(raw_name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise UnsafeArchiveError(f"archive path traverses outside its root: {raw_name!r}")
    parts = tuple(part for part in normalized.parts if part not in {"", "."})
    if not parts:
        return None
    if len(parts) > limits.max_path_depth:
        raise UnsafeArchiveError(f"archive path is nested too deeply: {raw_name!r}")
    return PurePosixPath(*parts)


def _copy_exactly(source: BinaryIO, target: BinaryIO, expected: int) -> None:
    remaining = expected
    while remaining:
        chunk = source.read(min(64 * 1024, remaining))
        if not chunk:
            raise SourceFormatError("archive member ended before its declared size")
        target.write(chunk)
        remaining -= len(chunk)


def _tar_members(archive: tarfile.TarFile, limits: ExtractionLimits) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
    seen: set[str] = set()
    total_bytes = 0
    member_count = 0
    for member in archive:
        member_count += 1
        if member_count > limits.max_members:
            raise UnsafeArchiveError(f"archive contains more than {limits.max_members} members")
        if member.issym() or member.islnk():
            raise UnsafeArchiveError(f"archive contains a link: {member.name!r}")
        if member.isdev() or member.isfifo():
            raise UnsafeArchiveError(f"archive contains a device or FIFO: {member.name!r}")
        if not member.isdir() and member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
            raise UnsafeArchiveError(f"archive contains an unsupported member type: {member.name!r}")
        relative = _safe_archive_path(member.name, limits)
        if relative is None:
            if member.isdir():
                continue
            raise UnsafeArchiveError("archive has a regular file with an empty path")
        normalized_name = relative.as_posix()
        if normalized_name in seen:
            raise UnsafeArchiveError(f"archive contains duplicate path: {normalized_name!r}")
        seen.add(normalized_name)
        if member.size < 0:
            raise UnsafeArchiveError(f"archive member has a negative size: {member.name!r}")
        if member.isdir() and member.size:
            raise UnsafeArchiveError(f"archive directory has a nonzero size: {member.name!r}")
        if not member.isdir():
            if member.size > limits.max_file_bytes:
                raise UnsafeArchiveError(
                    f"archive member {member.name!r} exceeds {limits.max_file_bytes} bytes"
                )
            total_bytes += member.size
            if total_bytes > limits.max_total_bytes:
                raise UnsafeArchiveError(
                    f"archive expands beyond {limits.max_total_bytes} total bytes"
                )
        members.append((member, relative))
    if not members:
        raise SourceFormatError("source archive contains no files")
    return members


def _extract_tar(source_path: Path, stage: Path, limits: ExtractionLimits) -> None:
    try:
        # Compression and raw-header validation have already happened.  Using
        # r: (not r:*) prevents tarfile from invoking another decompressor.
        with tarfile.open(source_path, mode="r:") as archive:
            members = _tar_members(archive, limits)
            root = stage.resolve()
            for member, relative in members:
                destination = stage.joinpath(*relative.parts)
                resolved = destination.resolve(strict=False)
                if not resolved.is_relative_to(root):
                    raise UnsafeArchiveError(f"archive path escapes destination: {member.name!r}")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SourceFormatError(f"cannot read archive member: {member.name!r}")
                try:
                    with destination.open("xb") as output:
                        _copy_exactly(extracted, output, member.size)
                finally:
                    extracted.close()
    except (tarfile.TarError, EOFError) as exc:
        raise SourceFormatError(f"invalid tar source archive: {exc}") from exc
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            raise UnsafeArchiveError(f"archive contains conflicting paths: {exc}") from exc
        raise SourceFormatError(f"failed to extract source archive: {exc}") from exc


def _compression_kind(path: Path) -> str | None:
    with path.open("rb") as handle:
        prefix = handle.read(6)
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if prefix.startswith(b"BZh"):
        return "bzip2"
    if prefix.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return None


def _read_exact_header(handle: BinaryIO, length: int, description: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise SourceFormatError(f"truncated gzip {description}")
    return value


def _validate_gzip_header(path: Path, limits: ExtractionLimits) -> None:
    """Bound optional gzip metadata before handing bytes to zlib."""

    with path.open("rb") as handle:
        fixed = _read_exact_header(handle, 10, "header")
        if fixed[:3] != b"\x1f\x8b\x08":
            raise SourceFormatError("invalid or unsupported gzip header")
        flags = fixed[3]
        if flags & 0xE0:
            raise SourceFormatError("gzip header uses reserved flags")
        consumed = 10

        def reserve(length: int) -> None:
            nonlocal consumed
            consumed += length
            if consumed > limits.max_compression_header_bytes:
                raise UnsafeArchiveError(
                    "gzip metadata exceeds "
                    f"{limits.max_compression_header_bytes} bytes"
                )

        if flags & 0x04:  # FEXTRA
            reserve(2)
            extra_length = int.from_bytes(_read_exact_header(handle, 2, "extra length"), "little")
            reserve(extra_length)
            _read_exact_header(handle, extra_length, "extra field")
        for flag, description in ((0x08, "filename"), (0x10, "comment")):
            if not flags & flag:
                continue
            while True:
                reserve(1)
                if _read_exact_header(handle, 1, description) == b"\x00":
                    break
        if flags & 0x02:  # FHCRC
            reserve(2)
            _read_exact_header(handle, 2, "header checksum")


def _write_bounded(output: BinaryIO, chunk: bytes, total: int, limit: int) -> int:
    total += len(chunk)
    if total > limit:
        raise UnsafeArchiveError(f"compressed source expands beyond {limit} bytes")
    output.write(chunk)
    return total


def _decompress_gzip(source: Path, output: BinaryIO, limits: ExtractionLimits) -> None:
    _validate_gzip_header(source, limits)
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    total = 0
    with source.open("rb") as handle:
        while True:
            compressed = handle.read(64 * 1024)
            if not compressed:
                break
            pending = compressed
            while pending:
                remaining = limits.max_archive_stream_bytes - total
                expanded = decompressor.decompress(pending, max(1, remaining + 1))
                total = _write_bounded(output, expanded, total, limits.max_archive_stream_bytes)
                pending = decompressor.unconsumed_tail
                if decompressor.eof:
                    if decompressor.unused_data or pending or handle.read(1):
                        raise UnsafeArchiveError(
                            "concatenated gzip members or trailing data are not supported"
                        )
                    break
            if decompressor.eof:
                break
    if not decompressor.eof:
        raise SourceFormatError("truncated gzip source artifact")


def _decompress_bzip2(source: Path, output: BinaryIO, limits: ExtractionLimits) -> None:
    total = 0
    with bz2.open(source, "rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            total = _write_bounded(output, chunk, total, limits.max_archive_stream_bytes)


def _normalize_source_payload(
    source: Path,
    parent: Path,
    limits: ExtractionLimits,
) -> tuple[Path, str | None]:
    """Return an uncompressed, size-bounded payload and its compression kind."""

    compressed_size = source.stat().st_size
    if compressed_size > limits.max_source_archive_bytes:
        raise UnsafeArchiveError(
            f"source artifact exceeds {limits.max_source_archive_bytes} bytes"
        )
    compression = _compression_kind(source)
    if compression == "xz":
        # liblzma may allocate the dictionary advertised by the stream before
        # yielding output.  Rejecting XZ avoids attacker-controlled dictionary
        # memory; arXiv's normal source transport is gzip/tar.
        raise UnsafeArchiveError(
            "XZ source artifacts are unsupported because their dictionary metadata is not safely bounded"
        )
    if compression is None:
        if compressed_size > limits.max_archive_stream_bytes:
            raise UnsafeArchiveError(
                f"source payload exceeds {limits.max_archive_stream_bytes} bytes"
            )
        return source, None

    handle = tempfile.NamedTemporaryFile(
        prefix=".arxiv-source-payload-",
        suffix=".raw",
        dir=parent,
        delete=False,
    )
    normalized = Path(handle.name)
    try:
        with handle:
            if compression == "gzip":
                _decompress_gzip(source, handle, limits)
            elif compression == "bzip2":
                _decompress_bzip2(source, handle, limits)
            else:  # pragma: no cover - kept defensive if detection grows.
                raise AssertionError(f"unsupported compression kind: {compression}")
        return normalized, compression
    except (OSError, EOFError, zlib.error) as exc:
        normalized.unlink(missing_ok=True)
        raise SourceFormatError(f"invalid {compression} source artifact: {exc}") from exc
    except Exception:
        normalized.unlink(missing_ok=True)
        raise


_RISKY_TAR_HEADER_TYPES = {
    b"x": "POSIX PAX extended header",
    b"g": "POSIX PAX global header",
    b"L": "GNU long-name header",
    b"K": "GNU long-link header",
    b"S": "GNU sparse header",
    b"X": "Solaris extended header",
}


def _tar_number(field: bytes, description: str) -> int:
    try:
        value = tarfile.nti(field)
    except (tarfile.InvalidHeaderError, ValueError) as exc:
        raise SourceFormatError(f"invalid tar {description}") from exc
    if not isinstance(value, int) or value < 0:
        raise SourceFormatError(f"invalid tar {description}")
    return value


def _valid_tar_checksum(header: bytes) -> bool:
    try:
        stored = _tar_number(header[148:156], "checksum")
    except SourceFormatError:
        return False
    unsigned = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
    signed = sum(byte if byte < 128 else byte - 256 for byte in header[:148])
    signed += 8 * ord(" ")
    signed += sum(byte if byte < 128 else byte - 256 for byte in header[156:])
    return stored in {unsigned, signed}


def _preflight_raw_tar(path: Path, limits: ExtractionLimits) -> bool:
    """Recognize and bound raw tar records before ``tarfile`` parses them.

    PAX/GNU metadata pseudo-members are rejected.  They are normally consumed
    internally by :mod:`tarfile` before a logical member is yielded, which
    would otherwise bypass member/path limits.
    """

    stream_size = path.stat().st_size
    if stream_size > limits.max_archive_stream_bytes:
        raise UnsafeArchiveError(
            f"source payload exceeds {limits.max_archive_stream_bytes} bytes"
        )
    if stream_size < tarfile.BLOCKSIZE:
        return False

    with path.open("rb") as handle:
        offset = 0
        members = 0
        saw_header = False
        while offset < stream_size:
            header = handle.read(tarfile.BLOCKSIZE)
            if len(header) != tarfile.BLOCKSIZE:
                if not saw_header:
                    return False
                raise SourceFormatError("tar source ends with a partial header")
            offset += tarfile.BLOCKSIZE
            if header == tarfile.NUL * tarfile.BLOCKSIZE:
                # A valid tar ends in zero blocks.  Reject hidden nonzero data
                # after the terminator instead of asking tarfile to interpret it.
                while True:
                    trailing = handle.read(64 * 1024)
                    if not trailing:
                        break
                    if trailing.strip(tarfile.NUL):
                        raise SourceFormatError("tar source has nonzero data after its terminator")
                return True
            if not _valid_tar_checksum(header):
                if not saw_header:
                    return False
                raise SourceFormatError("tar source contains an invalid header checksum")
            saw_header = True
            members += 1
            if members > limits.max_members:
                raise UnsafeArchiveError(f"archive contains more than {limits.max_members} members")
            type_flag = header[156:157] or tarfile.REGTYPE
            if type_flag in _RISKY_TAR_HEADER_TYPES:
                raise UnsafeArchiveError(
                    f"tar source uses unsupported {_RISKY_TAR_HEADER_TYPES[type_flag]}"
                )
            member_size = _tar_number(header[124:136], "member size")
            padded_size = ((member_size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
            if offset + padded_size > stream_size:
                raise SourceFormatError("tar member extends beyond the bounded source payload")
            if offset + padded_size > limits.max_archive_stream_bytes:
                raise UnsafeArchiveError(
                    f"tar stream exceeds {limits.max_archive_stream_bytes} bytes"
                )
            handle.seek(padded_size, os.SEEK_CUR)
            offset += padded_size
    return saw_header


def _is_probably_tex(prefix: bytes) -> bool:
    if b"\x00" in prefix:
        return False
    lowered = prefix.lower()
    return any(
        token in lowered
        for token in (b"\\documentclass", b"\\documentstyle", b"\\begin{document}", b"\\input", b"\\def")
    )


def _extract_single_tex(
    source_path: Path,
    stage: Path,
    limits: ExtractionLimits,
    compression: str | None,
) -> str:
    target = stage / "main.tex"
    total = 0
    try:
        format_name = "plain_tex" if compression is None else f"{compression}_tex"
        with source_path.open("rb") as input_handle, target.open("xb") as output:
            prefix = b""
            while True:
                chunk = input_handle.read(64 * 1024)
                if not chunk:
                    break
                if len(prefix) < 64 * 1024:
                    prefix += chunk[: 64 * 1024 - len(prefix)]
                total += len(chunk)
                if total > limits.max_file_bytes or total > limits.max_total_bytes:
                    raise UnsafeArchiveError("single TeX source expands beyond the configured size limit")
                output.write(chunk)
        if not _is_probably_tex(prefix):
            raise SourceFormatError("source artifact is neither a tar archive nor recognizable TeX")
        return format_name
    except (OSError, EOFError) as exc:
        raise SourceFormatError(f"invalid {compression or 'plain'} source artifact: {exc}") from exc


def _file_inventory(root: Path) -> tuple[list[dict[str, object]], int]:
    files: list[dict[str, object]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return files, total


def extract_source(
    source_path: str | Path,
    destination: str | Path,
    *,
    limits: ExtractionLimits | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Safely extract a tar archive or a single compressed/plain TeX file.

    Extraction happens in a sibling staging directory and is only made visible
    after every member has passed validation.  Links, devices, path traversal,
    duplicate normalized names, and excessive archives are rejected.  Gzip and
    bzip2 are decompressed into a bounded raw stream; XZ and PAX/GNU extended
    tar metadata are deliberately rejected before parser allocation.
    """

    limits = limits or ExtractionLimits()
    source = Path(source_path)
    target = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    resolved_target = target.resolve()
    if resolved_target == Path(resolved_target.anchor):
        raise ValueError("extraction destination cannot be a filesystem root")
    if target.exists() and not overwrite:
        raise FileExistsError(f"extraction destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    normalized: Path | None = None
    try:
        payload, compression = _normalize_source_payload(source, target.parent, limits)
        if payload != source:
            normalized = payload
        is_tar = _preflight_raw_tar(payload, limits)

        if is_tar:
            _extract_tar(payload, stage, limits)
            format_name = "tar"
        else:
            format_name = _extract_single_tex(payload, stage, limits, compression)

        files, total_bytes = _file_inventory(stage)
        if not files:
            raise SourceFormatError("source artifact produced no regular files")
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"extraction destination already exists: {target}")
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(stage, target)
        return {
            "format": format_name,
            "root": target.name,
            "files": files,
            "total_bytes": total_bytes,
        }
    finally:
        if normalized is not None:
            normalized.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


def _mask_tex_comments(text: str) -> str:
    """Replace TeX comments with spaces while retaining offsets/newlines."""

    masked: list[str] = []
    in_comment = False
    backslashes = 0
    for char in text:
        if in_comment:
            if char in "\r\n":
                in_comment = False
                masked.append(char)
            else:
                masked.append(" ")
            backslashes = 0
            continue
        if char == "%" and backslashes % 2 == 0:
            in_comment = True
            masked.append(" ")
            backslashes = 0
            continue
        masked.append(char)
        if char == "\\":
            backslashes += 1
        else:
            backslashes = 0
    return "".join(masked)


def _mask_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    """Blank non-newline characters in sorted or overlapping ranges."""

    if not ranges:
        return text
    characters = list(text)
    for start, end in ranges:
        for index in range(max(0, start), min(len(characters), end)):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


_PROTECTED_ENVIRONMENT_RE = re.compile(
    r"\\begin\s*\{(?P<name>"
    r"verbatim\*?|Verbatim\*?|BVerbatim|LVerbatim|SaveVerbatim|"
    r"lstlisting\*?|minted\*?|comment|filecontents\*?"
    r")\}"
)
_VERB_COMMAND_RE = re.compile(r"\\verb\*?(?![A-Za-z@])")
_LATEX_MACRO_DEFINITION_RE = re.compile(
    r"\\(?P<command>"
    r"newcommand|renewcommand|providecommand|DeclareRobustCommand|"
    r"newenvironment|renewenvironment"
    r")(?![A-Za-z@])\*?"
)
_TEX_DEF_RE = re.compile(r"\\(?:def|gdef|edef|xdef)(?![A-Za-z@])")


def _protected_environment_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        begin = _PROTECTED_ENVIRONMENT_RE.search(text, cursor)
        if begin is None:
            break
        name = begin.group("name")
        end_pattern = re.compile(rf"\\end\s*\{{{re.escape(name)}\}}")
        end = end_pattern.search(text, begin.end())
        stop = len(text) if end is None else end.end()
        ranges.append((begin.start(), stop))
        cursor = stop
    return ranges


def _inline_verb_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        command = _VERB_COMMAND_RE.search(text, cursor)
        if command is None:
            break
        delimiter_index = command.end()
        if delimiter_index >= len(text):
            ranges.append((command.start(), len(text)))
            break
        delimiter = text[delimiter_index]
        if delimiter.isspace() or delimiter.isalpha():
            cursor = command.end()
            continue
        line_end = len(text)
        for newline in (text.find("\n", delimiter_index + 1), text.find("\r", delimiter_index + 1)):
            if newline != -1:
                line_end = min(line_end, newline)
        closing = text.find(delimiter, delimiter_index + 1, line_end)
        stop = line_end if closing == -1 else closing + 1
        ranges.append((command.start(), stop))
        cursor = max(stop, command.end())
    return ranges


def _balanced_group_end(text: str, start: int, opening: str, closing: str) -> int | None:
    if start >= len(text) or text[start] != opening:
        return None
    depth = 0
    backslashes = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "\\":
            backslashes += 1
            continue
        escaped = backslashes % 2 == 1
        backslashes = 0
        if escaped:
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _skip_tex_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _latex_macro_definition_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        match = _LATEX_MACRO_DEFINITION_RE.search(text, cursor)
        if match is None:
            break
        index = _skip_tex_space(text, match.end())
        if index >= len(text):
            break
        if text[index] == "{":
            name_end = _balanced_group_end(text, index, "{", "}")
            if name_end is None:
                ranges.append((match.start(), len(text)))
                return ranges
            index = name_end
        elif text[index] == "\\":
            index += 1
            if index < len(text) and (text[index].isalpha() or text[index] == "@"):
                while index < len(text) and (text[index].isalpha() or text[index] == "@"):
                    index += 1
            elif index < len(text):
                index += 1
        else:
            cursor = match.end()
            continue

        index = _skip_tex_space(text, index)
        for _ in range(2):
            if index >= len(text) or text[index] != "[":
                break
            optional_end = _balanced_group_end(text, index, "[", "]")
            if optional_end is None:
                ranges.append((match.start(), len(text)))
                return ranges
            index = _skip_tex_space(text, optional_end)

        body_count = 2 if match.group("command") in {"newenvironment", "renewenvironment"} else 1
        found_body = False
        for _ in range(body_count):
            if index >= len(text) or text[index] != "{":
                break
            body_end = _balanced_group_end(text, index, "{", "}")
            if body_end is None:
                ranges.append((index, len(text)))
                return ranges
            ranges.append((index, body_end))
            found_body = True
            index = _skip_tex_space(text, body_end)
        cursor = index if found_body else match.end()
    return ranges


def _tex_def_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        match = _TEX_DEF_RE.search(text, cursor)
        if match is None:
            break
        search_end = min(len(text), match.end() + 4096)
        opening = text.find("{", match.end(), search_end)
        if opening == -1:
            cursor = match.end()
            continue
        body_end = _balanced_group_end(text, opening, "{", "}")
        if body_end is None:
            ranges.append((opening, len(text)))
            return ranges
        ranges.append((opening, body_end))
        cursor = body_end
    return ranges


def _mask_tex_for_include_scan(text: str) -> str:
    """Mask comments and common non-executing/protected TeX regions."""

    masked = _mask_tex_comments(text)
    masked = _mask_ranges(masked, _protected_environment_ranges(masked))
    masked = _mask_ranges(masked, _inline_verb_ranges(masked))
    definition_ranges = _latex_macro_definition_ranges(masked)
    definition_ranges.extend(_tex_def_ranges(masked))
    return _mask_ranges(masked, definition_ranges)


_SOURCE_MARKER_SPOOF_RE = re.compile(
    r"^[ \t]*%[ \t]*MATHARENA_SOURCE_(?:BEGIN|END)(?:\s|$)",
    re.MULTILINE,
)


def _read_tex(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise SourceFormatError(f"TeX file exceeds configured limit: {path}")
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise SourceFormatError(f"TeX file contains NUL bytes: {path}")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Older arXiv submissions commonly use ISO-8859-1.  Latin-1 is a
        # deterministic byte-preserving fallback; no external converter runs.
        return raw.decode("latin-1")


def find_main_tex(source_root: str | Path, *, limits: ExtractionLimits | None = None) -> Path:
    """Identify the most likely main TeX document deterministically."""

    limits = limits or ExtractionLimits()
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".tex", ".ltx", ".latex"} and path.is_file() and not path.is_symlink()
    )
    if not candidates:
        raise SourceFormatError("source tree contains no TeX document files")

    contents = {
        path: _mask_tex_for_include_scan(_read_tex(path, limits.max_file_bytes))
        for path in candidates
    }
    referenced_names: set[Path] = set()
    include_re = re.compile(r"\\(?:input|include)\s*\{\s*([^{}]+?)\s*\}")
    for parent, text in contents.items():
        for match in include_re.finditer(text):
            raw = match.group(1)
            if any(character in raw for character in "\\#$~"):
                continue
            child = parent.parent / raw
            if child.suffix == "":
                child = child.with_suffix(".tex")
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if resolved.is_relative_to(root):
                referenced_names.add(resolved)

    scored: list[tuple[int, int, str, Path]] = []
    preferred_names = {"main.tex": 20, "paper.tex": 15, "article.tex": 10, "ms.tex": 10}
    for path, text in contents.items():
        has_documentclass = re.search(r"\\document(?:class|style)(?:\s*\[|\s*\{)", text) is not None
        has_document = re.search(r"\\begin\s*\{document\}", text) is not None
        score = 0
        score += 100 if has_documentclass else 0
        score += 60 if has_document else 0
        score += preferred_names.get(path.name.lower(), 0)
        score += 8 if re.search(r"\\title(?:\s*\[|\s*\{)", text) else 0
        score += 5 if re.search(r"\\author(?:\s*\[|\s*\{)", text) else 0
        score += 20 if path.resolve() not in referenced_names else 0
        scored.append((score, path.stat().st_size, path.relative_to(root).as_posix(), path))

    viable = [entry for entry in scored if entry[0] >= 60]
    if not viable:
        raise SourceFormatError("could not identify a main TeX document")
    # Prefer semantic signals, then the larger complete document, then a stable
    # lexicographic path.  The final key makes selection reproducible.
    viable.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))
    return viable[0][3]


_INCLUDE_RE = re.compile(
    r"\\(?P<command>input|include)(?![A-Za-z@])\s*(?:\{\s*(?P<braced>[^{}]+?)\s*\}|(?P<bare>[^\s%{}]+))"
)


def _resolve_local_include(raw: str, parent: Path, root: Path) -> tuple[Path | None, str | None]:
    value = raw.strip()
    if not value:
        return None, "empty"
    if "\x00" in value or "\n" in value or "\r" in value:
        raise UnsafeArchiveError(f"unsafe TeX include path: {raw!r}")
    if _WINDOWS_ABSOLUTE_RE.match(value) or value.startswith("/"):
        raise UnsafeArchiveError(f"absolute TeX include path is forbidden: {raw!r}")
    if "\\" in value or any(character in value for character in "#$~"):
        return None, "dynamic"
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise UnsafeArchiveError(f"TeX include escapes source root: {raw!r}")
    if len(pure.parts) > 64:
        raise UnsafeArchiveError(f"TeX include path is nested too deeply: {raw!r}")

    base = parent.joinpath(*pure.parts)
    choices = [base]
    if base.suffix == "":
        choices.append(base.with_suffix(".tex"))
    for choice in choices:
        try:
            resolved = choice.resolve()
        except OSError as exc:
            raise UnsafeArchiveError(f"cannot resolve TeX include path {raw!r}: {exc}") from exc
        if not resolved.is_relative_to(root):
            raise UnsafeArchiveError(f"TeX include escapes source root: {raw!r}")
        if resolved.is_file():
            if resolved.is_symlink():
                raise UnsafeArchiveError(f"TeX include is a symbolic link: {raw!r}")
            return resolved, None
    return None, "missing"


def combine_tex_source(
    main_tex: str | Path,
    source_root: str | Path,
    *,
    limits: ExtractionLimits | None = None,
) -> CombinedSource:
    """Recursively inline safe local ``\\input`` and ``\\include`` files.

    File boundaries are retained as ``MATHARENA_SOURCE_BEGIN/END`` comments.
    Dynamic and missing references remain verbatim and are listed in
    ``unresolved_includes``.  Common verbatim/listings regions and macro
    definition bodies are not expanded.  Absolute/traversing references,
    reserved marker spoofing, excessive directives, and cycles fail.
    """

    limits = limits or ExtractionLimits()
    root = Path(source_root).resolve()
    main = Path(main_tex).resolve()
    if not root.is_dir() or not main.is_file() or not main.is_relative_to(root):
        raise ValueError("main_tex must be a regular file inside source_root")
    if main.is_symlink():
        raise UnsafeArchiveError("main TeX document cannot be a symbolic link")

    chunks: list[str] = []
    combined_bytes = 0
    included: list[str] = []
    included_seen: set[str] = set()
    unresolved: list[dict[str, object]] = []
    stack: list[Path] = []
    include_directives = 0

    def append(chunk: str) -> None:
        nonlocal combined_bytes
        combined_bytes += len(chunk.encode("utf-8"))
        if combined_bytes > limits.max_combined_bytes:
            raise SourceFormatError(
                f"combined TeX exceeds {limits.max_combined_bytes} bytes"
            )
        chunks.append(chunk)

    def expand(path: Path, *, via: str | None = None, via_line: int | None = None) -> None:
        nonlocal include_directives
        if len(stack) >= limits.max_include_depth:
            raise SourceFormatError(f"TeX include depth exceeds {limits.max_include_depth}")
        if path in stack:
            cycle = " -> ".join(item.relative_to(root).as_posix() for item in [*stack, path])
            raise SourceFormatError(f"cyclic TeX include detected: {cycle}")
        relative = path.relative_to(root).as_posix()
        if any(ord(character) < 32 or ord(character) == 127 for character in relative):
            raise UnsafeArchiveError(f"TeX source path contains a control character: {relative!r}")
        if path != main and relative not in included_seen:
            included_seen.add(relative)
            included.append(relative)

        marker_context = ""
        if via is not None and via_line is not None:
            marker_context = f" via={via}:{via_line}"
        append(f"% MATHARENA_SOURCE_BEGIN file={relative}{marker_context}\n")
        stack.append(path)
        try:
            text = _read_tex(path, limits.max_file_bytes)
            spoof = _SOURCE_MARKER_SPOOF_RE.search(text)
            if spoof is not None:
                spoof_line = text.count("\n", 0, spoof.start()) + 1
                raise SourceFormatError(
                    f"source file {relative!r} contains a reserved MathArena marker at line {spoof_line}"
                )
            masked = _mask_tex_for_include_scan(text)
            cursor = 0
            source_line = 1
            for match in _INCLUDE_RE.finditer(masked):
                prefix = text[cursor : match.start()]
                append(prefix)
                source_line += prefix.count("\n")
                include_directives += 1
                if include_directives > limits.max_include_directives:
                    raise SourceFormatError(
                        "TeX source contains more than "
                        f"{limits.max_include_directives} expanded include directives"
                    )
                raw_target = match.group("braced") or match.group("bare") or ""
                line = source_line
                directive = text[match.start() : match.end()]
                source_line += directive.count("\n")
                target, reason = _resolve_local_include(raw_target, path.parent, root)
                if target is None:
                    if len(unresolved) >= limits.max_unresolved_includes:
                        raise SourceFormatError(
                            "TeX source contains more than "
                            f"{limits.max_unresolved_includes} unresolved include directives"
                        )
                    unresolved.append(
                        {
                            "source": relative,
                            "line": line,
                            "command": match.group("command"),
                            "target": raw_target,
                            "reason": reason,
                        }
                    )
                    append(directive)
                else:
                    expand(target, via=relative, via_line=line)
                cursor = match.end()
            append(text[cursor:])
        finally:
            stack.pop()
        if chunks and not chunks[-1].endswith("\n"):
            append("\n")
        append(f"% MATHARENA_SOURCE_END file={relative}\n")

    expand(main)
    return CombinedSource(
        text="".join(chunks),
        included_files=tuple(included),
        unresolved_includes=tuple(unresolved),
    )


def _manifest_local_path(paper_dir: Path, value: object) -> Path:
    raw = str(value)
    if not raw or "\\" in raw:
        raise ValueError("manifest path is empty or non-POSIX")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("manifest path escapes paper_dir")
    path = paper_dir.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(paper_dir.resolve()):
        raise ValueError("manifest path escapes paper_dir")
    return path


def _cached_manifest_is_complete(
    manifest: dict[str, object],
    paper_dir: Path,
    version: ArxivVersion,
    *,
    require_pdf: bool,
) -> bool:
    if manifest.get("arxiv_id") != version.canonical_id:
        return False
    try:
        artifacts = manifest["artifacts"]
        tex = manifest["tex"]
        extraction = manifest["extraction"]
        if not isinstance(artifacts, dict) or not isinstance(tex, dict) or not isinstance(extraction, dict):
            return False
        required_artifacts = ("source", "pdf") if require_pdf else ("source",)
        for name in required_artifacts:
            metadata = artifacts[name]
            if not isinstance(metadata, dict):
                return False
            path = _manifest_local_path(paper_dir, metadata["path"])
            if (
                not path.is_file()
                or path.stat().st_size != metadata["bytes"]
                or _sha256_file(path) != metadata["sha256"]
            ):
                return False
        combined = _manifest_local_path(paper_dir, tex["combined_file"])
        if (
            not combined.is_file()
            or combined.stat().st_size != tex["combined_bytes"]
            or _sha256_file(combined) != tex["combined_sha256"]
        ):
            return False
        root = _manifest_local_path(paper_dir, extraction["root"])
        if not root.is_dir() or not isinstance(extraction.get("files"), list):
            return False
        for entry in extraction["files"]:
            if not isinstance(entry, dict):
                return False
            path = _manifest_local_path(paper_dir, entry["path"])
            if (
                not path.is_file()
                or path.stat().st_size != entry["bytes"]
                or _sha256_file(path) != entry["sha256"]
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def prepare_arxiv_source(
    arxiv_id: str | ArxivVersion,
    paper_dir: str | Path,
    *,
    session: requests.Session | None = None,
    base_url: str = "https://arxiv.org",
    download_limits: DownloadLimits | None = None,
    extraction_limits: ExtractionLimits | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    include_pdf: bool = True,
    source_artifact: str | Path | None = None,
    source_origin: str | None = None,
    overwrite: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Prepare an exact-version source tree, combined TeX, and manifest.

    The returned dictionary is also written to ``source_manifest.json``.  It
    contains only local relative paths, hashes, sizes, and provenance—not source
    text—so downstream exported records need not redistribute paper contents.
    ``include_pdf`` is retained for review tools; source-first generation should
    leave it disabled.  ``source_artifact`` imports one already-downloaded raw
    source member, for example from an official arXiv S3 monthly bundle, without
    putting a machine-local path into the manifest.
    """

    version = arxiv_id if isinstance(arxiv_id, ArxivVersion) else parse_arxiv_version(arxiv_id)
    directory = Path(paper_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "source_manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SourceFormatError(f"cannot read cached source manifest: {exc}") from exc
        if cached.get("arxiv_id") != version.canonical_id:
            raise ValueError(
                f"paper_dir already contains {cached.get('arxiv_id')!r}, not {version.canonical_id!r}"
            )
        if _cached_manifest_is_complete(cached, directory, version, require_pdf=include_pdf):
            return cached
        raise SourceFormatError(
            "cached source manifest is incomplete or its artifact hashes changed; use overwrite=True"
        )

    managed_names = ("source.raw", "source", "combined_source.tex")
    all_managed_names = (*managed_names, "paper.pdf")
    if not overwrite:
        partial = [name for name in all_managed_names if (directory / name).exists()]
        if partial:
            raise SourceFormatError(
                "paper_dir contains a partial source preparation "
                f"({', '.join(partial)}); use overwrite=True"
            )
    download_limits = download_limits or DownloadLimits()
    extraction_limits = extraction_limits or ExtractionLimits()
    if source_artifact is not None and include_pdf:
        raise ValueError("source_artifact import requires include_pdf=False")
    stage = Path(tempfile.mkdtemp(prefix=".arxiv-source-stage-", dir=directory))
    try:
        if source_artifact is None:
            artifacts = download_arxiv_artifacts(
                version,
                stage,
                session=session,
                base_url=base_url,
                limits=download_limits,
                user_agent=user_agent,
                include_pdf=include_pdf,
                overwrite=False,
                sleep=sleep,
                monotonic=monotonic,
            )
        else:
            local_source = Path(source_artifact)
            if not local_source.is_file() or local_source.is_symlink():
                raise SourceFormatError("source_artifact must be a regular, non-symlink file")
            size = local_source.stat().st_size
            if size <= 0 or size > download_limits.max_source_bytes:
                raise SourceFormatError(
                    f"source_artifact size {size} is outside 1..{download_limits.max_source_bytes} bytes"
                )
            staged_source = stage / "source.raw"
            shutil.copyfile(local_source, staged_source)
            _validate_artifact(staged_source, "source")
            artifacts = {
                "source": _artifact_metadata(
                    staged_source,
                    relative_to=stage,
                    url=source_origin or "local-source-artifact",
                    content_type=None,
                    reused=False,
                )
            }
        source_root = stage / "source"
        extraction = extract_source(
            stage / "source.raw",
            source_root,
            limits=extraction_limits,
        )
        main_tex = find_main_tex(source_root, limits=extraction_limits)
        combined = combine_tex_source(main_tex, source_root, limits=extraction_limits)
        combined_path = stage / "combined_source.tex"
        combined_path.write_text(combined.text, encoding="utf-8")

        def paper_relative_source(path: str) -> str:
            return (Path("source") / path).as_posix()

        extraction["root"] = "source"
        extraction["files"] = [
            {**entry, "path": paper_relative_source(str(entry["path"]))}
            for entry in extraction["files"]  # type: ignore[union-attr]
        ]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "arxiv_id": version.canonical_id,
            "base_id": version.base_id,
            "version": version.version,
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": artifacts,
            "extraction": extraction,
            "tex": {
                # ``find_main_tex`` returns a resolved path. Resolve the source
                # root as well so aliases such as macOS' /var -> /private/var
                # do not make two paths to the same tree look unrelated.
                "main_file": paper_relative_source(
                    main_tex.relative_to(source_root.resolve()).as_posix()
                ),
                "combined_file": "combined_source.tex",
                "combined_sha256": _sha256_file(combined_path),
                "combined_bytes": combined_path.stat().st_size,
                "included_files": [paper_relative_source(path) for path in combined.included_files],
                "unresolved_includes": list(combined.unresolved_includes),
            },
            "policy": {
                "storage": "local_curation_only",
                "redistribution": "check_each_paper license before distributing source or excerpts",
                "minimum_request_interval_seconds": download_limits.min_request_interval_seconds,
                "pdf_downloaded": include_pdf,
            },
        }
        _atomic_json(stage / "source_manifest.json", manifest)

        # The manifest is committed last and is the completeness marker.  A
        # failure before that point cannot make a partial preparation appear
        # valid to a subsequent call.
        manifest_path.unlink(missing_ok=True)
        for name in all_managed_names:
            target = directory / name
            staged = stage / name
            if target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if staged.exists():
                os.replace(staged, target)
        os.replace(stage / "source_manifest.json", manifest_path)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "ArxivSourceError",
    "ArxivVersion",
    "CombinedSource",
    "DEFAULT_USER_AGENT",
    "DownloadError",
    "DownloadLimits",
    "ExtractionLimits",
    "SourceFormatError",
    "UnsafeArchiveError",
    "combine_tex_source",
    "download_arxiv_artifacts",
    "extract_source",
    "find_main_tex",
    "parse_arxiv_version",
    "prepare_arxiv_source",
]
