#!/usr/bin/env python3
"""Prepare exact-version TeX sources for the source-first ArXivMath pipeline."""

from __future__ import annotations

import argparse
import os
import re
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from matharena.arxiv_source import DownloadLimits, parse_arxiv_version, prepare_arxiv_source
from matharena.arxivbench_utils import list_paper_ids, load_metadata
from matharena.arxivmath_source import (
    add_source_mode_argument,
    configure_source_mode,
    ABSTRACT_SCREEN_FILENAME,
    SOURCE_REFERENCE_FILENAME,
    SourceStateError,
    atomic_write_json,
    build_source_reference,
    load_json,
    metadata_arxiv_base_id,
    metadata_arxiv_version,
    resolve_source_cache,
    abstract_screen_selection,
    source_cache_key,
    utc_now,
)


ARXIV_MEMBER_RE = re.compile(r"(?<!\d)(?P<id>\d{4}\.\d{4,5})(?:v(?P<version>[1-9]\d*))?(?:\.[^/]*)?$")
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
MAX_VERSION_FEED_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class BundleMember:
    bundle_path: Path
    member_name: str
    size: int
    offset_data: int
    direct_seek: bool
    bundle_size: int
    bundle_mtime_ns: int


def _is_uncompressed_tar(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(6)
    return not (prefix.startswith(b"\x1f\x8b") or prefix.startswith(b"BZh") or prefix.startswith(b"\xfd7zXZ\x00"))


def build_bundle_index(bundle_paths: list[Path]) -> dict[str, BundleMember]:
    """Index only bundle members whose filenames carry an explicit ``vN``."""

    index: dict[str, BundleMember] = {}
    skipped_unversioned = 0
    for bundle_path in bundle_paths:
        if not bundle_path.is_file():
            raise FileNotFoundError(f"source bundle not found: {bundle_path}")
        bundle_stat = bundle_path.stat()
        direct_seek = _is_uncompressed_tar(bundle_path)
        with tarfile.open(bundle_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name or "\x00" in member.name:
                    continue
                match = ARXIV_MEMBER_RE.search(member.name)
                if not match:
                    continue
                raw_version = match.group("version")
                if raw_version is None:
                    skipped_unversioned += 1
                    continue
                base_id = match.group("id")
                canonical_id = f"{base_id}v{int(raw_version)}"
                if canonical_id in index:
                    raise ValueError(f"duplicate source bundle member for {canonical_id}")
                index[canonical_id] = BundleMember(
                    bundle_path=bundle_path,
                    member_name=member.name,
                    size=member.size,
                    offset_data=member.offset_data,
                    direct_seek=direct_seek,
                    bundle_size=bundle_stat.st_size,
                    bundle_mtime_ns=bundle_stat.st_mtime_ns,
                )
    if skipped_unversioned:
        print(
            f"Ignored {skipped_unversioned} unversioned bundle members; "
            "exact-version production uses direct versioned downloads for them."
        )
    return index


def copy_bundle_member(
    record: BundleMember,
    destination: Path,
    *,
    max_bytes: int,
) -> str:
    bundle_path = record.bundle_path
    member_name = record.member_name
    declared_size = record.size
    if declared_size <= 0 or declared_size > max_bytes:
        raise ValueError(f"bundle member {member_name!r} has invalid size {declared_size}")
    bundle_stat = bundle_path.stat()
    if bundle_stat.st_size != record.bundle_size or bundle_stat.st_mtime_ns != record.bundle_mtime_ns:
        raise ValueError(f"source bundle changed after indexing: {bundle_path}")
    if record.direct_seek:
        source = bundle_path.open("rb")
        source.seek(record.offset_data)
        archive = None
    else:
        archive = tarfile.open(bundle_path, mode="r:*")
        member = archive.getmember(member_name)
        if not member.isfile() or member.size != declared_size:
            archive.close()
            raise ValueError(f"bundle member changed while reading: {member_name}")
        source = archive.extractfile(member)
        if source is None:
            archive.close()
            raise ValueError(f"cannot read source bundle member: {member_name}")
    written = 0
    try:
        with destination.open("xb") as output:
            while written < declared_size:
                chunk = source.read(min(64 * 1024, declared_size - written))
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"bundle member exceeds {max_bytes} bytes")
                output.write(chunk)
    finally:
        source.close()
        if archive is not None:
            archive.close()
    if written != declared_size:
        raise ValueError(f"bundle member ended at {written} bytes, expected {declared_size}")
    return f"s3-bundle:{bundle_path.name}#{member_name}"


def ingestion_record(*, status: str, arxiv_id: str, cache_key: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "arxiv_id": arxiv_id,
        "cache_key": cache_key,
        "updated_at": utc_now(),
        **extra,
    }


def ingestion_exit_code(
    *,
    prepared: int,
    reused: int,
    failed: int,
    allow_source_unavailable: bool,
) -> int:
    if prepared + reused == 0:
        return 1
    if failed and not allow_source_unavailable:
        return 1
    return 0


def parse_version_feed(payload: bytes, expected_base_ids: set[str]) -> dict[str, str]:
    """Extract exact latest revisions from one arXiv Atom API response."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise SourceStateError(f"arXiv version response is malformed XML: {exc}") from exc
    versions: dict[str, str] = {}
    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NAMESPACE) or "").strip()
        parsed_url = urllib.parse.urlparse(raw_id)
        if (
            parsed_url.scheme not in {"http", "https"}
            or parsed_url.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
            or not parsed_url.path.startswith("/abs/")
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise SourceStateError(f"arXiv version response has an invalid entry id: {raw_id!r}")
        candidate = urllib.parse.unquote(parsed_url.path[len("/abs/") :])
        try:
            version = parse_arxiv_version(candidate)
        except ValueError as exc:
            raise SourceStateError(f"arXiv version response did not pin an exact revision: {raw_id!r}") from exc
        if version.base_id not in expected_base_ids:
            raise SourceStateError(f"arXiv version response returned unexpected id {version.base_id!r}")
        if version.base_id in versions:
            raise SourceStateError(f"arXiv version response duplicated id {version.base_id!r}")
        versions[version.base_id] = version.canonical_id

    missing = sorted(expected_base_ids - set(versions))
    if missing:
        examples = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else ", ..."
        raise SourceStateError(f"arXiv version response omitted {len(missing)} requested ids ({examples}{suffix})")
    return versions


def _fetch_version_feed(url: str, *, user_agent: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/atom+xml", "User-Agent": user_agent},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_VERSION_FEED_BYTES + 1)
    if len(payload) > MAX_VERSION_FEED_BYTES:
        raise SourceStateError(f"arXiv version response exceeds {MAX_VERSION_FEED_BYTES} bytes")
    return payload


def resolve_latest_versions(
    base_ids: list[str],
    *,
    api_url: str,
    user_agent: str,
    batch_size: int,
    timeout: float,
    retries: int,
    request_interval: float,
    fetch_xml: Callable[[str], bytes] | None = None,
) -> dict[str, str]:
    """Resolve and return one explicit latest ``vN`` for every base identifier."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    if request_interval < 0:
        raise ValueError("request_interval cannot be negative")
    requested = sorted(set(base_ids))
    fetch = fetch_xml or (lambda url: _fetch_version_feed(url, user_agent=user_agent, timeout=timeout))
    resolved: dict[str, str] = {}
    last_request_at: float | None = None
    for offset in range(0, len(requested), batch_size):
        batch = requested[offset : offset + batch_size]
        query = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": len(batch)})
        separator = "&" if "?" in api_url else "?"
        url = f"{api_url}{separator}{query}"
        for attempt in range(retries + 1):
            if last_request_at is not None:
                wait = request_interval - (time.monotonic() - last_request_at)
                if wait > 0:
                    time.sleep(wait)
            last_request_at = time.monotonic()
            try:
                payload = fetch(url)
                resolved.update(parse_version_feed(payload, set(batch)))
                break
            except (OSError, SourceStateError) as exc:
                if attempt >= retries:
                    raise SourceStateError(
                        f"failed to resolve exact arXiv revisions after {attempt + 1} attempts: {exc}"
                    ) from exc
    return resolved


def pin_missing_metadata_versions(
    paper_root: Path,
    paper_ids: list[str],
    *,
    api_url: str,
    user_agent: str,
    batch_size: int,
    timeout: float,
    retries: int,
    request_interval: float,
) -> int:
    metadata_by_paper: dict[str, dict[str, Any]] = {}
    base_id_by_paper: dict[str, str] = {}
    missing_base_ids: list[str] = []
    for paper_id in paper_ids:
        metadata = load_metadata(str(paper_root), paper_id)
        base_id = metadata_arxiv_base_id(metadata)
        metadata_by_paper[paper_id] = metadata
        base_id_by_paper[paper_id] = base_id
        if metadata.get("versioned_id"):
            metadata_arxiv_version(metadata)
        else:
            missing_base_ids.append(base_id)
    if not missing_base_ids:
        return 0

    resolved = resolve_latest_versions(
        missing_base_ids,
        api_url=api_url,
        user_agent=user_agent,
        batch_size=batch_size,
        timeout=timeout,
        retries=retries,
        request_interval=request_interval,
    )
    resolved_at = utc_now()
    for paper_id, metadata in metadata_by_paper.items():
        if metadata.get("versioned_id"):
            continue
        base_id = base_id_by_paper[paper_id]
        metadata["versioned_id"] = resolved[base_id]
        metadata["version_resolution"] = {
            "source": "arxiv_atom_api",
            "endpoint": api_url,
            "resolved_at": resolved_at,
        }
        metadata_arxiv_version(metadata)
        atomic_write_json(paper_root / paper_id / "metadata.json", metadata)
    return len(missing_base_ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and safely prepare version-pinned arXiv TeX source without PDFs or OCR."
    )
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--abstract-screen-filename", default=ABSTRACT_SCREEN_FILENAME)
    parser.add_argument(
        "--source-cache",
        default=None,
        help="Defaults to ARXIV_SOURCE_CACHE or arxivmath/source_cache.",
    )
    parser.add_argument("--base-url", default="https://arxiv.org")
    parser.add_argument("--version-api-url", default=ARXIV_API_URL)
    parser.add_argument("--version-batch-size", type=int, default=100)
    parser.add_argument("--version-timeout", type=float, default=60.0)
    parser.add_argument("--version-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Maximum successfully prepared papers.")
    parser.add_argument("--max-papers", type=int, default=None, help="Maximum metadata folders inspected.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unresolved-includes",
        action="store_true",
        help="Explicitly accept incomplete TeX expansion and record it in provenance.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Previously downloaded official arXiv S3 source tar; may be repeated.",
    )
    parser.add_argument(
        "--no-direct-fallback",
        action="store_true",
        help="Fail papers absent from supplied bundles instead of using arxiv.org/src.",
    )
    parser.add_argument(
        "--allow-source-unavailable",
        action="store_true",
        help="Record unavailable sources and continue successfully instead of stopping the month.",
    )
    parser.add_argument("--max-source-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--request-interval", type=float, default=3.0)
    parser.add_argument(
        "--user-agent",
        default="MathArena-ArXivMathSource/1.0 (source-first benchmark research; https://github.com/eth-sri/matharena)",
    )
    add_source_mode_argument(parser)
    args = parser.parse_args()
    configure_source_mode(args)
    if args.version_batch_size < 1:
        parser.error("--version-batch-size must be positive")
    if args.version_timeout <= 0:
        parser.error("--version-timeout must be positive")
    if args.version_retries < 0:
        parser.error("--version-retries cannot be negative")
    if args.request_interval < 3:
        parser.error("--request-interval must be at least 3 seconds")

    paper_root = Path(args.paper_root)
    source_cache = resolve_source_cache(args.source_cache)
    source_cache.mkdir(parents=True, exist_ok=True)
    bundle_paths = [Path(value).expanduser().resolve() for value in args.bundle]
    bundle_index = build_bundle_index(bundle_paths) if bundle_paths else {}
    limits = DownloadLimits(
        max_source_bytes=args.max_source_bytes,
        min_request_interval_seconds=args.request_interval,
    )

    paper_ids = list_paper_ids(str(paper_root))
    if args.max_papers is not None:
        paper_ids = paper_ids[: args.max_papers]
    screen_summary = abstract_screen_selection(
        paper_root,
        paper_ids=paper_ids,
        screen_filename=args.abstract_screen_filename,
    )
    paper_ids = screen_summary["accepted"]
    print(
        f"Abstract screen: {len(paper_ids)} accepted, {len(screen_summary['rejected'])} rejected, "
        f"{len(screen_summary['incomplete'])} not ready (skipped)."
    )
    if not paper_ids:
        print("No accepted papers ready for source download.")
        return 0
    try:
        pinned = pin_missing_metadata_versions(
            paper_root,
            paper_ids,
            api_url=args.version_api_url,
            user_agent=args.user_agent,
            batch_size=args.version_batch_size,
            timeout=args.version_timeout,
            retries=args.version_retries,
            request_interval=args.request_interval,
        )
    except (OSError, SourceStateError, ValueError) as exc:
        print(f"Refusing source download: exact arXiv revision resolution failed: {exc}")
        return 1
    print(f"Exact arXiv revisions: pinned {pinned}, reused {len(paper_ids) - pinned}.")
    prepared = reused = failed = inspected = 0
    for paper_id in paper_ids:
        inspected += 1
        paper_dir = paper_root / paper_id
        status_path = paper_dir / "source_ingestion.json"
        try:
            metadata = load_metadata(str(paper_root), paper_id)
            version = metadata_arxiv_version(metadata)
            cache_key = source_cache_key(version)
            cache_dir = source_cache / cache_key
            manifest_path = cache_dir / "source_manifest.json"

            if manifest_path.is_file() and not args.overwrite:
                manifest = prepare_arxiv_source(
                    version,
                    cache_dir,
                    include_pdf=False,
                    download_limits=limits,
                    user_agent=args.user_agent,
                )
                was_reused = True
            else:
                bundle_record = bundle_index.get(version.canonical_id)
                if bundle_record is not None:
                    fd, tmp_name = tempfile.mkstemp(prefix=".arxiv-bundle-member-", dir=source_cache)
                    os.close(fd)
                    tmp_path = Path(tmp_name)
                    tmp_path.unlink()
                    try:
                        origin = copy_bundle_member(bundle_record, tmp_path, max_bytes=args.max_source_bytes)
                        manifest = prepare_arxiv_source(
                            version,
                            cache_dir,
                            include_pdf=False,
                            source_artifact=tmp_path,
                            source_origin=origin,
                            download_limits=limits,
                            user_agent=args.user_agent,
                            overwrite=args.overwrite,
                        )
                    finally:
                        tmp_path.unlink(missing_ok=True)
                elif args.no_direct_fallback:
                    raise FileNotFoundError(f"{version.base_id} is absent from supplied source bundles")
                else:
                    manifest = prepare_arxiv_source(
                        version,
                        cache_dir,
                        base_url=args.base_url,
                        include_pdf=False,
                        download_limits=limits,
                        user_agent=args.user_agent,
                        overwrite=args.overwrite,
                    )
                was_reused = False

            reference = build_source_reference(
                cache_dir,
                manifest,
                allow_unresolved_includes=args.allow_unresolved_includes,
            )
            atomic_write_json(paper_dir / SOURCE_REFERENCE_FILENAME, reference)
            atomic_write_json(
                status_path,
                ingestion_record(
                    status="complete",
                    arxiv_id=version.canonical_id,
                    cache_key=cache_key,
                    source_origin=(manifest.get("artifacts") or {}).get("source", {}).get("url"),
                    reused=was_reused,
                    unresolved_include_count=reference["unresolved_include_count"],
                    unresolved_includes_allowed=reference["unresolved_includes_allowed"],
                ),
            )
            if was_reused:
                reused += 1
            else:
                prepared += 1
            if args.limit is not None and prepared + reused >= args.limit:
                break
        except Exception as exc:
            failed += 1
            # Never let a reference from an earlier metadata version survive a
            # failed ingestion attempt and enter downstream source-first stages.
            (paper_dir / SOURCE_REFERENCE_FILENAME).unlink(missing_ok=True)
            fallback_id = str((load_json(paper_dir / "metadata.json", {}) or {}).get("id") or paper_id)
            atomic_write_json(
                status_path,
                ingestion_record(
                    status="failed",
                    arxiv_id=fallback_id,
                    cache_key="",
                    error_type=type(exc).__name__,
                    message=str(exc)[:2000],
                ),
            )
            print(f"Source preparation failed for {paper_id}: {type(exc).__name__}: {exc}")

    print(
        f"Inspected {inspected} papers: prepared {prepared}, reused {reused}, failed {failed}. "
        f"Source cache: {source_cache}"
    )
    return ingestion_exit_code(
        prepared=prepared,
        reused=reused,
        failed=failed,
        allow_source_unavailable=args.allow_source_unavailable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
