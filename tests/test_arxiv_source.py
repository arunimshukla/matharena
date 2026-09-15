import bz2
import gzip
import hashlib
import io
import json
import lzma
import tarfile
import uuid
from pathlib import Path

import pytest

from matharena.arxiv_source import (
    ArxivVersion,
    DownloadError,
    DownloadLimits,
    ExtractionLimits,
    SourceFormatError,
    UnsafeArchiveError,
    combine_tex_source,
    download_arxiv_artifacts,
    extract_source,
    find_main_tex,
    parse_arxiv_version,
    prepare_arxiv_source,
)


class FakeResponse:
    def __init__(self, body: bytes, *, headers=None, error=None):
        self.body = body
        self.headers = headers or {}
        self.error = error
        self.closed = False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def make_tar(entries, *, mode="w", archive_format=None):
    """Build a tar from (name, bytes, optional type/linkname) tuples."""

    buffer = io.BytesIO()
    options = {} if archive_format is None else {"format": archive_format}
    with tarfile.open(fileobj=buffer, mode=mode, **options) as archive:
        for entry in entries:
            name, body, *metadata = entry
            info = tarfile.TarInfo(name)
            if metadata:
                info.type = metadata[0]
                if len(metadata) > 1:
                    info.linkname = metadata[1]
            if info.type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
            else:
                archive.addfile(info)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("raw", "base_id", "version"),
    [
        ("2608.12345v1", "2608.12345", 1),
        ("arXiv:2608.1234v12", "2608.1234", 12),
        ("https://arxiv.org/abs/2608.12345v3", "2608.12345", 3),
        ("https://arxiv.org/pdf/hep-th/9901001v2.pdf", "hep-th/9901001", 2),
        ("physics.optics/0601001v4", "physics.optics/0601001", 4),
    ],
)
def test_parse_arxiv_version_requires_and_normalizes_explicit_version(raw, base_id, version):
    parsed = parse_arxiv_version(raw)

    assert parsed == ArxivVersion(base_id, version)
    assert str(parsed) == f"{base_id}v{version}"
    assert "/" not in parsed.storage_key


@pytest.mark.parametrize(
    "raw",
    [
        "2608.12345",
        "2608.12345v0",
        "2613.12345v1",
        "../../2608.12345v1",
        "https://example.com/abs/2608.12345v1",
        "https://arxiv.org/search/?query=2608.12345v1",
        "hep-th/../../etc/passwdv1",
    ],
)
def test_parse_arxiv_version_rejects_unpinned_or_invalid_shapes(raw):
    with pytest.raises(ValueError):
        parse_arxiv_version(raw)


def test_arxiv_version_object_cannot_bypass_identifier_validation():
    with pytest.raises(ValueError):
        ArxivVersion("../../etc/passwd", 1)
    with pytest.raises(ValueError):
        ArxivVersion("2608.12345", 0)


def zero_interval_limits(**overrides):
    values = {"min_request_interval_seconds": 0}
    values.update(overrides)
    return DownloadLimits(**values)


def test_download_artifacts_streams_exact_version_with_hashes_and_user_agent(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    pdf = b"%PDF-1.7\nexample"
    session = FakeSession(
        [
            FakeResponse(source, headers={"Content-Type": "application/x-eprint"}),
            FakeResponse(pdf, headers={"Content-Type": "application/pdf"}),
        ]
    )

    metadata = download_arxiv_artifacts(
        "2608.12345v2",
        tmp_path,
        session=session,
        base_url="https://mirror.example",
        limits=zero_interval_limits(chunk_size=7),
        user_agent="MathArena-Test/1 contact@example.test",
    )

    assert [call[0] for call in session.calls] == [
        "https://mirror.example/src/2608.12345v2",
        "https://mirror.example/pdf/2608.12345v2.pdf",
    ]
    assert all(call[1]["stream"] is True for call in session.calls)
    assert all(call[1]["headers"]["User-Agent"].startswith("MathArena-Test") for call in session.calls)
    assert metadata["source"]["sha256"] == hashlib.sha256(source).hexdigest()
    assert metadata["pdf"]["sha256"] == hashlib.sha256(pdf).hexdigest()
    assert metadata["source"]["bytes"] == len(source)
    assert (tmp_path / "source.raw").read_bytes() == source
    assert (tmp_path / "paper.pdf").read_bytes() == pdf


def test_download_artifacts_can_skip_pdf_for_source_only_pipeline(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    session = FakeSession([FakeResponse(source)])

    metadata = download_arxiv_artifacts(
        "2608.12345v2",
        tmp_path,
        session=session,
        base_url="https://mirror.example",
        limits=zero_interval_limits(),
        include_pdf=False,
    )

    assert [call[0] for call in session.calls] == ["https://mirror.example/src/2608.12345v2"]
    assert set(metadata) == {"source"}
    assert (tmp_path / "source.raw").read_bytes() == source
    assert not (tmp_path / "paper.pdf").exists()


def test_download_artifacts_rate_limits_across_consecutive_calls(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    pdf = b"%PDF-1.7\nexample"
    session = FakeSession([FakeResponse(source), FakeResponse(pdf), FakeResponse(source), FakeResponse(pdf)])
    sleeps = []
    origin = f"https://pace-{uuid.uuid4().hex}.example"
    limits = DownloadLimits(min_request_interval_seconds=3, chunk_size=1024)

    for index in range(2):
        download_arxiv_artifacts(
            f"2608.1234{index}v1",
            tmp_path / str(index),
            session=session,
            base_url=origin,
            limits=limits,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )

    # source->PDF for each paper and PDF(1)->source(2) are all paced.
    assert sleeps == [3.0, 3.0, 3.0]


def test_rate_limit_wait_is_not_charged_to_transfer_timeout(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    pdf = b"%PDF-1.7\nexample"
    session = FakeSession([FakeResponse(source), FakeResponse(pdf)])
    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds

    download_arxiv_artifacts(
        "2608.12345v1",
        tmp_path,
        session=session,
        base_url=f"https://timeout-{uuid.uuid4().hex}.example",
        limits=DownloadLimits(total_timeout_seconds=1, min_request_interval_seconds=3),
        sleep=advance,
        monotonic=lambda: clock[0],
    )

    assert clock[0] == 3.0


def test_download_rejects_content_length_before_writing_artifact(tmp_path):
    response = FakeResponse(
        b"not used",
        headers={"Content-Length": "1000", "Content-Type": "application/x-eprint"},
    )
    session = FakeSession([response])

    with pytest.raises(DownloadError, match="Content-Length"):
        download_arxiv_artifacts(
            "2608.12345v1",
            tmp_path,
            session=session,
            limits=zero_interval_limits(max_source_bytes=10),
        )

    assert not (tmp_path / "source.raw").exists()
    assert not (tmp_path / ".source.raw.part").exists()


def test_download_enforces_streamed_total_timeout(tmp_path):
    source = b"\\documentclass{article}\n"
    session = FakeSession([FakeResponse(source)])
    ticks = iter([0.0, 2.0])

    with pytest.raises(DownloadError, match="total timeout"):
        download_arxiv_artifacts(
            "2608.12345v1",
            tmp_path,
            session=session,
            limits=zero_interval_limits(total_timeout_seconds=1),
            monotonic=lambda: next(ticks),
        )

    assert not (tmp_path / "source.raw").exists()


def test_extract_tar_inventory_and_hashes(tmp_path):
    archive = tmp_path / "source.raw"
    main = b"\\documentclass{article}\n\\begin{document}\n\\input{section}\n\\end{document}\n"
    section = b"Physics.\n"
    archive.write_bytes(make_tar([("main.tex", main), ("parts/section.tex", section)]))

    result = extract_source(archive, tmp_path / "source")

    assert result["format"] == "tar"
    assert result["total_bytes"] == len(main) + len(section)
    assert [entry["path"] for entry in result["files"]] == ["main.tex", "parts/section.tex"]
    assert result["files"][0]["sha256"] == hashlib.sha256(main).hexdigest()


@pytest.mark.parametrize(
    "entry",
    [
        ("../escape.tex", b"bad"),
        ("/absolute.tex", b"bad"),
        ("link.tex", b"", tarfile.SYMTYPE, "outside.tex"),
        ("hard.tex", b"", tarfile.LNKTYPE, "main.tex"),
        ("device", b"", tarfile.CHRTYPE),
        ("pipe", b"", tarfile.FIFOTYPE),
    ],
)
def test_extract_tar_rejects_unsafe_member_types_and_paths_without_partial_tree(tmp_path, entry):
    archive = tmp_path / "source.raw"
    archive.write_bytes(make_tar([entry]))
    destination = tmp_path / "source"

    with pytest.raises(UnsafeArchiveError):
        extract_source(archive, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape.tex").exists()
    assert not list(tmp_path.glob(".source.stage-*"))


def test_extract_tar_rejects_member_and_uncompressed_size_limits(tmp_path):
    archive = tmp_path / "source.raw"
    archive.write_bytes(make_tar([("one.tex", b"1"), ("two.tex", b"2")]))

    with pytest.raises(UnsafeArchiveError, match="more than 1 members"):
        extract_source(archive, tmp_path / "members", limits=ExtractionLimits(max_members=1))
    with pytest.raises(UnsafeArchiveError, match="beyond 1 total bytes"):
        extract_source(archive, tmp_path / "bytes", limits=ExtractionLimits(max_total_bytes=1))


@pytest.mark.parametrize(
    ("payload", "expected_format"),
    [
        (b"\\documentclass{article}\n\\begin{document}x\\end{document}\n", "plain_tex"),
        (
            gzip.compress(b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"),
            "gzip_tex",
        ),
        (
            bz2.compress(b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"),
            "bzip2_tex",
        ),
    ],
)
def test_extract_single_plain_or_compressed_tex(tmp_path, payload, expected_format):
    archive = tmp_path / "source.raw"
    archive.write_bytes(payload)

    result = extract_source(archive, tmp_path / "source")

    assert result["format"] == expected_format
    assert (tmp_path / "source" / "main.tex").read_text().startswith("\\documentclass")


def test_extract_recognizes_gzipped_tar_before_single_gzip_tex(tmp_path):
    archive = tmp_path / "source.raw"
    archive.write_bytes(make_tar([("paper.tex", b"\\documentclass{article}\n")], mode="w:gz"))

    result = extract_source(archive, tmp_path / "source")

    assert result["format"] == "tar"
    assert (tmp_path / "source" / "paper.tex").exists()


@pytest.mark.parametrize(
    ("archive_format", "compression", "message"),
    [
        (tarfile.PAX_FORMAT, True, "PAX extended header"),
        (tarfile.GNU_FORMAT, False, "GNU long-name header"),
    ],
)
def test_extract_rejects_pax_and_gnu_metadata_before_tarfile_parses_it(
    tmp_path,
    archive_format,
    compression,
    message,
):
    long_name = ("a" * 140) + ".tex"
    payload = make_tar(
        [(long_name, b"\\documentclass{article}\n")],
        archive_format=archive_format,
    )
    if compression:
        payload = gzip.compress(payload)
    archive = tmp_path / "source.raw"
    archive.write_bytes(payload)

    with pytest.raises(UnsafeArchiveError, match=message):
        extract_source(archive, tmp_path / "source")

    assert not (tmp_path / "source").exists()
    assert not list(tmp_path.glob(".arxiv-source-payload-*"))


def test_extract_rejects_xz_before_dictionary_allocation(tmp_path):
    archive = tmp_path / "source.raw"
    archive.write_bytes(lzma.compress(b"\\documentclass{article}\n", format=lzma.FORMAT_XZ))

    with pytest.raises(UnsafeArchiveError, match="XZ source artifacts"):
        extract_source(archive, tmp_path / "source")


def test_extract_bounds_decompressed_stream_before_format_detection(tmp_path):
    archive = tmp_path / "source.raw"
    archive.write_bytes(gzip.compress(b"\\documentclass{article}\n" + (b"x" * 200)))

    with pytest.raises(UnsafeArchiveError, match="expands beyond 64 bytes"):
        extract_source(
            archive,
            tmp_path / "source",
            limits=ExtractionLimits(max_archive_stream_bytes=64),
        )

    assert not list(tmp_path.glob(".arxiv-source-payload-*"))


def test_extract_bounds_gzip_optional_header_metadata(tmp_path):
    archive = tmp_path / "source.raw"
    # A valid fixed header with FNAME set, followed by an unterminated name.
    archive.write_bytes(b"\x1f\x8b\x08\x08" + (b"\x00" * 4) + b"\x00\xff" + (b"a" * 100))

    with pytest.raises(UnsafeArchiveError, match="gzip metadata exceeds 32 bytes"):
        extract_source(
            archive,
            tmp_path / "source",
            limits=ExtractionLimits(max_compression_header_bytes=32),
        )


def test_find_main_tex_prefers_complete_unreferenced_document(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\title{Pilot}\n\\begin{document}\n\\input{fragment}\n\\end{document}\n"
    )
    (root / "fragment.tex").write_text("A fragment only.\n")
    (root / "notes.tex").write_text("\\begin{document}\nnotes\n")

    assert find_main_tex(root) == root / "main.tex"


def test_find_main_tex_rejects_a_lone_fragment(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "macros.tex").write_text("\\def\\answer{42}\n")

    with pytest.raises(SourceFormatError, match="could not identify"):
        find_main_tex(root)


def test_find_main_tex_accepts_case_insensitive_tex_extensions(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    paper = root / "PAPER.TEX"
    paper.write_text("\\documentclass{article}\n\\begin{document}x\\end{document}\n")

    assert find_main_tex(root) == paper


def test_combine_tex_recursively_inlines_local_files_but_not_comments(tmp_path):
    root = tmp_path / "source"
    (root / "sections" / "nested").mkdir(parents=True)
    main = root / "main.tex"
    main.write_text(
        "\\documentclass{article}\n"
        "% \\input{do-not-expand}\n"
        "\\begin{document}\n"
        "\\input{sections/one}\n"
        "\\input{\\jobname.generated}\n"
        "\\input{missing}\n"
        "\\end{document}\n"
    )
    (root / "sections" / "one.tex").write_text("one \\include{nested/two}\n")
    (root / "sections" / "nested" / "two.tex").write_text("two\n")

    result = combine_tex_source(main, root)

    assert "one " in result.text and "two\n" in result.text
    assert "% \\input{do-not-expand}" in result.text
    assert "do-not-expand" not in result.included_files
    assert result.included_files == ("sections/one.tex", "sections/nested/two.tex")
    assert [item["reason"] for item in result.unresolved_includes] == ["dynamic", "missing"]
    assert "% MATHARENA_SOURCE_BEGIN file=sections/one.tex via=main.tex:4" in result.text
    assert "% MATHARENA_SOURCE_END file=main.tex" in result.text


def test_combine_does_not_expand_protected_tex_regions_or_macro_definitions(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    main = root / "main.tex"
    main.write_text(
        "\\documentclass{article}\n"
        "\\newcommand{\\loader}{\\input{macro-body}}\n"
        "\\def\\other{\\input{def-body}}\n"
        "\\begin{verbatim}\n\\input{verbatim-body}\n\\end{verbatim}\n"
        "\\begin{lstlisting}\n\\input{listing-body}\n\\end{lstlisting}\n"
        "\\verb|\\input{inline-verb}|\n"
        "\\begin{document}\n\\input{real}\n\\end{document}\n"
    )
    for name in ("macro-body", "def-body", "verbatim-body", "listing-body", "inline-verb", "real"):
        (root / f"{name}.tex").write_text(f"expanded {name}\n")

    result = combine_tex_source(main, root)

    assert result.included_files == ("real.tex",)
    assert "expanded real" in result.text
    assert "expanded macro-body" not in result.text
    assert "\\input{macro-body}" in result.text
    assert result.unresolved_includes == ()


def test_combine_fails_closed_after_an_unbalanced_macro_definition(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    main = root / "main.tex"
    main.write_text(
        "\\documentclass{article}\n"
        "\\newcommand{\\broken}{unclosed body\n"
        "\\input{must-not-expand}\n"
    )
    (root / "must-not-expand.tex").write_text("unsafe expansion\n")

    result = combine_tex_source(main, root)

    assert result.included_files == ()
    assert result.unresolved_includes == ()
    assert "\\input{must-not-expand}" in result.text


def test_combine_rejects_reserved_marker_spoofing(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    main = root / "main.tex"
    main.write_text(
        "\\documentclass{article}\n"
        "% MATHARENA_SOURCE_BEGIN file=spoofed.tex\n"
        "\\begin{document}x\\end{document}\n"
    )

    with pytest.raises(SourceFormatError, match="reserved MathArena marker at line 2"):
        combine_tex_source(main, root)


def test_combine_enforces_include_and_unresolved_directive_limits(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    main = root / "main.tex"
    main.write_text("\\documentclass{article}\n\\input{one}\n\\input{two}\n")

    with pytest.raises(SourceFormatError, match="more than 1 expanded include directives"):
        combine_tex_source(main, root, limits=ExtractionLimits(max_include_directives=1))
    with pytest.raises(SourceFormatError, match="more than 1 unresolved include directives"):
        combine_tex_source(
            main,
            root,
            limits=ExtractionLimits(max_include_directives=2, max_unresolved_includes=1),
        )


def test_combine_tracks_many_include_lines_with_a_single_forward_scan(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    main = root / "main.tex"
    directive_count = 2_000
    main.write_text(
        "\\documentclass{article}\n"
        + "".join(f"\\input{{missing-{index}}}\n" for index in range(directive_count))
    )

    result = combine_tex_source(
        main,
        root,
        limits=ExtractionLimits(
            max_include_directives=directive_count,
            max_unresolved_includes=directive_count,
        ),
    )

    assert len(result.unresolved_includes) == directive_count
    assert result.unresolved_includes[0]["line"] == 2
    assert result.unresolved_includes[-1]["line"] == directive_count + 1


def test_combine_tex_rejects_escape_and_include_cycle(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.tex"
    outside.write_text("outside")
    main = root / "main.tex"
    main.write_text("\\documentclass{article}\n\\input{../outside}\n")

    with pytest.raises(UnsafeArchiveError, match="escapes source root"):
        combine_tex_source(main, root)

    main.write_text("\\documentclass{article}\n\\input{a}\n")
    (root / "a.tex").write_text("\\input{main}\n")
    with pytest.raises(SourceFormatError, match="cyclic"):
        combine_tex_source(main, root)


def test_prepare_arxiv_source_writes_complete_manifest_and_reuses_it(tmp_path):
    source_tar = make_tar(
        [
            (
                "paper.tex",
                b"\\documentclass{article}\n\\begin{document}\n\\input{result}\n\\end{document}\n",
            ),
            ("result.tex", b"The answer is 42.\n"),
        ],
        mode="w:gz",
    )
    pdf = b"%PDF-1.7\nphysics"
    session = FakeSession([FakeResponse(source_tar), FakeResponse(pdf)])

    manifest = prepare_arxiv_source(
        "2608.12345v1",
        tmp_path,
        session=session,
        base_url="https://mirror.example",
        download_limits=zero_interval_limits(),
    )

    on_disk = json.loads((tmp_path / "source_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["arxiv_id"] == "2608.12345v1"
    assert manifest["artifacts"]["source"]["sha256"] == hashlib.sha256(source_tar).hexdigest()
    assert manifest["tex"]["main_file"] == "source/paper.tex"
    assert manifest["tex"]["included_files"] == ["source/result.tex"]
    assert manifest["tex"]["combined_sha256"] == hashlib.sha256(
        (tmp_path / "combined_source.tex").read_bytes()
    ).hexdigest()
    assert manifest["tex"]["combined_sha256"] != manifest["artifacts"]["source"]["sha256"]
    assert all(entry["path"].startswith("source/") for entry in manifest["extraction"]["files"])
    assert not list(tmp_path.glob(".arxiv-source-stage-*"))

    empty_session = FakeSession([])
    assert prepare_arxiv_source(
        "2608.12345v1",
        tmp_path,
        session=empty_session,
        download_limits=zero_interval_limits(),
    ) == manifest
    assert empty_session.calls == []


def test_prepare_arxiv_source_without_pdf_writes_and_reuses_source_only_manifest(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    session = FakeSession([FakeResponse(source)])

    manifest = prepare_arxiv_source(
        "2608.12345v1",
        tmp_path,
        session=session,
        base_url="https://mirror.example",
        download_limits=zero_interval_limits(),
        include_pdf=False,
    )

    assert set(manifest["artifacts"]) == {"source"}
    assert manifest["policy"]["pdf_downloaded"] is False
    assert not (tmp_path / "paper.pdf").exists()
    assert prepare_arxiv_source(
        "2608.12345v1",
        tmp_path,
        session=FakeSession([]),
        download_limits=zero_interval_limits(),
        include_pdf=False,
    ) == manifest


def test_prepare_arxiv_source_imports_local_bulk_artifact_without_leaking_path(tmp_path):
    source_artifact = tmp_path / "bundle-member.gz"
    source_artifact.write_bytes(
        gzip.compress(b"\\documentclass{article}\n\\begin{document}x\\end{document}\n")
    )
    destination = tmp_path / "prepared" / "2608.12345v1"

    manifest = prepare_arxiv_source(
        "2608.12345v1",
        destination,
        include_pdf=False,
        source_artifact=source_artifact,
        source_origin="s3://arxiv/src/arXiv_src_2608_001.tar#2608/2608.12345.gz",
        download_limits=zero_interval_limits(),
    )

    assert manifest["artifacts"]["source"]["url"].startswith("s3://arxiv/")
    assert str(tmp_path) not in json.dumps(manifest)
    assert (destination / "combined_source.tex").is_file()


def test_prepare_cache_detects_changed_extracted_source(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    session = FakeSession([FakeResponse(source), FakeResponse(b"%PDF-1.7\nphysics")])
    prepare_arxiv_source(
        "2608.12345v1",
        tmp_path,
        session=session,
        download_limits=zero_interval_limits(),
    )
    (tmp_path / "source" / "main.tex").write_text("changed")

    with pytest.raises(SourceFormatError, match="artifact hashes changed"):
        prepare_arxiv_source(
            "2608.12345v1",
            tmp_path,
            session=FakeSession([]),
            download_limits=zero_interval_limits(),
        )


def test_prepare_failure_leaves_no_apparently_complete_or_partial_artifacts(tmp_path):
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    session = FakeSession([FakeResponse(source), FakeResponse(b"this is not a PDF")])

    with pytest.raises(DownloadError, match="PDF header"):
        prepare_arxiv_source(
            "2608.12345v1",
            tmp_path,
            session=session,
            download_limits=zero_interval_limits(),
        )

    for name in ("source.raw", "paper.pdf", "source", "combined_source.tex", "source_manifest.json"):
        assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob(".arxiv-source-stage-*"))


def test_prepare_handles_paper_directory_reached_through_a_symlink(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    source = b"\\documentclass{article}\n\\begin{document}x\\end{document}\n"
    pdf = b"%PDF-1.7\nphysics"
    session = FakeSession([FakeResponse(source), FakeResponse(pdf)])

    manifest = prepare_arxiv_source(
        "2608.12345v1",
        alias_root,
        session=session,
        base_url="https://mirror.example",
        download_limits=zero_interval_limits(),
    )

    assert manifest["tex"]["main_file"] == "source/main.tex"
    assert (real_root / "source_manifest.json").is_file()
