import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from matharena.json_zst import dump_json_zst, load_json_zst


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/curation/nuke_problems.py"
_SPEC = importlib.util.spec_from_file_location("nuke_problems", _SCRIPT)
_NUKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NUKE)


@pytest.mark.parametrize("removed", [{13}, {3, 13}, {3}])
def test_nuke_outputs_preserves_missing_run_gaps(tmp_path, removed):
    outputs = tmp_path / "outputs"
    original_ids = {
        "complete": list(range(1, 42)),
        "partial": [1, 4, 6, 13, 14, 19, 41],
    }
    for model, ids in original_ids.items():
        directory = outputs / "provider" / model
        directory.mkdir(parents=True)
        for idx in ids:
            dump_json_zst({"idx": idx, "problem": f"original-{idx}"}, directory / f"{idx}.json.zst")

    _NUKE.update_json_outputs(str(outputs), removed)

    for model, ids in original_ids.items():
        directory = outputs / "provider" / model
        expected = {
            idx - sum(deleted < idx for deleted in removed): idx
            for idx in ids
            if idx not in removed
        }
        assert {int(path.name.split(".")[0]) for path in directory.glob("*.json.zst")} == set(expected)
        for new_idx, old_idx in expected.items():
            assert load_json_zst(directory / f"{new_idx}.json.zst") == {
                "idx": new_idx,
                "problem": f"original-{old_idx}",
            }
        assert not list(directory.glob("temp_*"))


def test_load_problem_ids_counts_unique_numeric_ids(tmp_path):
    for name in ("1.lean", "1.tex", "1.png", "3.lean", "README.md", "notes.tex"):
        (tmp_path / name).write_text("fixture")
    assert _NUKE.load_problem_ids(str(tmp_path)) == {1, 3}


@pytest.mark.parametrize("extensions", [(".lean",), (".png",), (".tex", ".png", ".lean")])
def test_nuke_standalone_problem_formats_preserves_gaps(tmp_path, extensions):
    for old_id in (1, 2, 4, 5):
        for extension in extensions:
            (tmp_path / f"{old_id}{extension}").write_text(f"original-{old_id}{extension}")

    _NUKE.update_problems_directory(str(tmp_path), {2})

    expected = {f"{idx}{ext}" for idx in (1, 3, 4) for ext in extensions}
    assert {path.name for path in tmp_path.iterdir()} == expected
    for new_id, old_id in ((1, 1), (3, 4), (4, 5)):
        for extension in extensions:
            assert (tmp_path / f"{new_id}{extension}").read_text() == f"original-{old_id}{extension}"


def test_nuke_lean_only_competition_cli_without_answer_table(tmp_path):
    script = tmp_path / "scripts/curation/nuke_problems.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(_SCRIPT, script)
    dataset = tmp_path / "data/arxivlean/june"
    problems = dataset / "problems"
    originals = dataset / "original"
    problems.mkdir(parents=True)
    originals.mkdir()
    (dataset / "source.csv").write_text("id,source\n1,paper-a\n2,paper-b\n3,paper-c\n")
    (dataset / "source_metadata.csv").write_text("id,title,authors\n1,A,AA\n2,B,BB\n3,C,CC\n")
    outputs = tmp_path / "outputs/arxivlean/june/openai/example"
    outputs.mkdir(parents=True)
    for idx in (1, 2, 3):
        (problems / f"{idx}.lean").write_text(f"formal-{idx}")
        (originals / f"{idx}.tex").write_text(f"statement-{idx}")
        dump_json_zst({"idx": idx, "problem": f"statement-{idx}"}, outputs / f"{idx}.json.zst")

    command = [sys.executable, str(script), "arxivlean/june", "2"]
    dry_run = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
    assert "Would remove problem IDs: [2]" in dry_run.stdout
    assert (problems / "2.lean").is_file()
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Error" not in result.stdout
    assert "2 problems remaining" in result.stdout
    assert not (dataset / "answers.csv").exists()
    assert {path.name for path in problems.iterdir()} == {"1.lean", "2.lean"}
    assert {path.name for path in originals.iterdir()} == {"1.tex", "2.tex"}
    assert (problems / "2.lean").read_text() == "formal-3"
    assert (originals / "2.tex").read_text() == "statement-3"
    for name in ("source.csv", "source_metadata.csv"):
        with (dataset / name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["id"] for row in rows] == ["1", "2"]
        assert "paper-b" not in str(rows) and "BB" not in str(rows)
    assert {path.name for path in outputs.iterdir()} == {"1.json.zst", "2.json.zst"}
    assert load_json_zst(outputs / "2.json.zst") == {"idx": 2, "problem": "statement-3"}
