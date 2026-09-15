"""The output viewer must tolerate new runs that have not been judged yet."""

import copy
import importlib.util
import sys
import re
from html import unescape
from urllib.parse import parse_qs, urlsplit
from pathlib import Path

import pytest
import yaml

import matharena.configs
from matharena.json_zst import dump_json_zst, load_json_zst


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    comp_dir = tmp_path / "competitions/test"
    comp_dir.mkdir(parents=True)
    (comp_dir / "partial.yaml").write_text(
        yaml.safe_dump({"dataset_path": str(tmp_path / "data")})
    )
    monkeypatch.setattr(
        matharena.configs, "extract_existing_configs", lambda *args, **kwargs: ({}, {})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--comp",
            "test/partial",
            "--output-folder",
            str(tmp_path / "outputs"),
            "--competition-config-folder",
            str(comp_dir.parent),
            "--disable-overwrite",
        ],
    )
    source = Path(__file__).resolve().parents[1] / "app/app.py"
    spec = importlib.util.spec_from_file_location(
        "matharena_test_output_viewer", source
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    return module


def judgment(points=0):
    return {
        "judge_id": "judges/answer_judge",
        "points": points,
        "max_points": 1,
        "details": [
            {"points": points, "max_points": 1, "desc": "First run was judged."}
        ],
        "error": None,
    }


@pytest.mark.parametrize("gold_answer", ["0", None])
@pytest.mark.parametrize("estimated", [False, True])
def test_problem_page_loads_with_short_judgment_row(viewer, gold_answer, estimated):
    # Exact layout observed in Astra August P5 after a second run was added.
    record = {
        "problem": "Compute the answer.",
        "gold_answer": gold_answer,
        "messages": [[{"role": "assistant", "type": "response", "content": "0"}]] * 2,
        "answers": ["0", "0"],
        "correct": [0, "TODO Grading"],
        "warnings": [0, 0],
        "detailed_costs": [
            {"cost": 0, "time": None, "input_tokens": 0, "output_tokens": 0}
            for _ in range(2)
        ],
        "judgment": [[judgment()], None],
    }
    if estimated:
        record["detailed_costs"][0]["token_usage_recovery"] = {
            "estimated": True,
            "warning": "Reasoning recovered from API timing; one failed request excluded.",
        }
    before = copy.deepcopy(record["judgment"])
    viewer.results = {"Astra": {5: record}}

    response = viewer.app.test_client().get("/view/Astra/5")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="run-tab-0"' in page
    assert 'id="run-tab-1"' in page
    assert viewer.get_run_judgments(record, 0)[0]["points"] == 0
    assert viewer.get_run_judgments(record, 1) == []
    assert record["judgment"] == before
    assert ("includes estimated reasoning" in page) is estimated
    assert ("Reasoning recovered from API timing" in page) is estimated
    if gold_answer is None:
        assert "First run was judged." in page


@pytest.mark.parametrize("raw", [None, [], [None], [[]], [[], None], "invalid", {}])
def test_absent_judgments_are_safe(viewer, raw):
    assert viewer.get_run_judgments({"judgment": raw}, 0) == []
    assert viewer.get_run_judgments({"judgment": raw}, 1) == []
    assert viewer.get_run_judgments({}, 0) == []


def test_legacy_judgments_and_out_of_range_runs(viewer):
    record = {"judgment": [judgment(1), None]}
    assert viewer.get_run_judgments(record, 0)[0]["judge_idx"] == 1
    assert viewer.get_run_judgments(record, 0)[0]["points"] == 1
    assert viewer.get_run_judgments(record, 1) == []
    assert viewer.get_run_judgments(record, 2) == []
    assert viewer.get_run_judgments(record, -1) == []


def test_uneven_judgment_rows_preserve_judge_and_run_indices(viewer):
    record = {"judgment": [[judgment(0)], None, [None, judgment(1)], []]}
    before = copy.deepcopy(record)
    first = viewer.get_run_judgments(record, 0)
    second = viewer.get_run_judgments(record, 1)
    assert [(x["judge_idx"], x["points"]) for x in first] == [(1, 0)]
    assert [(x["judge_idx"], x["points"]) for x in second] == [(3, 1)]
    assert record == before


@pytest.mark.parametrize("estimated", [False, True])
def test_model_totals_label_estimated_costs(viewer, monkeypatch, estimated):
    # Mixed measured/estimated runs must not present the model total as measured.
    monkeypatch.setattr(viewer, "get_problem_stats", lambda *args: {
        "nb_instances": 2, "corrects": [1, 0], "accuracy": 0.5,
        "warnings": [0, 0], "llm_annotations": [None, None],
    })
    records = {"Kimi": {1: {"cost": {"cost": 2}, "detailed_costs": [
        {"cost": 1}, {"cost": 1, "token_usage_recovery": {"estimated": estimated}},
    ]}}}
    stats = viewer.get_model_stats(records, "Kimi")
    assert stats["total_cost"] == 2
    assert viewer.model_stats_to_html(stats)["cost_estimated"] is estimated


@pytest.mark.parametrize("edit_judgment", [False, True])
@pytest.mark.parametrize("model_in_other_comp", [False, True])
def test_override_keeps_competition_after_reload(viewer, monkeypatch, edit_judgment, model_in_other_comp):
    model = "DeepSeek-V4.1-Flash (Max)"
    config = "deepseek/deepseek_v41_flash"
    record = {
        "problem": "May problem", "gold_answer": None if edit_judgment else "1",
        "messages": [[{"role": "assistant", "type": "response", "content": "1"}]],
        "answers": ["1"], "correct": [False], "llm_annotation": [True],
        "manual_overwrite": [False], "pass_at_1": 0,
        "cost": {"cost": 0},
        "detailed_costs": [{"cost": 0, "input_tokens": 0, "output_tokens": 0}],
        "judgment": [[judgment(0)]],
    }
    paths = {}
    for comp in ("test/may", "test/partial"):
        path = Path(viewer.args.output_folder) / comp / config / "1.json.zst"
        path.parent.mkdir(parents=True)
        dump_json_zst(record, path)
        paths[comp] = path
        comp_path = Path(viewer.args.competition_config_folder) / f"{comp}.yaml"
        comp_path.write_text(yaml.safe_dump({"dataset_path": "unused"}))
    viewer.all_comps = ["test/may", "test/partial"]
    viewer.args.disable_overwrite = False
    monkeypatch.setattr(viewer, "extract_existing_configs", lambda comp, *a, **kw:
        ({config: {}}, {config: model}) if comp == "test/may" or model_in_other_comp else ({}, {}))
    client = viewer.app.test_client()
    page = client.get(f"/view/{model}/1?comp=test/may")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    pattern = r'data-endpoint="([^"]*override_judgment[^"]*)"' if edit_judgment else r'action="([^"]*override/[^"]*)"'
    endpoint = unescape(re.search(pattern, html).group(1))
    assert parse_qs(urlsplit(endpoint).query)["comp"] == ["test/may"]
    other_before = paths["test/partial"].read_bytes()

    # A server reload or another tab resets the shared cache to another bench.
    viewer.select_competition("test/partial")
    form = {"points": "1", "desc": "Manually verified."} if edit_judgment else {"manual_correct": "correct"}
    response = client.post(endpoint, data=form, follow_redirects=True)
    assert response.status_code == 200
    assert response.request.args["comp"] == "test/may"
    saved = load_json_zst(paths["test/may"])
    assert saved["correct"] == [1]
    assert saved["manual_overwrite"] == [True]
    assert saved["pass_at_1"] == 1
    assert saved["messages"] == record["messages"]
    assert paths["test/partial"].read_bytes() == other_before
    if edit_judgment:
        assert saved["judgment"][0][0]["original_judgment"]["points"] == 0
    else:
        assert saved["llm_annotation"] == [None]

    # Conversation AJAX also selects its own competition after another tab changes it.
    viewer.select_competition("test/partial")
    conversation = client.get(f"/modelinteraction/{model}>>1>>0?comp=test/may")
    assert conversation.status_code == 200
    assert "Assistant" in conversation.get_data(as_text=True)


def test_stale_model_link_returns_not_found_instead_of_key_error(viewer):
    client = viewer.app.test_client()
    assert client.get("/view/missing/1").status_code == 404
    # Error teardown must release the competition lock too.
    assert client.get("/").status_code == 200


def test_refresh_redirect_preserves_competition(viewer):
    response = viewer.app.test_client().get("/refresh/test---partial/Model>>>1")
    assert response.status_code == 302
    assert parse_qs(urlsplit(response.location).query)["comp"] == ["test/partial"]
    assert urlsplit(response.location).path == "/view/Model/1"
