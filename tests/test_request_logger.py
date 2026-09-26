import json

from matharena.request_logger import RequestLogger


def test_single_competition_request_log_keeps_comp_name(tmp_path):
    request_logger = RequestLogger()
    request_logger.log_dir = str(tmp_path)
    request_logger.enabled = True
    request_logger.set_metadata("aime/aime_2026", "model", {0: 1})

    request_logger.log_request("single", 0, {"prompt": "test"})

    log_path = tmp_path / "aime/aime_2026" / "model" / "single_p1_idx0.json"
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["comp_name"] == "aime/aime_2026"
    assert data["problem_idx"] == 1


def test_multi_competition_request_log_preserves_source_comp_name(tmp_path):
    request_logger = RequestLogger()
    request_logger.log_dir = str(tmp_path)
    request_logger.enabled = True
    request_logger.set_metadata(
        "multi",
        "model",
        {0: 1, 1: 1},
        {
            0: "aime/aime_2026",
            1: "hmmt/hmmt_feb_2026",
        },
    )

    request_logger.log_request("aime", 0, {"prompt": "aime"})
    request_logger.log_request("hmmt", 1, {"prompt": "hmmt"})

    log_dir = tmp_path / "multi" / "model"
    aime = json.loads((log_dir / "aime_p1_idx0.json").read_text(encoding="utf-8"))
    hmmt = json.loads((log_dir / "hmmt_p1_idx1.json").read_text(encoding="utf-8"))

    assert aime["comp_name"] == "aime/aime_2026"
    assert hmmt["comp_name"] == "hmmt/hmmt_feb_2026"
    assert aime["problem_idx"] == hmmt["problem_idx"] == 1
