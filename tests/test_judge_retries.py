"""Judge format recovery and bounded retries without model requests."""

import pytest

from matharena.solvers.judges.judge_pool import JudgePool
from matharena.solvers.judges.judge_response import JudgeResponse
from matharena.solvers.judges.simple_judge import SimpleJudge


@pytest.mark.parametrize(
    "outputs, expected",
    [
        ([0], 0), ([3], 3), ([None, 3], 3), ([4, 2], 2),
        ([None, 4, 0], 0), ([None, None, None], None),
    ],
)
def test_judge_retries_only_invalid_scores_and_accumulates_costs(outputs, expected):
    agents, inputs = [], []

    class FakeJudge:
        def __init__(self, **kwargs):
            self.idx = kwargs["batch_idx"]
            self.attempt = len(agents)
            agents.append(self)

        def solve(self, *args):
            inputs.append(args)
            return JudgeResponse(
                self.idx, outputs[self.attempt], f"Attempt {self.attempt + 1}",
                {
                    "cost": 0.25, "input_tokens": 10, "output_tokens": 5,
                    "cached_input_tokens": 3, "cached_write_tokens": 1,
                    "time": 2, "request_time": 1, "n_retries": 0,
                    "read_cost": 0.75, "model": "gemini-test",
                    "cost_by_judge_model": {"gemini-test": 0.25},
                },
                [{"step": "judge", "messages": [{"content": f"Attempt {self.attempt + 1}"}]}],
                {"existing_info": True},
            )

    pool = JudgePool({"scaffold_name": "simple_judge", "judge_points_max": 3})
    pool.AGENT_CLASS = FakeJudge
    response = pool._run_agent(7, 29, 1, ("Problem", {
        "guidelines": "Rubric", "ground_truth_solutions": [],
        "student_answer": "Answer", "original_problem_statement": "Reference",
    }))
    attempts = len(outputs)
    assert len(agents) == attempts
    assert inputs == [("Problem", "Rubric", [], "Answer", "Reference")] * attempts
    assert response.idx == 7 and response.points == expected
    assert response.explanation == f"Attempt {attempts}"
    assert response.additional_info == {"existing_info": True, "judge_attempts": attempts}
    cost = response.detailed_cost
    for key, per_attempt in {
        "cost": 0.25, "input_tokens": 10, "output_tokens": 5,
        "cached_input_tokens": 3, "cached_write_tokens": 1,
        "time": 2, "request_time": 1,
    }.items():
        assert cost[key] == per_attempt * attempts
    assert cost["n_retries"] == attempts - 1
    assert cost["read_cost"] == 0.75
    assert cost["model"] == "gemini-test"
    assert cost["cost_by_judge_model"] == {"gemini-test": 0.25 * attempts}
    assert [h["judge_attempt"] for h in response.history] == list(range(1, attempts + 1))


@pytest.mark.parametrize("text, expected", [
    ("<points>0</points>", 0),
    ("<points>3</points>", 3),
    ("<points>Final score: 3</points><assessment>False as written.</assessment>", 3),
    ("<POINTS> final SCORE:  2 \n</POINTS>", 2),
    ("<points>4</points>", None),
    ("<points>9</points>", None),
    ("<points>3.0</points>", None),
    ("<points>3 or 2</points>", None),
    ("<points>3</points><points>2</points>", None),
    ("<points>unknown</points><points>3</points>", None),
    ("No score returned.", None),
])
def test_simple_judge_parses_unambiguous_score_only(monkeypatch, text, expected):
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", lambda **kwargs: object())
    monkeypatch.setattr(SimpleJudge, "_query", lambda self, *args: [
        {"role": "assistant", "content": text},
    ])
    config = {
        "model_config": {}, "enabled_tools": [], "max_tool_calls": 0,
        "judge_prompt": "{problem_statement} {student_answer}", "judge_points_max": 3,
    }
    response = SimpleJudge(0, 29, 1, config).solve("Problem", "", [], "Answer")
    assert response.points == expected
    assert response.explanation == text


def test_labeled_score_still_respects_allowed_points(monkeypatch):
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", lambda **kwargs: object())
    monkeypatch.setattr(SimpleJudge, "_query", lambda self, *args: [
        {"role": "assistant", "content": "<points>Final score: 2</points>"},
    ])
    config = {
        "model_config": {}, "enabled_tools": [], "max_tool_calls": 0,
        "judge_prompt": "{problem_statement}", "allowed_points": [0, 1],
    }
    assert SimpleJudge(0, 29, 1, config).solve("Problem", "", [], "Answer").points is None
