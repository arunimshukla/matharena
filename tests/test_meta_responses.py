"""Exercise Meta configuration and Responses transport without network access."""

import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.runner import Runner


@pytest.fixture(autouse=True)
def isolated_api(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "meta-test-key")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(api_module.request_logger, "enabled", False)
    monkeypatch.setattr(api_module.time, "sleep", lambda _: None)


def muse_args():
    runner = object.__new__(Runner)
    # Meta uses our local tool implementation, even when an OpenAI built-in exists.
    runner.competition_config = {
        "max_tool_calls": 1,
        "tools": [{
            "tool_spec": {"type": "function", "function": {
                "name": "execute_code", "parameters": {"type": "object"},
            }},
            "tool_spec_openai_responses_api": {"type": "code_interpreter"},
        }],
    }
    path = Path(__file__).resolve().parents[1] / "configs/models/meta/muse_spark_13.yaml"
    config = runner.load_solver_config(str(path))
    args = runner._prepare_default_api_client_args(config["model_config"])
    # The runner strips CLI settings when a competition uses the regular API.
    for key in ("harness", "harness_version", "harness_config"):
        args.pop(key, None)
    assert callable(args["tools"][0][0])
    assert args["tools"][0][1]["type"] == "function"
    args.update(max_tokens=512, max_retries=1, max_retries_inner=0,
                sleep_on_error=0, sleep_after_request=0, throw_error_on_failure=True)
    return args


@pytest.mark.parametrize("key_source", ["meta", "model", "explicit"])
def test_meta_credentials_and_endpoint(monkeypatch, key_source):
    monkeypatch.setenv("MODEL_API_KEY", "model-test-key")
    kwargs = {}
    expected = "meta-test-key"
    if key_source == "model":
        monkeypatch.delenv("META_API_KEY")
        expected = "model-test-key"
    elif key_source == "explicit":
        monkeypatch.setenv("TEST_META_CREDENTIAL", "explicit-test-key")
        kwargs["api_key_env"] = "TEST_META_CREDENTIAL"
        expected = "explicit-test-key"
    client = APIClient(model="muse-spark-1.3", api="meta", **kwargs)
    assert client.api_key == expected
    assert client.api == "openai"
    assert client.base_url == "https://api.meta.ai/v1"


def response_body(output, input_tokens=100, output_tokens=30, cached=20, status="completed"):
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "muse-spark-1.3", "status": status, "output": output,
        "error": None, "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens,
                  "input_tokens_details": {"cached_tokens": cached},
                  "output_tokens_details": {"reasoning_tokens": 10}},
    }


def sse_response(body):
    created = {**body, "status": "in_progress", "output": [], "usage": None}
    events = [
        {"type": "response.created", "response": created, "sequence_number": 0},
        {"type": "response." + body["status"], "response": body, "sequence_number": 1},
    ]
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                          content="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events))


def install_transport(monkeypatch, handler, dict_responses=False):
    @contextmanager
    def dictionary_stream(stream):
        def events():
            for event in stream:
                if hasattr(event, "response"):
                    event.response = event.response.model_dump()
                yield event

        with stream:
            yield events()

    def factory(**kwargs):
        client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        if dict_responses:
            original_create = client.responses.create

            def create(**payload):
                response = original_create(**payload)
                return dictionary_stream(response) if payload.get("stream") else response.model_dump()

            monkeypatch.setattr(client.responses, "create", create)
        return client

    monkeypatch.setattr(api_module, "OpenAI", factory)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("final_status", ["completed", "incomplete"])
@pytest.mark.parametrize("dict_responses", [True, False])
def test_responses_tool_round_trip_and_usage(monkeypatch, stream, final_status, dict_responses):
    requests = []
    executions = []
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [],
                 "encrypted_content": "opaque-reasoning-state", "status": "completed"}
    tool_call = {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                 "name": "add", "arguments": '{"a":6,"b":7}', "status": "completed"}
    message = {"id": "msg_1", "type": "message", "role": "assistant",
               "status": final_status, "content": [
                   {"type": "output_text", "text": "13", "annotations": []},
               ]}
    responses = [response_body([reasoning, tool_call]),
                 response_body([message], 200, 50, 40, final_status)]

    def handler(request):
        assert str(request.url) == "https://api.meta.ai/v1/responses"
        assert request.headers["Authorization"] == "Bearer meta-test-key"
        body = json.loads(request.content)
        requests.append(body)
        result = responses[len(requests) - 1]
        return sse_response(result) if body.get("stream") else httpx.Response(200, json=result)

    def add(a, b):
        executions.append((a, b))
        return str(a + b)

    install_transport(monkeypatch, handler, dict_responses=dict_responses)
    args = muse_args()
    args.update(stream_openai_responses=stream, tools=[(add, {
        "type": "function", "function": {"name": "add", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }},
    })])
    client = APIClient(**args)
    [(idx, conversation, usage)] = client.run_queries(
        [[{"role": "user", "content": "Use add to compute 6+7."}]], no_tqdm=True
    )
    assert idx == 0
    assert executions == [(6, 7)]
    assert conversation[-1]["content"].strip() == "13"
    assert len(requests) == 2
    assert usage["n_retries"] == 0
    for body in requests:
        assert body["model"] == "muse-spark-1.3"
        assert body.get("stream", False) is stream
        assert body["max_output_tokens"] == 512
        assert body["reasoning"] == {"effort": "max"}
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["tools"][0]["name"] == "add"
    replay = requests[1]["input"]
    assert replay[1]["encrypted_content"] == "opaque-reasoning-state"
    assert "status" not in replay[1]
    assert replay[2]["call_id"] == replay[3]["call_id"] == "call_1"
    assert replay[3]["type"] == "function_call_output"
    assert replay[3]["output"].startswith("13")
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 80  # Already includes reasoning tokens.
    assert usage["cached_input_tokens"] == 60
    assert usage["cost"] == pytest.approx((240 * 1.25 + 60 * .15 + 80 * 4.25) / 1e6)


@pytest.mark.parametrize("event", [
    {"type": "response.created", "response": response_body([])},
    {"type": "response.failed", "response": {
        **response_body([], status="failed"), "error": {"code": "server_error", "message": "Failed"},
    }},
    {"type": "error", "code": "server_error", "message": "Failed", "param": None},
])
@pytest.mark.parametrize("dict_responses", [True, False])
def test_failed_or_truncated_stream_is_not_a_success(monkeypatch, event, dict_responses):
    def handler(request):
        created = {"type": "response.created", "response": response_body([]), "sequence_number": 0}
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=(
            f"data: {json.dumps(created)}\n\ndata: {json.dumps({**event, 'sequence_number': 1})}\n\n"
        ))

    install_transport(monkeypatch, handler, dict_responses=dict_responses)
    logged = []
    monkeypatch.setattr(api_module.request_logger, "log_response", lambda **kwargs: logged.append(kwargs))
    client = APIClient(**muse_args())
    with pytest.raises(ValueError, match="Max outer retries reached"):
        list(client.run_queries([[{"role": "user", "content": "Hello"}]], no_tqdm=True))
    errors = [entry["exception"]["exception"] for entry in logged if "exception" in entry]
    expected_error = (
        "ended without a completed or incomplete response"
        if event["type"] == "response.created" else "Failed"
    )
    assert len(errors) == 1 and expected_error in errors[0]
