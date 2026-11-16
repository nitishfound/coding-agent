import json
import shutil
from pathlib import Path

import httpx
import pytest

from agent import Agent, OpenAICompatibleModel, Reply, ScriptedModel, ToolCall, Workspace, call
from agent.llm import parse_tool_calls

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "buggy_calc"


@pytest.fixture
def ws(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "proj")
    return Workspace(tmp_path / "proj")


def fixing_script() -> list[Reply]:
    """What a good model does on the example: look, test, fix 3 bugs, re-test, finish."""
    return [
        Reply("Let me see what fails.", [call("run_tests")]),
        Reply(None, [call("read_file", path="calc.py")]),
        Reply(None, [call("edit_file", path="calc.py", old="sum(values) / (len(values) - 1)", new="sum(values) / len(values)")]),
        Reply(None, [
            call("edit_file", path="calc.py", old="    n = len(values)\n", new="    values = sorted(values)\n    n = len(values)\n"),
            call("edit_file", path="calc.py", old="if n < 1:", new="if n < 2:"),
        ]),
        Reply(None, [call("run_tests")]),
        Reply(None, [call("finish", summary="Fixed mean divisor, sorted input in median, 1 is not prime.")]),
    ]


def test_agent_fixes_the_example_project(ws, tmp_path):
    transcript = tmp_path / "run.jsonl"
    result = Agent(ScriptedModel(fixing_script()), ws, transcript=transcript).run("Make the tests pass")
    assert result.status == "finished"
    assert [s.tool for s in result.steps] == ["run_tests", "read_file", "edit_file", "edit_file", "edit_file", "run_tests"]
    assert result.steps[0].result.startswith("FAILED")
    assert result.steps[-1].result.startswith("PASSED")
    assert "+    values = sorted(values)" in result.diff
    assert len(transcript.read_text().splitlines()) == 6


def test_tool_results_go_back_to_the_model_with_matching_ids(ws):
    model = ScriptedModel(fixing_script()[:2] + [Reply(None, [call("finish", summary="done")])])
    Agent(model, ws).run("task")
    last_messages = model.seen[-1]
    assistant = [m for m in last_messages if m["role"] == "assistant"][-1]
    tool_msg = last_messages[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "   1 | " in tool_msg["content"]  # the file content came back


def test_errors_are_reported_to_the_model_and_it_can_recover(ws):
    bad_json = ToolCall("c1", "read_file", {}, "{not json", parse_error="invalid JSON arguments")
    model = ScriptedModel([
        Reply(None, [call("delete_everything")]),
        Reply(None, [call("read_file", file="calc.py")]),  # wrong argument name
        Reply(None, [bad_json]),
        Reply(None, [call("read_file", path="../../etc/passwd")]),
        Reply(None, [call("read_file", path="calc.py")]),
        Reply(None, [call("finish", summary="ok")]),
    ])
    result = Agent(model, ws, max_consecutive_errors=5).run("task")
    assert result.status == "finished"
    assert [s.ok for s in result.steps] == [False, False, False, False, True]
    assert "unknown tool" in result.steps[0].result
    assert "bad arguments" in result.steps[1].result
    assert "outside the workspace" in result.steps[3].result


def test_stops_after_too_many_errors(ws):
    model = ScriptedModel([Reply(None, [call("nope")]) for _ in range(10)])
    assert Agent(model, ws, max_consecutive_errors=3).run("task").status == "too_many_errors"


def test_stops_at_max_steps(ws):
    model = ScriptedModel([Reply(None, [call("list_files")]) for _ in range(10)])
    result = Agent(model, ws, max_steps=4).run("task")
    assert result.status == "max_steps" and len(result.steps) == 4


def test_plain_text_reply_ends_the_run(ws):
    result = Agent(ScriptedModel([Reply("I can't help with that.")]), ws).run("task")
    assert result.status == "model_stopped" and result.diff == ""


def test_old_tool_outputs_are_compacted(ws):
    model = ScriptedModel([Reply(None, [call("read_file", path="calc.py")]) for _ in range(8)] + [Reply(None, [call("finish", summary="x")])])
    Agent(model, ws, keep_full_results=2).run("task")
    tool_msgs = [m for m in model.seen[-1] if m["role"] == "tool"]
    assert "[older output shortened]" in tool_msgs[0]["content"]
    assert "[older output shortened]" not in tool_msgs[-1]["content"]


def test_parses_openai_tool_calls_over_http():
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["type"] == "function" and body["tool_choice"] == "auto"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "abc", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"calc.py\"}"}}
            ]}}],
            "usage": {"total_tokens": 42},
        })

    model = OpenAICompatibleModel("https://x.test/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    reply = model.chat([{"role": "user", "content": "hi"}], [{"name": "read_file", "parameters": {}}])
    assert reply.tool_calls[0].name == "read_file" and reply.tool_calls[0].arguments == {"path": "calc.py"}
    assert reply.usage["total_tokens"] == 42


def test_parse_tool_calls_handles_garbage():
    calls = parse_tool_calls([{"id": "1", "function": {"name": "x", "arguments": "[1,2]"}},
                              {"id": "2", "function": {"name": "y", "arguments": "{oops"}}])
    assert calls[0].parse_error and calls[1].parse_error
