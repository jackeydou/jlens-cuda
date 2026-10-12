"""Agent mode: case validation, mock resolution, XML tool-call parsing, and
the /api/agent_stream loop (no model; the Qwen3.8 tokenizer when cached)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import jlens_qwen.serve as serve
from jlens_qwen import agent

WEATHER = {
    "name": "get_weather",
    "description": "Tomorrow's weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                       "metric": {"type": "boolean"}, "opts": {"type": "object"}},
        "required": ["city"],
    },
    "mock": {
        "mode": "match",
        "cases": [
            {"when": {"city": "Paris"}, "return": {"temp": 18, "cond": "sunny"}},
            {"when": {"city": {"regex": "(?i)^rom"}}, "return": "rain"},
        ],
        "on_miss": "error",
    },
}


def _case(**overrides):
    return agent.validate_case({"id": "weather", "tools": [WEATHER], **overrides})


# ----- case validation -----

def test_case_normalizes_and_hash_is_stable():
    case = _case()
    assert case["run"] == agent.DEFAULT_RUN
    assert case["tools"][0]["mock"]["cases"][0]["return"] == '{"temp": 18, "cond": "sunny"}'
    assert agent.case_hash(case) == agent.case_hash(_case())
    assert agent.case_hash(case) != agent.case_hash(_case(system="x"))


def test_case_accepts_openai_function_wrapper():
    case = agent.validate_case({"id": "c", "tools": [{
        "type": "function",
        "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
    }]})
    assert case["tools"][0]["name"] == "f"
    assert case["tools"][0]["mock"] == {"mode": "manual"}


@pytest.mark.parametrize("raw, match", [
    ({"id": "bad id"}, "id must"),
    ({"id": "c", "tools": [{"name": "f"}, {"name": "f"}]}, "duplicate"),
    ({"id": "c", "tools": [{**WEATHER, "mock": {"mode": "match", "cases": [
        {"when": {"town": "Paris"}, "return": "x"}]}}]}, "not a parameter"),
    ({"id": "c", "tools": [{**WEATHER, "mock": {"mode": "fixed"}}]}, "return is required"),
    ({"id": "c", "tools": [{**WEATHER, "mock": {"mode": "match", "cases": [],
                                                "on_miss": "default"}}]}, "default is required"),
    ({"id": "c", "tools": [{**WEATHER, "mock": {"mode": "match", "cases": [
        {"when": {"city": {"regex": "("}}, "return": "x"}]}}]}, "regex"),
    ({"id": "c", "run": {"max_steps": 0}}, "max_steps"),
])
def test_case_rejects(raw, match):
    with pytest.raises(agent.CaseError, match=match):
        agent.validate_case(raw)


# ----- mock resolution -----

def test_match_regex_wildcard_and_strict_miss():
    tool = _case()["tools"][0]
    st = agent.MockState()
    assert agent.resolve_mock(tool, {"city": "Paris"}, st).source == "match[0]"
    assert agent.resolve_mock(tool, {"city": "Roma"}, st).content == "rain"
    assert agent.resolve_mock(tool, {"city": "Oslo"}, st).kind == "miss"
    assert agent.when_matches({"city": "*"}, {"city": "x"})
    assert not agent.when_matches({"city": "*"}, {})
    assert agent.when_matches({"days": 2}, {"days": "2"})


def test_sequence_then_default():
    tool = agent.validate_case({"id": "c", "tools": [{
        "name": "next_page",
        "mock": {"mode": "sequence", "returns": ["p1", "p2"],
                 "on_miss": "default", "default": "END"},
    }]})["tools"][0]
    st = agent.MockState()
    assert [agent.resolve_mock(tool, {}, st).content for _ in range(3)] == ["p1", "p2", "END"]


# ----- XML parsing -----

def test_parse_parallel_calls_with_type_coercion():
    tools = _case()["tools"]
    text = ("Let me check.\n\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n"
            "<parameter=metric>\ntrue\n</parameter>\n"
            "<parameter=opts>\n{\"lang\": \"fr\"}\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n</parameter>\n"
            "</function>\n</tool_call>")
    p = agent.parse_assistant(text, tools)
    assert p.content == "Let me check."
    assert [c["arguments"] for c in p.tool_calls] == [
        {"city": "Paris", "days": 2, "metric": True, "opts": {"lang": "fr"}},
        {"city": "New\nYork"},
    ]
    assert all(c["error"] is None for c in p.tool_calls)
    msg = p.template_message()
    assert msg["tool_calls"][0]["function"]["arguments"]["days"] == 2


def test_parse_thinking_split_and_errors():
    tools = _case()["tools"]
    p = agent.parse_assistant(
        "<think>\nhmm\n</think>\n\n<tool_call>\n<function=nope>\n</function>\n</tool_call>"
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", tools)
    assert p.reasoning == "hmm"
    assert p.content == ""
    assert [c["error"] for c in p.tool_calls] == [
        "unknown tool 'nope'", "missing required parameters ['city']",
        "tool call truncated (no </tool_call>)"]
    # Malformed calls have no structured form: kept verbatim as content.
    assert "tool_calls" not in p.template_message()


def test_no_tool_call_is_final_answer():
    p = agent.parse_assistant("Paris is sunny.", _case()["tools"])
    assert (p.content, p.tool_calls) == ("Paris is sunny.", [])


def test_char_to_token_spans():
    offsets = [(0, 3), (3, 4), (4, 9), (9, 10)]
    assert agent.char_to_token_spans(offsets, [(4, 9), (10, 10)]) == [(2, 3), (4, 4)]


# ----- against the real Qwen3.8 chat template -----

def _qwen_tokenizer():
    snaps = sorted(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("Qwen3.8 tokenizer not cached")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(snaps[0].parent))


@pytest.fixture(scope="module")
def qwen_tok():
    return _qwen_tokenizer()


CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>")


@pytest.mark.parametrize("thinking", [False, True])
def test_render_and_suffix_match_template(qwen_tok, thinking):
    case = _case(system="Be brief.", run={"enable_thinking": thinking})
    msgs = [{"role": "user", "content": "Paris or Rome?"}]
    ids, spans = agent.render_prompt(qwen_tok, case, msgs)
    ref = qwen_tok.apply_chat_template(
        agent.template_messages(case, msgs), tools=agent.tool_schemas(case),
        add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
    assert ids == list(ref)
    assert qwen_tok.decode(ids[slice(*spans[0])]) == "Paris or Rome?"

    gen = ("Compare.\n</think>\n\n" if thinking else "") + CALL
    suffix, char_spans = agent.tool_results_suffix(['{"t": 18}', "rain"], thinking)
    sids, offs = agent.encode_with_offsets(qwen_tok, suffix)
    ledger = ids + qwen_tok.encode(gen, add_special_tokens=False) + sids
    full = msgs + [{"role": "assistant", "content": gen},
                   {"role": "tool", "content": '{"t": 18}'},
                   {"role": "tool", "content": "rain"}]
    rendered, spans2 = agent.render_prompt(qwen_tok, case, full)
    assert ledger == rendered
    assert [qwen_tok.decode(rendered[slice(*s)]) for s in spans2[2:]] == ['{"t": 18}', "rain"]
    # History spans skip the empty-think scaffolding, keep real reasoning.
    past = qwen_tok.decode(rendered[slice(*spans2[1])])
    assert past.startswith("<think>\nCompare." if thinking else "<tool_call>")
    assert [qwen_tok.decode(sids[slice(*s)])
            for s in agent.char_to_token_spans(offs, char_spans)] == ['{"t": 18}', "rain"]


# ----- /api/agent_stream end to end (scripted sampler, fake session) -----

class _Session:
    def __init__(self, fed):
        self.fed = fed

    def set_edits(self, edits):
        pass

    def extend(self, ids):
        self.fed.extend(ids)
        return None, {}


class _Model:
    n_layers = 4

    def __init__(self, tok, fed):
        self.tokenizer = tok
        self.fed = fed

    def make_stream(self, capture_layers=None):
        return _Session(self.fed)


def _parse_sse(raw: bytes):
    out = []
    for frame in raw.decode().strip().split("\n\n"):
        lines = frame.splitlines()
        out.append((lines[0].removeprefix("event: "),
                    json.loads(lines[1].removeprefix("data: "))))
    return out


def _run_agent(monkeypatch, tmp_path, tok, case, script, on_event=None):
    fed: list[int] = []
    (tmp_path / f"{case['id']}.json").write_text(json.dumps(case))
    monkeypatch.setattr(serve, "_cases_dir", tmp_path)
    monkeypatch.setattr(serve, "_model", _Model(tok, fed))
    monkeypatch.setattr(serve, "_lens", None)
    monkeypatch.setattr(serve, "_record_layers", lambda: [3])
    monkeypatch.setattr(serve, "_readout_at_positions", lambda acts, pos, layers, n: {
        l: {"top_ids": [[0]] * len(pos), "top_tokens": [["x"]] * len(pos),
            "top_scores": [[0.0]] * len(pos)} for l in layers})
    queue = list(script)
    monkeypatch.setattr(serve, "_sample_tok", lambda logits, temp: queue.pop(0))

    async def main():
        resp = await serve.agent_stream(serve.AgentStreamRequest(
            case_id=case["id"],
            messages=[serve.AgentMessage(role="user", content="Paris or Rome?")]))
        body = b""
        async for chunk in resp.body_iterator:
            body += chunk
            if on_event:
                for ev, data in _parse_sse(chunk):
                    await on_event(ev, data)
        return _parse_sse(body)

    return asyncio.run(main()), fed, queue


def test_agent_stream_two_steps(monkeypatch, tmp_path, qwen_tok):
    case = _case()
    eos = qwen_tok.eos_token_id
    step1 = qwen_tok.encode(CALL, add_special_tokens=False)
    step2 = qwen_tok.encode("Paris: sunny.", add_special_tokens=False)
    events, fed, left = _run_agent(monkeypatch, tmp_path, qwen_tok, case,
                                   step1 + [eos] + step2 + [eos])
    assert left == []
    names = [e for e, _ in events]
    assert names[0] == "stream_start" and names[-1] == "done"
    results = [d for e, d in events if e == "tool_result"]
    assert [(r["content"], r["source"]) for r in results] == [
        ('{"temp": 18, "cond": "sunny"}', "match[0]"), ("rain", "match[1]")]
    tool_snaps = [d for e, d in events if e == "snapshot" and d["role"] == "tool"]
    assert [(d["msg_idx"], "".join(d["token_strs"])) for d in tool_snaps] == [
        (2, '{"temp": 18, "cond": "sunny"}'), (3, "rain")]
    segs = {d["segment"] for e, d in events if e == "snapshot" and d.get("step") == 1
            and d["token_idx"] >= 0}
    assert segs == {"tool_call", "content"}
    done = dict(events)["done"]
    assert (done["stop_reason"], done["n_steps"], done["msg_idx"]) == ("final", 2, 4)
    assert dict(events)["ledger_check"]["match"] is True
    assert len(fed) == done["ledger_len"]


def test_agent_stream_strict_miss_stops(monkeypatch, tmp_path, qwen_tok):
    call = CALL.replace("Rome", "Oslo")
    script = qwen_tok.encode(call, add_special_tokens=False) + [qwen_tok.eos_token_id]
    events, _, _ = _run_agent(monkeypatch, tmp_path, qwen_tok, _case(), script)
    assert dict(events)["done"]["stop_reason"] == "fixture_miss"
    assert [d["source"] for e, d in events if e == "tool_result"] == ["match[0]", "miss"]


def test_agent_stream_manual_result(monkeypatch, tmp_path, qwen_tok):
    case = agent.validate_case({"id": "manual", "tools": [
        {**WEATHER, "mock": {"mode": "manual"}}]})
    eos = qwen_tok.eos_token_id
    one_call = CALL.split("\n<tool_call>")[0]
    script = (qwen_tok.encode(one_call, add_special_tokens=False) + [eos]
              + qwen_tok.encode("ok", add_special_tokens=False) + [eos])
    stream = {}

    async def on_event(ev, data):
        if ev == "stream_start":
            stream["id"] = data["stream_id"]
        if ev == "tool_waiting":
            await serve.agent_tool_result(serve.AgentToolResultRequest(
                stream_id=stream["id"], call_id=data["call_id"], content="typed by hand"))

    events, _, _ = _run_agent(monkeypatch, tmp_path, qwen_tok, case, script, on_event)
    assert [(d["content"], d["source"]) for e, d in events if e == "tool_result"] == [
        ("typed by hand", "manual")]
    assert dict(events)["done"]["stop_reason"] == "final"
    assert dict(events)["ledger_check"]["match"] is True
