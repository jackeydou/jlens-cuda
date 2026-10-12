"""Agent-mode building blocks: cases, mock tool execution, Qwen3.x XML
tool-call parsing and exact prompt/token bookkeeping.

A *case* is pure data (``data/cases/<id>.json``): a system prompt, the tool
schemas the model sees, and for every tool a *mock* describing what the
tool returns. Adding or changing tools never needs server code — only a new
case file (authored in the web UI or by hand).

Mock modes (one generic executor each):

- ``fixed``     every call returns ``return``.
- ``match``     first ``cases[i].when`` that matches the arguments wins;
                ``when`` values are literals, ``"*"`` (present, any value)
                or ``{"regex": "..."}``.
- ``sequence``  the n-th call of this tool returns ``returns[n]``.
- ``manual``    the UI types the result while the stream waits.
- ``http``      POST ``{"name", "arguments", "case_id"}`` to ``url``; the
                response body is the result (for the few tools that must
                really execute somewhere).

When ``match``/``sequence`` has no answer, ``on_miss`` decides: ``manual``
(ask the UI), ``default`` (return ``default``) or ``error`` (stop the run
with a ``fixture_miss`` — the strict reproduction mode).

Qwen3.x (3.5+) tool calls are XML::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

``<tool_call>``/``</tool_call>``/``<tool_response>``/``<think>`` are single
special tokens in the Qwen3.x vocabulary, so the stream can classify every
generated token without re-tokenizing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

MOCK_MODES = ("fixed", "match", "sequence", "manual", "http")
ON_MISS = ("manual", "default", "error")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
DEFAULT_RUN = {"max_steps": 8, "max_tokens_per_step": 2048, "enable_thinking": False}


class CaseError(ValueError):
    """A case file that cannot be used; the message names the bad field."""


# ---------------------------------------------------------------------------
# Case validation
# ---------------------------------------------------------------------------

def _as_text(value: Any) -> str:
    """Tool results are text; JSON values are serialized verbatim."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _int_in(raw: Any, lo: int, hi: int, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw:
        raise CaseError(f"{where} must be an integer")
    if not lo <= int(raw) <= hi:
        raise CaseError(f"{where} must be in [{lo}, {hi}]")
    return int(raw)


def _validate_when(when: Any, props: dict, where: str) -> dict:
    if not isinstance(when, dict) or not when:
        raise CaseError(f"{where}.when must be a non-empty object")
    for key, expected in when.items():
        if props and key not in props:
            raise CaseError(f"{where}.when.{key} is not a parameter of this tool")
        if isinstance(expected, dict):
            if set(expected) != {"regex"} or not isinstance(expected["regex"], str):
                raise CaseError(f'{where}.when.{key} object must be {{"regex": "..."}}')
            try:
                re.compile(expected["regex"])
            except re.error as e:
                raise CaseError(f"{where}.when.{key} regex: {e}") from None
    return dict(when)


def _validate_mock(raw: Any, props: dict, where: str) -> dict:
    raw = raw if raw is not None else {"mode": "manual"}
    if not isinstance(raw, dict):
        raise CaseError(f"{where} must be an object")
    mode = raw.get("mode", "manual")
    if mode not in MOCK_MODES:
        raise CaseError(f"{where}.mode must be one of {', '.join(MOCK_MODES)}")
    mock: dict[str, Any] = {"mode": mode}
    if mode == "fixed":
        if "return" not in raw:
            raise CaseError(f"{where}.return is required for mode=fixed")
        mock["return"] = _as_text(raw["return"])
    elif mode == "match":
        cases = raw.get("cases") or []
        if not isinstance(cases, list):
            raise CaseError(f"{where}.cases must be a list")
        mock["cases"] = []
        for i, c in enumerate(cases):
            if not isinstance(c, dict) or "return" not in c:
                raise CaseError(f"{where}.cases[{i}] needs when + return")
            mock["cases"].append({
                "when": _validate_when(c.get("when"), props, f"{where}.cases[{i}]"),
                "return": _as_text(c["return"]),
            })
    elif mode == "sequence":
        returns = raw.get("returns") or []
        if not isinstance(returns, list):
            raise CaseError(f"{where}.returns must be a list")
        mock["returns"] = [_as_text(r) for r in returns]
    elif mode == "http":
        url = raw.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            raise CaseError(f"{where}.url must be an http(s) URL")
        mock["url"] = url
        mock["timeout_s"] = float(raw.get("timeout_s", 30))
    if mode in ("match", "sequence"):
        on_miss = raw.get("on_miss", "manual")
        if on_miss not in ON_MISS:
            raise CaseError(f"{where}.on_miss must be one of {', '.join(ON_MISS)}")
        mock["on_miss"] = on_miss
        if on_miss == "default":
            if "default" not in raw:
                raise CaseError(f"{where}.default is required for on_miss=default")
            mock["default"] = _as_text(raw["default"])
    return mock


def _validate_tool(raw: Any, i: int) -> dict:
    where = f"tools[{i}]"
    if not isinstance(raw, dict):
        raise CaseError(f"{where} must be an object")
    # Accept OpenAI's {"type": "function", "function": {...}} wrapper too,
    # so tool lists can be pasted from existing harnesses.
    if isinstance(raw.get("function"), dict):
        raw = {**raw["function"], **{k: v for k, v in raw.items()
                                     if k not in ("function", "type")}}
    name = raw.get("name")
    if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
        raise CaseError(f"{where}.name must match {TOOL_NAME_RE.pattern}")
    params = raw.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(params, dict) or params.get("type", "object") != "object":
        raise CaseError(f'{where}.parameters must be a JSON schema with type "object"')
    props = params.get("properties") or {}
    if not isinstance(props, dict):
        raise CaseError(f"{where}.parameters.properties must be an object")
    required = params.get("required") or []
    missing = [r for r in required if r not in props]
    if missing:
        raise CaseError(f"{where}.parameters.required names unknown {missing}")
    return {
        "name": name,
        "description": str(raw.get("description") or ""),
        "parameters": {**params, "type": "object", "properties": props},
        "enabled": bool(raw.get("enabled", True)),
        "mock": _validate_mock(raw.get("mock"), props, f"{where}.mock"),
    }


def validate_case(raw: Any) -> dict:
    """Normalize a case dict or raise CaseError naming the first bad field."""
    if not isinstance(raw, dict):
        raise CaseError("case must be a JSON object")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not CASE_ID_RE.match(case_id):
        raise CaseError("id must be 1-64 chars of [A-Za-z0-9_-]")
    tools_raw = raw.get("tools") or []
    if not isinstance(tools_raw, list):
        raise CaseError("tools must be a list")
    tools = [_validate_tool(t, i) for i, t in enumerate(tools_raw)]
    names = [t["name"] for t in tools]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise CaseError(f"duplicate tool names {dupes}")
    run_raw = raw.get("run") or {}
    if not isinstance(run_raw, dict):
        raise CaseError("run must be an object")
    run = {
        "max_steps": _int_in(run_raw.get("max_steps", DEFAULT_RUN["max_steps"]),
                             1, 64, "run.max_steps"),
        "max_tokens_per_step": _int_in(
            run_raw.get("max_tokens_per_step", DEFAULT_RUN["max_tokens_per_step"]),
            1, 262144, "run.max_tokens_per_step"),
        "enable_thinking": bool(run_raw.get("enable_thinking",
                                            DEFAULT_RUN["enable_thinking"])),
    }
    return {
        "id": case_id,
        "name": str(raw.get("name") or case_id),
        "description": str(raw.get("description") or ""),
        "system": str(raw.get("system") or ""),
        "prompt": str(raw.get("prompt") or ""),
        "tools": tools,
        "run": run,
    }


def case_hash(case: dict) -> str:
    """Content hash of a normalized case: equal hash => identical setup."""
    blob = json.dumps(case, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def enabled_tools(case: dict) -> list[dict]:
    return [t for t in case["tools"] if t["enabled"]]


def tool_schemas(case: dict) -> list[dict]:
    """The OpenAI-style tool list the chat template renders."""
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": t["parameters"]}} for t in enabled_tools(case)]


# ---------------------------------------------------------------------------
# Mock execution
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    return value.strip() if isinstance(value, str) else json.dumps(value, sort_keys=True)


def when_matches(when: dict, args: dict) -> bool:
    for key, expected in when.items():
        if key not in args:
            return False
        actual = args[key]
        if expected == "*":
            continue
        if isinstance(expected, dict):
            text = actual if isinstance(actual, str) else json.dumps(actual)
            if not re.search(expected["regex"], text):
                return False
        elif actual != expected and _norm(actual) != _norm(expected):
            return False
    return True


@dataclass
class MockOutcome:
    """What the mock layer decided for one call.

    ``kind``: ``result`` (``content`` is final), ``manual`` (ask the UI),
    ``http`` (caller performs the request) or ``miss`` (strict stop).
    """

    kind: str
    content: str = ""
    source: str = ""


@dataclass
class MockState:
    """Per-run counters (``sequence`` mode)."""

    calls: dict[str, int] = field(default_factory=dict)


def resolve_mock(tool: dict, args: dict, st: MockState) -> MockOutcome:
    mock = tool["mock"]
    mode = mock["mode"]
    n = st.calls.get(tool["name"], 0)
    st.calls[tool["name"]] = n + 1
    if mode == "fixed":
        return MockOutcome("result", mock["return"], "fixed")
    if mode == "manual":
        return MockOutcome("manual", source="manual")
    if mode == "http":
        return MockOutcome("http", source="http")
    if mode == "match":
        for i, c in enumerate(mock["cases"]):
            if when_matches(c["when"], args):
                return MockOutcome("result", c["return"], f"match[{i}]")
    elif mode == "sequence" and n < len(mock["returns"]):
        return MockOutcome("result", mock["returns"][n], f"sequence[{n}]")
    on_miss = mock.get("on_miss", "manual")
    if on_miss == "default":
        return MockOutcome("result", mock["default"], "default")
    if on_miss == "manual":
        return MockOutcome("manual", source="manual")
    return MockOutcome("miss", source="miss")


# ---------------------------------------------------------------------------
# Qwen3.x XML tool-call parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)


def _strip_one_newline(s: str) -> str:
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def coerce_param(value: str, schema: Any) -> Any:
    """Convert a raw XML parameter string by its JSON-schema type (same
    policy as vLLM's qwen3_coder parser; unknown/failed => keep string)."""
    t = schema.get("type") if isinstance(schema, dict) else None
    types = t if isinstance(t, list) else [t]
    if value.strip() == "null" and "null" in types:
        return None
    if t is None or "string" in types:
        return value
    v = value.strip()
    for ty in types:
        try:
            if ty == "integer":
                return int(v)
            if ty == "number":
                f = float(v)
                return int(f) if f.is_integer() and re.fullmatch(r"-?\d+", v) else f
            if ty == "boolean" and v.lower() in ("true", "false"):
                return v.lower() == "true"
            if ty in ("object", "array"):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            continue
    return value


@dataclass
class ParsedTurn:
    """A generated assistant turn split the way the chat template sees it."""

    reasoning: str
    content: str
    tool_calls: list[dict]
    trailing_text: str = ""

    def template_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        good = [c for c in self.tool_calls if not c.get("error")]
        if good and len(good) == len(self.tool_calls):
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": c["name"], "arguments": c["arguments"]}} for c in good]
        elif self.tool_calls:
            # A malformed call has no structured form; keep it verbatim.
            msg["content"] = self.content + "".join(c["raw"] for c in self.tool_calls)
        return msg


def parse_assistant(text: str, tools: list[dict]) -> ParsedTurn:
    """Split generated text into reasoning / content / tool calls.

    ``text`` is the decoded assistant output (special tokens kept). Calls
    carry ``error`` instead of raising: a malformed call is reported back
    to the model as its tool response, like a real harness would.
    """
    by_name = {t["name"]: t for t in tools}
    reasoning = ""
    body = text
    if "</think>" in body:
        reasoning, body = body.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        body = body.lstrip("\n")
    first = body.find("<tool_call>")
    content = (body if first < 0 else body[:first]).strip()
    calls: list[dict] = []
    last_end = 0
    for m in _TOOL_CALL_RE.finditer(body):
        calls.append(_parse_call(m.group(1), m.group(0), by_name))
        last_end = m.end()
    tail = body[last_end:] if calls else ""
    if "<tool_call>" in tail:
        cut = tail.index("<tool_call>")
        calls.append({"name": None, "arguments": {}, "raw": tail[cut:],
                      "error": "tool call truncated (no </tool_call>)"})
        tail = tail[:cut]
    return ParsedTurn(reasoning, content, calls, tail.strip())


def _parse_call(inner: str, raw: str, by_name: dict[str, dict]) -> dict:
    fm = _FUNCTION_RE.search(inner)
    if not fm:
        return {"name": None, "arguments": {}, "raw": raw,
                "error": "malformed tool call: no <function=...>...</function>"}
    name = fm.group(1).strip()
    tool = by_name.get(name)
    props = (tool or {}).get("parameters", {}).get("properties", {})
    args = {}
    for pm in _PARAM_RE.finditer(fm.group(2)):
        key = pm.group(1).strip()
        args[key] = coerce_param(_strip_one_newline(pm.group(2)), props.get(key))
    call = {"name": name, "arguments": args, "raw": raw, "error": None}
    if tool is None:
        call["error"] = f"unknown tool {name!r}"
    else:
        missing = [r for r in tool["parameters"].get("required", []) if r not in args]
        if missing:
            call["error"] = f"missing required parameters {missing}"
    return call


# ---------------------------------------------------------------------------
# Prompt / token bookkeeping
# ---------------------------------------------------------------------------

def template_messages(case: dict, messages: list[dict]) -> list[dict]:
    """Client transcript -> chat-template messages (system from the case,
    assistant turns re-parsed into reasoning/content/tool_calls)."""
    out: list[dict] = []
    if case["system"].strip():
        out.append({"role": "system", "content": case["system"]})
    tools = enabled_tools(case)
    for m in messages:
        if m["role"] == "assistant":
            out.append(parse_assistant(m["content"], tools).template_message())
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


EMPTY_THINK = "<think>\n\n</think>\n\n"


def _blocks(text: str) -> list[tuple[str, int, int]]:
    """``<|im_start|>role\\n BODY <|im_end|>`` blocks as (role, start, end)."""
    out = []
    i = 0
    while True:
        s = text.find("<|im_start|>", i)
        if s < 0:
            return out
        nl = text.find("\n", s)
        nl = len(text) if nl < 0 else nl
        role = text[s + len("<|im_start|>"):nl]
        e = text.find("<|im_end|>", nl)
        e = len(text) if e < 0 else e
        out.append((role, min(nl + 1, len(text)), e))
        i = e


def message_char_spans(text: str, roles: list[str], has_system: bool) -> list[tuple[int, int]]:
    """Character span of each client message's content in rendered ``text``.

    ``roles`` are the CLIENT roles (no system). Consecutive tool messages
    share one ``user`` block (Qwen3.x template), split by <tool_response>.
    """
    blocks = _blocks(text)
    bi = 1 if blocks and blocks[0][0] == "system" else 0
    if has_system and bi == 0:
        raise ValueError("system message rendered without a system block")
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(roles):
        if bi >= len(blocks):
            raise ValueError("rendered prompt has fewer blocks than messages")
        _, bs, be = blocks[bi]
        bi += 1
        if roles[i] != "tool":
            # Non-thinking history renders an empty think block; it is
            # template scaffolding, not part of what the model said.
            if roles[i] == "assistant" and text.startswith(EMPTY_THINK, bs):
                bs += len(EMPTY_THINK)
            spans.append((bs, be))
            i += 1
            continue
        cursor = bs
        while i < len(roles) and roles[i] == "tool":
            open_ = text.find("<tool_response>\n", cursor, be)
            close = text.find("\n</tool_response>", open_, be)
            if open_ < 0 or close < 0:
                raise ValueError("tool block without matching <tool_response>")
            spans.append((open_ + len("<tool_response>\n"), close))
            cursor = close + len("\n</tool_response>")
            i += 1
    return spans


def char_to_token_spans(offsets: list[tuple[int, int]],
                        spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Token span [a, b) of the tokens STARTING inside each char span."""
    starts = [o[0] for o in offsets]
    out = []
    for cs, ce in spans:
        a = next((k for k, s in enumerate(starts) if s >= cs), len(starts))
        b = a
        while b < len(starts) and starts[b] < ce:
            b += 1
        out.append((a, b))
    return out


def encode_with_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def render_prompt(tok, case: dict, messages: list[dict]):
    """Render the full prompt (with tools + generation prompt).

    Returns ``(ids, token_spans)``: token ids exactly as the template
    tokenizes them, and each client message's content token span.
    """
    tmsgs = template_messages(case, messages)
    text = tok.apply_chat_template(
        tmsgs, tools=tool_schemas(case) or None, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=case["run"]["enable_thinking"])
    ids, offsets = encode_with_offsets(tok, text)
    char_spans = message_char_spans(
        text, [m["role"] for m in messages], has_system=bool(case["system"].strip()))
    return ids, char_to_token_spans(offsets, char_spans)


def generation_prompt(enable_thinking: bool) -> str:
    return "<|im_start|>assistant\n" + (
        "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


def tool_results_suffix(results: list[str], enable_thinking: bool):
    """Text appended after a generated tool-calling turn.

    Starts with the turn's own ``<|im_end|>`` (sampled as EOS but not yet
    fed), then one ``user`` block holding every ``<tool_response>`` (the
    Qwen3.x template groups consecutive tool messages), then the next
    generation prompt. Returns ``(text, content_char_spans)``.
    """
    s = "<|im_end|>\n<|im_start|>user"
    spans = []
    for r in results:
        s += "\n<tool_response>\n"
        start = len(s)
        s += r.strip()
        spans.append((start, len(s)))
        s += "\n</tool_response>"
    s += "<|im_end|>\n" + generation_prompt(enable_thinking)
    return s, spans


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """First index where ``a`` is not a prefix-match of ``b`` (None = a ⊑ b)."""
    for i, x in enumerate(a):
        if i >= len(b) or b[i] != x:
            return i
    return None
