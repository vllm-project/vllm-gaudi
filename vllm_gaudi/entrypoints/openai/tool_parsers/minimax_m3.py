# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Python MiniMax-M3 tool-call parser for vLLM on HPU (streaming-hardened).

Upstream vLLM ships a MiniMax-M3 tool parser (``minimax_m3``) implemented in
Rust (the ``vllm._rust_tool_parser`` PyO3 extension). That extension is not
compiled into the Gaudi ``+empty`` wheel, so the built-in parser 500s on HPU.
This module is a pure-Python re-implementation of the same XML tool-call grammar
that runs anywhere, registered under the name ``minimax_m3_py`` so it never
collides with the upstream Rust ``minimax_m3``.

It is registered as an out-of-tree tool parser by ``vllm_gaudi`` at plugin load
(see ``vllm_gaudi.register_tool_parsers``), so **no ``--tool-parser-plugin`` file
is needed** -- just select it at serve time::

    vllm serve ... \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3_py

Streaming is truncation-safe (the real fix for the customer "agent gets stuck
after the discovery phase" symptom with MiniMax-M3).

Why a naive streaming path stalls
---------------------------------
Buffering a whole ``<invoke>...</invoke>`` and only emitting the
``DeltaToolCall`` (name + full arguments) once ``</invoke>`` arrives means that
if the model is cut off (``finish_reason=length``) before ``</invoke>`` -- e.g.
it inlined a large file into one tool-call argument -- **nothing is ever
emitted**. The client (e.g. Cursor) then receives zero tool calls and no
content, and hangs.

What this parser does instead (incremental streaming)
-----------------------------------------------------
It follows vLLM's own streaming contract (see
``vllm/tool_parsers/abstract_tool_parser.py`` and
``vllm/parser/abstract_parser.py``):

1. **Name first.** The instant ``<invoke name="X">`` is seen, a ``DeltaToolCall``
   carrying just the tool ``name`` + ``id`` is emitted.
2. **Argument deltas.** As parameters arrive, their JSON is streamed as
   append-only ``arguments`` fragments. Fully-closed parameters are schema-coerced
   exactly like the non-streaming path; a trailing *string* parameter (the common
   "large file content" case) is streamed character-by-character (JSON-escaped).
3. **End-of-stream flush.** ``prev_tool_call_arr`` / ``streamed_args_for_tool``
   are kept in sync, so vLLM's ``finalize_generation`` (called with
   ``finished=True`` on the last chunk) auto-closes a truncated argument via
   ``get_remaining_unstreamed_args()``.

Net effect on truncation: the client still receives ``name`` + partial (but
valid, auto-closed) ``arguments`` + ``finish_reason=length`` -- a visible,
recoverable tool call instead of a silent hang. Completed invokes stream exactly
the same final arguments (no regression), and multi-invoke turns keep working.

Defensive: any parse error degrades to returning content instead of a 500.
"""

import json
import math
import re
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParser, ToolParserManager

logger = init_logger(__name__)

NS = "]<]minimax[>["
TOOL_CALL_START = NS + "<tool_call>"
TOOL_CALL_END = NS + "</tool_call>"
INVOKE_START = NS + "<invoke"
# MiniMax-M3 rarely drops the leading "<" of the invoke open tag, emitting
# "]<]minimax[>[invoke name=..." instead of "]<]minimax[>[<invoke name=...".
# Tolerate that variant, mirroring the headless-parameter recovery already done
# for dropped parameter opening tags (see the NS branches below).
INVOKE_START_NOLT = NS + "invoke"
INVOKE_END = NS + "</invoke>"
ELEMENT_START = NS + "<"
ELEMENT_END_START = NS + "</"
MIXED_TEXT_FIELD = "$text"

_WS = " \t\r\n"
_INVOKE_NAME_RE = re.compile(r"""name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")

# Sentinel for "schema-aware conversion failed, use fallback".
_FAIL = object()


def _find_invoke_open(text: str, start: int) -> tuple[int, int]:
    """Find the next invoke open tag at/after ``start``.

    Returns ``(pos, opener_len)`` for the earliest match of either the
    well-formed ``INVOKE_START`` or the dropped-"<" variant
    ``INVOKE_START_NOLT`` (a known MiniMax-M3 quirk); on a tie the well-formed
    form wins. Returns ``(-1, 0)`` when neither is present. This can never
    false-positive on well-formed text: right after the namespace marker a real
    invoke has "<", so ``INVOKE_START_NOLT`` only matches where "<" was dropped.
    """
    p_full = text.find(INVOKE_START, start)
    p_nolt = text.find(INVOKE_START_NOLT, start)
    if p_full == -1:
        return (p_nolt, len(INVOKE_START_NOLT)) if p_nolt != -1 else (-1, 0)
    if p_nolt == -1 or p_full <= p_nolt:
        return (p_full, len(INVOKE_START))
    return (p_nolt, len(INVOKE_START_NOLT))


class _ParseError(Exception):
    """Recoverable parse failure; the affected invoke/block is dropped."""


# --------------------------------------------------------------------------- #
# Small cursor over a fully-arrived string.
# --------------------------------------------------------------------------- #
class _Cur:
    __slots__ = ("s", "i", "n")

    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def eof(self) -> bool:
        return self.i >= self.n

    def startswith(self, lit: str) -> bool:
        return self.s.startswith(lit, self.i)

    def consume(self, lit: str) -> None:
        if not self.s.startswith(lit, self.i):
            raise _ParseError(f"expected {lit!r}")
        self.i += len(lit)

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in _WS:
            self.i += 1

    def take_until(self, marker: str) -> str:
        j = self.s.find(marker, self.i)
        if j == -1:
            raise _ParseError(f"marker {marker!r} not found")
        val = self.s[self.i:j]
        self.i = j
        return val


# --------------------------------------------------------------------------- #
# Schema normalization + value coercion (port of parameters.rs).
# --------------------------------------------------------------------------- #
def _one_of(types: list) -> tuple:
    return types[0] if len(types) == 1 else ("oneof", types)


def _norm(schema) -> tuple | None:
    if not isinstance(schema, dict):
        return None
    if "type" in schema:
        return _norm_type_value(schema["type"], schema)
    composite = schema.get("anyOf")
    if composite is None:
        composite = schema.get("oneOf")
    if composite is not None:
        if isinstance(composite, list):
            types = [t for t in (_norm(x) for x in composite) if t is not None]
            if types:
                return _one_of(types)
        return _object_from(schema)
    if "enum" in schema:
        return ("string", )
    if "items" in schema:
        return _array_from(schema)
    if "properties" in schema or "additionalProperties" in schema:
        return _object_from(schema)
    return None


def _norm_type_value(type_value, schema) -> tuple | None:
    if isinstance(type_value, str):
        return _from_type_name(type_value, schema)
    if isinstance(type_value, list):
        types = [t for t in (_from_type_name(k, schema) for k in type_value if isinstance(k, str)) if t is not None]
        return _one_of(types) if types else None
    return None


def _from_type_name(kind: str, schema) -> tuple | None:
    k = kind.strip().lower()
    if k in ("string", "str", "text", "varchar", "char", "enum"):
        return ("string", )
    if k in ("integer", "int"):
        return ("integer", )
    if k in ("number", "float", "double"):
        return ("number", )
    if k in ("boolean", "bool", "binary"):
        return ("boolean", )
    if k in ("object", "dict", "map"):
        return _object_from(schema)
    if k in ("array", "arr", "list", "sequence"):
        return _array_from(schema)
    if k == "null":
        return ("null", )
    if (k.startswith("int") or k.startswith("uint") or k.startswith("long") or k.startswith("short")
            or k.startswith("unsigned")):
        return ("integer", )
    if k.startswith("num") or k.startswith("float"):
        return ("number", )
    if k.startswith("dict"):
        return _object_from(schema)
    if k.startswith("list"):
        return _array_from(schema)
    return None


def _object_from(schema) -> tuple:
    props: dict = {}
    raw_props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(raw_props, dict):
        for name, sub in raw_props.items():
            nt = _norm(sub)
            if nt is not None:
                props[name] = nt
    additional = None
    ap = schema.get("additionalProperties") if isinstance(schema, dict) else None
    if isinstance(ap, dict):
        additional = _norm(ap)
    return ("object", props, additional)


def _array_from(schema) -> tuple:
    items = None
    if isinstance(schema, dict):
        items = _norm(schema.get("items"))
    return ("array", items)


def _to_int(s: str):
    if re.fullmatch(r"[+-]?\d+", s.strip()):
        try:
            return int(s.strip())
        except ValueError:
            return _FAIL
    return _FAIL


def _to_number(s: str):
    t = s.strip()
    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except ValueError:
            return _FAIL
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", t):
        try:
            f = float(t)
            if math.isfinite(f):
                return f
        except ValueError:
            return _FAIL
    return _FAIL


def _to_bool(s: str):
    t = s.strip().lower()
    if t in ("true", "1"):
        return True
    if t in ("false", "0"):
        return False
    return _FAIL


def _insert_object_value(obj: dict, key: str, value) -> None:
    """Preserve duplicate keys as arrays (mirrors insert_object_value)."""
    if key in obj:
        existing = obj[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            obj[key] = [existing, value]
    else:
        obj[key] = value


def _convert(ntype, pinput):
    """convert_with_optional_schema: coerce one parameter input to a JSON value."""
    if pinput[0] == "text" and pinput[1].strip().lower() == "null":
        return None
    if ntype is not None:
        v = _try_convert(ntype, pinput)
        if v is not _FAIL:
            return v
    # Fallback: no schema, or conversion failed.
    if pinput[0] == "text":
        return pinput[1]
    return _elements_to_object(pinput[1], {}, None)


def _try_convert(ntype, pinput):
    kind = ntype[0]
    if pinput[0] == "text":
        s = pinput[1]
        if kind == "string":
            return s
        if kind == "integer":
            return _to_int(s)
        if kind == "number":
            return _to_number(s)
        if kind == "boolean":
            return _to_bool(s)
        if kind == "null":
            return None if s.strip().lower() == "null" else _FAIL
        if kind in ("object", "array"):
            if s == "":
                return {} if kind == "object" else []
            try:
                return json.loads(s)
            except (ValueError, json.JSONDecodeError):
                return _FAIL
        if kind == "oneof":
            for t in ntype[1]:
                v = _try_convert(t, pinput)
                if v is not _FAIL:
                    return v
            return _FAIL
        return _FAIL
    # elements
    els = pinput[1]
    if kind == "object":
        return _elements_to_object(els, ntype[1], ntype[2])
    if kind == "array":
        items = ntype[1]
        return [_convert(items, v) for (_, v) in els]
    if kind == "oneof":
        for t in ntype[1]:
            v = _try_convert(t, pinput)
            if v is not _FAIL:
                return v
        return _FAIL
    return _FAIL  # primitive from structured input -> fail


def _elements_to_object(els, properties: dict, additional) -> dict:
    obj: dict = {}
    for name, value in els:
        pt = properties.get(name, additional)
        _insert_object_value(obj, name, _convert(pt, value))
    return obj


def _push_mixed(elements: list, text: str) -> None:
    name = MIXED_TEXT_FIELD
    existing = {n for n, _ in elements}
    while name in existing:
        name = "$" + name
    elements.append((name, ("text", text)))


# --------------------------------------------------------------------------- #
# Recursive element parsing (port of minimax_m3.rs).
# --------------------------------------------------------------------------- #
def _open_tag(cur: _Cur) -> str:
    cur.consume(ELEMENT_START)
    name = cur.take_until(">")
    cur.consume(">")
    if not name.strip() or name.startswith("/"):
        raise _ParseError("bad open tag")
    return name


def _close_tag(cur: _Cur, name: str) -> None:
    cur.consume(ELEMENT_END_START)
    cur.consume(name)
    cur.consume(">")


def _close_tag_name(cur: _Cur) -> str:
    cur.consume(ELEMENT_END_START)
    name = cur.take_until(">").strip()
    cur.consume(">")
    if not name or name.startswith("/"):
        raise _ParseError("bad close tag")
    return name


def _element_body(cur: _Cur, closing_name: str):
    close_tag = ELEMENT_END_START + closing_name + ">"
    text_parts: list[str] = []
    elements: list = []
    while True:
        text_parts.append(cur.take_until(NS))
        if cur.startswith(close_tag):
            break
        if cur.startswith(ELEMENT_START):
            elements.append(_parameter_element(cur))
            continue
        # A bare namespace marker that is neither our close tag nor a child
        # open tag is malformed for this element body.
        raise _ParseError("unexpected namespace marker in element body")
    text = "".join(text_parts)
    if not elements:
        return ("text", text)
    if text.strip():
        _push_mixed(elements, text)
    return ("elements", elements)


def _parameter_element(cur: _Cur):
    name = _open_tag(cur)
    value = _element_body(cur, name)
    _close_tag(cur, name)
    return (name, value)


def _headless_element(cur: _Cur):
    """Recover a param whose opening tag was dropped: ``NS value NS </name>``."""
    cur.consume(NS)
    value = cur.take_until(ELEMENT_END_START)
    name = _close_tag_name(cur)
    return (name, ("text", value))


def _parse_invoke_params(body: str) -> list:
    cur = _Cur(body)
    params: list = []
    while True:
        cur.skip_ws()
        if cur.eof():
            break
        if cur.startswith(ELEMENT_START):
            params.append(_parameter_element(cur))
            continue
        if cur.startswith(NS):
            # Dropped opening tag (top level) -> headless recovery.
            params.append(_headless_element(cur))
            continue
        # Ordinary text at a parameter boundary ends this invoke: keep what we
        # parsed and drop the rest (mirrors upstream tolerance).
        break
    return params


# --------------------------------------------------------------------------- #
# Streaming helpers.
# --------------------------------------------------------------------------- #
def _hold_partial(text: str, marker: str) -> int:
    """Largest end index such that ``text[end:]`` is not a non-empty proper
    prefix of ``marker`` -- so we never emit a half namespace marker as a value
    fragment (it might be the start of the value's closing tag)."""
    n = len(text)
    max_hold = min(len(marker) - 1, n)
    for hold in range(max_hold, 0, -1):
        if marker.startswith(text[n - hold:]):
            return n - hold
    return n


def _json_escape_body(s: str) -> str:
    """JSON-escape *s* without the surrounding quotes. This mapping is
    prefix-stable: escape(a + b) == escape(a) + escape(b), which keeps the
    streamed argument fragments append-only."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def _scan_open_invoke_body(body_raw: str, props: dict):
    """Best-effort scan of a still-streaming invoke body.

    Returns ``(closed, open_param)`` where ``closed`` is the list of
    ``(name, coerced_value)`` for every fully-closed parameter, and
    ``open_param`` is either ``None`` or ``(name, partial_text)`` for a trailing
    *string* parameter whose closing tag has not arrived yet (only when the
    tool schema declares it a string and the partial value is plain text -- so
    coercion at close is guaranteed to match what we streamed)."""
    cur = _Cur(body_raw)
    closed: list = []
    open_param = None
    while True:
        cur.skip_ws()
        if cur.eof():
            break
        if cur.startswith(ELEMENT_START):
            gt = body_raw.find(">", cur.i + len(ELEMENT_START))
            if gt == -1:
                break  # open tag itself still streaming
            name = body_raw[cur.i + len(ELEMENT_START):gt]
            if not name.strip() or name.startswith("/"):
                break
            save = cur.i
            try:
                pname, pinput = _parameter_element(cur)
            except _ParseError:
                cur.i = save
                nm = name.strip()
                # Only stream a trailing param when the schema says it is a
                # string and the value seen so far is plain text (no nested
                # markers), so the final coerced value equals the streamed one.
                if props.get(nm) == ("string", ):
                    rest = body_raw[gt + 1:]
                    cut = _hold_partial(rest, NS)
                    if NS not in rest[:cut]:
                        open_param = (nm, rest[:cut])
                break
            else:
                closed.append((pname, _convert(props.get(pname), pinput)))
                continue
        if cur.startswith(NS):
            save = cur.i
            try:
                pname, pinput = _headless_element(cur)
            except _ParseError:
                cur.i = save
                break
            else:
                closed.append((pname, _convert(props.get(pname), pinput)))
                continue
        break
    return closed, open_param


def _build_partial_args(closed: list, open_param):
    """Build an append-only JSON *prefix* for the arguments seen so far, plus
    the equivalent dict (used to compute the end-of-stream flush).

    The returned string is deliberately unclosed (no final ``}``, and an open
    string value has no closing quote) so successive calls only ever *extend*
    it. The dict form is fully closed, so ``json.dumps(dict)`` provides the
    remainder to flush on truncation."""
    d: dict = {}
    parts: list[str] = ["{"]
    first = True
    for name, val in closed:
        if not first:
            parts.append(", ")
        first = False
        parts.append(json.dumps(name, ensure_ascii=False))
        parts.append(": ")
        parts.append(json.dumps(val, ensure_ascii=False))
        d[name] = val
    if open_param is not None:
        name, partial = open_param
        if not first:
            parts.append(", ")
        first = False
        parts.append(json.dumps(name, ensure_ascii=False))
        parts.append(': "')
        parts.append(_json_escape_body(partial))
        d[name] = partial  # dict form is complete; json.dumps closes the quote
    return "".join(parts), d


class MinimaxM3PyToolParser(ToolParser):
    """Python re-implementation of the MiniMax-M3 tool-call grammar with
    incremental, truncation-safe streaming."""

    # Our output is XML, not guided-decoding JSON. Keep this False so the
    # serving layer routes auto tool_choice through this parser (validated).
    supports_required_and_named = False

    # ---- shared helpers ---------------------------------------------------- #
    def _tool_props(self, func_name: str, request) -> dict:
        if not hasattr(self, "_props_cache"):
            self._props_cache: dict = {}
        cache = self._props_cache
        if func_name in cache:
            return cache[func_name]
        props: dict = {}
        try:
            for tool in (getattr(request, "tools", None) or []):
                fn = getattr(tool, "function", None)
                if getattr(fn, "name", None) == func_name:
                    schema = getattr(fn, "parameters", None) or {}
                    for k, v in (schema.get("properties") or {}).items():
                        nt = _norm(v)
                        if nt is not None:
                            props[k] = nt
                    break
        except Exception:
            props = {}
        cache[func_name] = props
        return props

    def _parse_one_invoke(self, inv_block: str, request) -> ToolCall:
        """Parse a complete ``<invoke ...>...</invoke>`` block into a ToolCall."""
        # Tolerate a dropped leading "<" on the invoke open tag (MiniMax-M3
        # quirk); the block may start with INVOKE_START or INVOKE_START_NOLT.
        opener_len = (len(INVOKE_START) if inv_block.startswith(INVOKE_START) else
                      len(INVOKE_START_NOLT) if inv_block.startswith(INVOKE_START_NOLT) else len(INVOKE_START))
        # Search past the opener: the namespace marker itself contains a ">".
        header_end = inv_block.find(">", opener_len)
        if header_end == -1:
            raise _ParseError("invoke header not closed")
        header = inv_block[opener_len:header_end]
        m = _INVOKE_NAME_RE.search(header)
        if not m:
            raise _ParseError("invoke name not found")
        name = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if not name:
            raise _ParseError("empty invoke name")
        body = inv_block[header_end + 1:len(inv_block) - len(INVOKE_END)]
        params = _parse_invoke_params(body)
        props = self._tool_props(name, request)
        args: dict = {}
        for pname, pinput in params:
            # Top-level duplicate keys overwrite (mirrors convert_params).
            args[pname] = _convert(props.get(pname), pinput)
        return ToolCall(
            id=make_tool_call_id(),
            type="function",
            function=FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        )

    def _parse_block(self, text: str, start: int, request) -> list[ToolCall]:
        """Parse completed invokes from a tool-call block; tolerate truncation."""
        cur = _Cur(text)
        cur.i = start
        cur.consume(TOOL_CALL_START)
        calls: list[ToolCall] = []
        while True:
            cur.skip_ws()
            if cur.eof() or cur.startswith(TOOL_CALL_END):
                break
            if not (cur.startswith(INVOKE_START) or cur.startswith(INVOKE_START_NOLT)):
                break  # stray text / malformed boundary -> stop
            inv_end = text.find(INVOKE_END, cur.i)
            if inv_end == -1:
                break  # truncated invoke -> keep completed calls
            inv_block = text[cur.i:inv_end + len(INVOKE_END)]
            cur.i = inv_end + len(INVOKE_END)
            try:
                calls.append(self._parse_one_invoke(inv_block, request))
            except _ParseError:
                break
        return calls

    # ---- non-streaming ----------------------------------------------------- #
    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        try:
            if not model_output or TOOL_CALL_START not in model_output:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output or None,
                )
            start = model_output.find(TOOL_CALL_START)
            content = model_output[:start].strip() or None
            calls = self._parse_block(model_output, start, request)
            if not calls:
                # Preserve the generated output when no valid call can be
                # recovered. Returning only the prefix would silently discard
                # malformed or truncated tool-call text.
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
            return ExtractedToolCallInformation(tools_called=True, tool_calls=calls, content=content)
        except Exception:
            logger.exception("MiniMax-M3 tool parse failed; degrading to plain content")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output or None)

    # ---- streaming --------------------------------------------------------- #
    def _ensure_stream_state(self) -> None:
        if not hasattr(self, "_m3_pos"):
            self._m3_pos = 0
            self._m3_mode = "text"  # text -> tool -> done/fallback
            self._m3_index = 0
            self._open: dict | None = None
        # These are provided by the real vLLM ToolParser base __init__, but we
        # (re)ensure them so the plugin is robust to stubbed/base variants and
        # so vLLM's finalize_generation() flush works.
        if getattr(self, "prev_tool_call_arr", None) is None:
            self.prev_tool_call_arr = []
        if not hasattr(self, "streamed_args_for_tool"):
            self.streamed_args_for_tool = []
        if not hasattr(self, "current_tool_id"):
            self.current_tool_id = -1

    def get_remaining_unstreamed_args(self) -> str:
        """Return arguments parsed but not yet streamed for the last tool call.

        Overrides the base implementation with identical semantics so vLLM's
        ``finalize_generation`` auto-closes a truncated argument on the final
        chunk (``finished=True``)."""
        if not getattr(self, "prev_tool_call_arr", None):
            return ""
        index = len(self.prev_tool_call_arr) - 1
        args = self.prev_tool_call_arr[index].get("arguments", {})
        expected = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        actual = (self.streamed_args_for_tool[index] if index < len(self.streamed_args_for_tool) else "")
        if expected.startswith(actual):
            return expected[len(actual):]
        return ""

    def _safe_text_end(self, text: str, pos: int) -> int:
        """Largest end >= pos such that text[pos:end] cannot be a partial
        TOOL_CALL_START prefix (so we never leak a half marker)."""
        n = len(text)
        max_hold = min(len(TOOL_CALL_START) - 1, n - pos)
        for hold in range(max_hold, 0, -1):
            if TOOL_CALL_START.startswith(text[n - hold:]):
                return n - hold
        return n

    def _emit_args(self, tool_deltas: list, index: int, diff: str) -> None:
        tool_deltas.append(
            DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
            ))

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        fallback_pos = getattr(self, "_m3_pos", len(previous_text))
        try:
            self._ensure_stream_state()
            fallback_pos = self._m3_pos

            if self._m3_mode == "fallback":
                fallback = current_text[self._m3_pos:]
                self._m3_pos = len(current_text)
                return DeltaMessage(content=fallback) if fallback else None

            if self._m3_mode == "done":
                return None

            content_out = None
            tool_deltas: list[DeltaToolCall] = []

            # ---- TEXT phase: stream content until the tool-call block opens.
            if self._m3_mode == "text":
                start = current_text.find(TOOL_CALL_START, self._m3_pos)
                if start == -1:
                    safe_end = self._safe_text_end(current_text, self._m3_pos)
                    if safe_end > self._m3_pos:
                        content_out = current_text[self._m3_pos:safe_end]
                        self._m3_pos = safe_end
                    return (DeltaMessage(content=content_out) if content_out else None)
                if start > self._m3_pos:
                    content_out = current_text[self._m3_pos:start]
                self._m3_pos = start + len(TOOL_CALL_START)
                self._m3_mode = "tool"

            # ---- TOOL phase: incremental, truncation-safe invoke streaming.
            if self._m3_mode == "tool":
                while True:
                    if self._open is None:
                        end_pos = current_text.find(TOOL_CALL_END, self._m3_pos)
                        inv_pos, inv_open_len = _find_invoke_open(current_text, self._m3_pos)
                        if end_pos != -1 and (inv_pos == -1 or end_pos < inv_pos):
                            self._m3_pos = end_pos + len(TOOL_CALL_END)
                            self._m3_mode = "done"
                            break
                        if inv_pos == -1:
                            break  # nothing to open yet
                        header_end = current_text.find(">", inv_pos + inv_open_len)
                        if header_end == -1:
                            break  # invoke header still streaming
                        header = current_text[inv_pos + inv_open_len:header_end]
                        m = _INVOKE_NAME_RE.search(header)
                        name = ((m.group(1) or m.group(2) or m.group(3) or "").strip() if m else "")
                        if not name:
                            # Malformed header: skip this invoke if it closes,
                            # else wait. Avoids emitting a nameless tool call.
                            inv_end0 = current_text.find(INVOKE_END, inv_pos)
                            if inv_end0 == -1:
                                break
                            self._m3_pos = inv_end0 + len(INVOKE_END)
                            continue
                        tc_id = make_tool_call_id()
                        self._open = {
                            "name": name,
                            "id": tc_id,
                            "index": self._m3_index,
                            "body_start": header_end + 1,
                            "emitted": "",
                        }
                        self.prev_tool_call_arr.append({"name": name, "arguments": {}})
                        self.streamed_args_for_tool.append("")
                        self.current_tool_id = self._m3_index
                        # Emit the NAME immediately -- the core anti-stall fix.
                        tool_deltas.append(
                            DeltaToolCall(
                                index=self._m3_index,
                                id=tc_id,
                                type="function",
                                function=DeltaFunctionCall(name=name, arguments="").model_dump(exclude_none=True),
                            ))

                    op = self._open
                    idx = op["index"]
                    props = self._tool_props(op["name"], request)
                    inv_end = current_text.find(INVOKE_END, op["body_start"])

                    if inv_end != -1:
                        # Invoke complete: emit authoritative final arguments.
                        body = current_text[op["body_start"]:inv_end]
                        try:
                            params = _parse_invoke_params(body)
                            closed_full = [(pn, _convert(props.get(pn), pin)) for pn, pin in params]
                            full_json, full_d = _build_partial_args(closed_full, None)
                            full_json += "}"
                        except _ParseError:
                            full_json, full_d = self._close_partial(op)
                        if full_json.startswith(op["emitted"]) and len(full_json) > len(op["emitted"]):
                            self._emit_args(tool_deltas, idx, full_json[len(op["emitted"]):])
                        self.prev_tool_call_arr[idx]["arguments"] = full_d
                        self.streamed_args_for_tool[idx] = full_json
                        self._m3_pos = inv_end + len(INVOKE_END)
                        self._m3_index += 1
                        self._open = None
                        continue  # look for the next invoke / block end

                    # Invoke still streaming: emit any newly-parseable prefix.
                    body_raw = current_text[op["body_start"]:]
                    closed, open_param = _scan_open_invoke_body(body_raw, props)
                    target, d = _build_partial_args(closed, open_param)
                    op["open_string"] = open_param is not None
                    if target.startswith(op["emitted"]) and len(target) > len(op["emitted"]):
                        self._emit_args(tool_deltas, idx, target[len(op["emitted"]):])
                        op["emitted"] = target
                        self.streamed_args_for_tool[idx] = target
                    # Keep the fully-closed dict so finalize_generation() can
                    # auto-close a truncated argument on the final chunk.
                    self.prev_tool_call_arr[idx]["arguments"] = d
                    break  # wait for more text

            if content_out and tool_deltas:
                return DeltaMessage(content=content_out, tool_calls=tool_deltas)
            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            if content_out:
                return DeltaMessage(content=content_out)
            return None
        except Exception:
            logger.exception("MiniMax-M3 streaming parse failed; emitting remaining text")
            # No deltas accumulated during this call have been returned yet.
            # Emit the unconsumed cumulative-text suffix as plain content so an
            # unexpected parser error cannot silently stall the client.
            fallback = current_text[fallback_pos:]
            self._m3_pos = len(current_text)
            self._m3_mode = "fallback"
            return DeltaMessage(content=fallback) if fallback else None

    def _close_partial(self, op: dict) -> tuple[str, dict]:
        """Best-effort close of the emitted args prefix into valid JSON, used
        only for the rare complete-but-unparseable invoke."""
        suffix = ""
        if op.get("open_string"):
            suffix += '"'
        suffix += "}"
        full_json = (op["emitted"] or "{") + suffix
        try:
            full_d = json.loads(full_json)
        except (ValueError, json.JSONDecodeError):
            full_json, full_d = "{}", {}
        return full_json, full_d


ToolParserManager.register_module(name="minimax_m3_py", module=MinimaxM3PyToolParser)
