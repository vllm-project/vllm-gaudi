# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.tool_parsers import ToolParserManager

from vllm_gaudi import register_tool_parsers
from vllm_gaudi.entrypoints.openai.tool_parsers.minimax_m3 import (
    ELEMENT_END_START,
    ELEMENT_START,
    INVOKE_END,
    INVOKE_START,
    INVOKE_START_NOLT,
    TOOL_CALL_END,
    TOOL_CALL_START,
    MinimaxM3PyToolParser,
)


def _request():
    properties = {
        "path": {
            "type": "string"
        },
        "count": {
            "type": "integer"
        },
        "enabled": {
            "type": "boolean"
        },
    }
    function = SimpleNamespace(name="write_file", parameters={"properties": properties})
    return SimpleNamespace(tools=[SimpleNamespace(function=function)])


def _element(name: str, value: str) -> str:
    return f"{ELEMENT_START}{name}>{value}{ELEMENT_END_START}{name}>"


def _invoke(path: str, count: int = 2, enabled: bool = True) -> str:
    body = (_element("path", path) + _element("count", str(count)) + _element("enabled", str(enabled).lower()))
    return f'{INVOKE_START} name="write_file">{body}{INVOKE_END}'


def _invoke_dropped_lt(path: str, count: int = 2, enabled: bool = True) -> str:
    # MiniMax-M3 sometimes drops the leading "<" of the invoke open tag.
    body = (_element("path", path) + _element("count", str(count)) + _element("enabled", str(enabled).lower()))
    return f'{INVOKE_START_NOLT} name="write_file">{body}{INVOKE_END}'


def _function_field(function, name: str):
    if isinstance(function, dict):
        return function.get(name)
    return getattr(function, name, None)


def _stream(parser: MinimaxM3PyToolParser, text: str, request):
    content: list[str] = []
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    previous = ""

    for end in range(1, len(text) + 1):
        current = text[:end]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=current[len(previous):],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous = current
        if delta is None:
            continue
        if delta.content:
            content.append(delta.content)
        for tool_call in delta.tool_calls or []:
            function = tool_call.function
            name = _function_field(function, "name")
            if name is not None:
                names[tool_call.index] = name
            args = _function_field(function, "arguments") or ""
            arguments[tool_call.index] = arguments.get(tool_call.index, "") + args

    return "".join(content), names, arguments


def test_register_tool_parser():
    register_tool_parsers()
    assert ToolParserManager.get_tool_parser("minimax_m3_py") is MinimaxM3PyToolParser


def test_extract_tool_calls():
    text = f"before{TOOL_CALL_START}{_invoke('/tmp/out')}{TOOL_CALL_END}"

    result = MinimaxM3PyToolParser(None).extract_tool_calls(text, _request())

    assert result.tools_called
    assert result.content == "before"
    assert len(result.tool_calls) == 1
    function = result.tool_calls[0].function
    assert function.name == "write_file"
    assert json.loads(function.arguments) == {
        "path": "/tmp/out",
        "count": 2,
        "enabled": True,
    }


def test_extract_multiple_tool_calls():
    text = f"{TOOL_CALL_START}{_invoke('first')}{_invoke('second', count=3)}{TOOL_CALL_END}"

    result = MinimaxM3PyToolParser(None).extract_tool_calls(text, _request())

    assert result.tools_called
    assert [json.loads(call.function.arguments)["path"] for call in result.tool_calls] == ["first", "second"]


def test_extract_tool_calls_dropped_invoke_lt():
    text = f"before{TOOL_CALL_START}{_invoke_dropped_lt('/tmp/out')}{TOOL_CALL_END}"

    result = MinimaxM3PyToolParser(None).extract_tool_calls(text, _request())

    assert result.tools_called
    assert result.content == "before"
    assert len(result.tool_calls) == 1
    function = result.tool_calls[0].function
    assert function.name == "write_file"
    assert json.loads(function.arguments) == {
        "path": "/tmp/out",
        "count": 2,
        "enabled": True,
    }


def test_streaming_dropped_invoke_lt():
    text = f"before{TOOL_CALL_START}{_invoke_dropped_lt('/tmp/out')}{TOOL_CALL_END}"
    parser = MinimaxM3PyToolParser(None)

    content, names, arguments = _stream(parser, text, _request())

    assert content == "before"
    assert names == {0: "write_file"}
    assert json.loads(arguments[0]) == {
        "path": "/tmp/out",
        "count": 2,
        "enabled": True,
    }


def test_non_streaming_malformed_call_preserves_output():
    text = f"before{TOOL_CALL_START}{INVOKE_START} name=\"write_file\">truncated"

    result = MinimaxM3PyToolParser(None).extract_tool_calls(text, _request())

    assert not result.tools_called
    assert result.content == text


def test_streaming_matches_non_streaming_with_split_markers():
    text = f"before{TOOL_CALL_START}{_invoke('/tmp/out')}{TOOL_CALL_END}"
    parser = MinimaxM3PyToolParser(None)

    content, names, arguments = _stream(parser, text, _request())

    assert content == "before"
    assert names == {0: "write_file"}
    assert json.loads(arguments[0]) == {
        "path": "/tmp/out",
        "count": 2,
        "enabled": True,
    }


def test_streaming_truncated_string_can_be_finalized():
    text = f"{TOOL_CALL_START}{INVOKE_START} name=\"write_file\">{ELEMENT_START}path>partial content"
    parser = MinimaxM3PyToolParser(None)

    _, names, arguments = _stream(parser, text, _request())
    arguments[0] += parser.get_remaining_unstreamed_args()

    assert names == {0: "write_file"}
    assert json.loads(arguments[0]) == {"path": "partial content"}


def test_streaming_error_emits_unconsumed_text(monkeypatch):
    text = f"{TOOL_CALL_START}{_invoke('/tmp/out')}{TOOL_CALL_END}"
    parser = MinimaxM3PyToolParser(None)

    def fail(*args, **kwargs):
        raise RuntimeError("injected parser failure")

    monkeypatch.setattr(parser, "_tool_props", fail)
    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=_request(),
    )

    assert delta is not None
    assert delta.content == text

    continued = parser.extract_tool_calls_streaming(
        previous_text=text,
        current_text=text + " remaining",
        delta_text=" remaining",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=_request(),
    )
    assert continued is not None
    assert continued.content == " remaining"
