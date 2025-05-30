import pytest
from app.nodes.tool_call_utils import detect_tool_call, format_tool_output

def test_detect_tool_call_valid():
    response = '{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}'
    tool_name, tool_args = detect_tool_call(response)
    assert tool_name == 'calculator'
    assert tool_args == {'a': 2, 'b': 3}

def test_detect_tool_call_invalid_json():
    response = 'not a json'
    tool_name, tool_args = detect_tool_call(response)
    assert tool_name is None
    assert tool_args is None

def test_detect_tool_call_no_tool_call():
    response = '{"some_other_key": 123}'
    tool_name, tool_args = detect_tool_call(response)
    assert tool_name is None
    assert tool_args is None

def test_format_tool_output():
    output = {'result': 5}
    formatted = format_tool_output('calculator', output)
    assert formatted.startswith("Tool 'calculator' output: ")
    assert '"result": 5' in formatted or "'result': 5" in formatted 