import pytest
from app.services.tool_service import ToolService
from app.tools.calculator import CalculatorTool
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

@pytest.fixture
def tool_service():
    service = ToolService()
    service.register_tool(CalculatorTool())
    return service

def test_tool_registration_and_listing(tool_service):
    tools = tool_service.list_tools_with_schemas()
    assert any(t['name'] == 'calculator' for t in tools)
    calc_tool = next(t for t in tools if t['name'] == 'calculator')
    assert 'parameters_schema' in calc_tool
    assert 'output_schema' in calc_tool

@pytest.mark.asyncio
async def test_tool_argument_validation_success(tool_service):
    output = await tool_service.execute('calculator', {'a': 2, 'b': 3})
    assert output['success']
    assert output['output']['result'] == 5

@pytest.mark.asyncio
async def test_tool_argument_validation_failure(tool_service):
    output = await tool_service.execute('calculator', {'a': 2})
    assert not output['success']
    assert 'validation' in output['error'].lower()

@pytest.mark.asyncio
async def test_tool_not_found(tool_service):
    output = await tool_service.execute('not_a_tool', {'a': 1})
    assert not output['success']
    assert 'not found' in output['error'].lower() 