import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text="Response text", stop_reason="end_turn"):
    """Build a mock Anthropic response that ends with plain text."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp.content = [block]
    return resp


def _tool_use_response(tool_name="search_course_content", tool_id="tu_001", inputs=None):
    """Build a mock Anthropic response that requests a tool call."""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = inputs or {"query": "test"}
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_api_client():
    """Patch anthropic.Anthropic so no real HTTP calls are made."""
    with patch("ai_generator.anthropic.Anthropic") as MockClass:
        client = MagicMock()
        MockClass.return_value = client
        yield client


@pytest.fixture
def generator(mock_api_client):
    return AIGenerator(api_key="test-key", model="test-model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_response_returns_text_directly_on_end_turn(generator, mock_api_client):
    """stop_reason='end_turn' → returns first content block's text immediately."""
    mock_api_client.messages.create.return_value = _text_response("Hello world")

    result = generator.generate_response(query="What is Python?")

    assert result == "Hello world"


def test_generate_response_executes_tool_on_tool_use(generator, mock_api_client):
    """stop_reason='tool_use' → tool_manager.execute_tool is called with correct args."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "t1", {"query": "python"}),
        _text_response("Python is a language"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results here"

    generator.generate_response(
        query="Tell me about Python",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="python"
    )


def test_generate_response_makes_two_api_calls_when_tool_used(generator, mock_api_client):
    """Two API calls are made: initial + follow-up after tool execution."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _text_response("Final answer"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    generator.generate_response(query="test", tool_manager=tool_manager)

    assert mock_api_client.messages.create.call_count == 2


def test_handle_tool_execution_includes_tool_result_in_second_call(generator, mock_api_client):
    """Second API call's messages contain a tool_result block with the tool output."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tool_abc", {"query": "test"}),
        _text_response("Done"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "my tool result"

    generator.generate_response(query="test", tool_manager=tool_manager)

    second_call_kwargs = mock_api_client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]

    # Walk the message list to find the tool_result block
    tool_result_found = False
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        assert block["content"] == "my tool result"
                        assert block["tool_use_id"] == "tool_abc"
                        tool_result_found = True

    assert tool_result_found, "tool_result block not found in second API call messages"


def test_synthesis_call_excludes_tools(generator, mock_api_client):
    """Synthesis (3rd) call does NOT include 'tools'; 2nd round call does."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _tool_use_response(),
        _text_response("Done"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]
    generator.generate_response(query="test", tools=tools, tool_manager=tool_manager)

    # 2nd call (round 1) should still include tools
    second_call_kwargs = mock_api_client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs

    # 3rd call (synthesis) must NOT include tools
    third_call_kwargs = mock_api_client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs


def test_generate_response_returns_final_text_after_tool_execution(generator, mock_api_client):
    """Text from the second API call is returned as the final response."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _text_response("The final answer is 42"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    result = generator.generate_response(query="test", tool_manager=tool_manager)

    assert result == "The final answer is 42"


# ---------------------------------------------------------------------------
# New tests: two sequential tool rounds
# ---------------------------------------------------------------------------

def test_two_sequential_tool_calls_makes_three_api_calls(generator, mock_api_client):
    """Two tool rounds plus synthesis = 3 API calls total."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _tool_use_response(),
        _text_response("Final answer"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_api_client.messages.create.call_count == 3


def test_two_sequential_tool_calls_executes_both_tools(generator, mock_api_client):
    """Both tool calls are executed when two rounds occur."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "t1", {"query": "first"}),
        _tool_use_response("search_course_content", "t2", {"query": "second"}),
        _text_response("Final"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert tool_manager.execute_tool.call_count == 2


def test_two_sequential_tool_calls_returns_synthesis_text(generator, mock_api_client):
    """Text from the synthesis (3rd) call is returned as the final response."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _tool_use_response(),
        _text_response("Synthesized answer"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    result = generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Synthesized answer"


def test_second_round_api_call_includes_tools(generator, mock_api_client):
    """Second API call (round 1) includes tools so Claude can search again."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _text_response("Done"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]
    generator.generate_response(query="test", tools=tools, tool_manager=tool_manager)

    second_call_kwargs = mock_api_client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs


def test_max_two_rounds_enforced(generator, mock_api_client):
    """After 2 tool rounds, synthesis is called; no StopIteration raised."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _tool_use_response(),
        _text_response("Done"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_api_client.messages.create.call_count == 3


def test_tool_error_does_not_crash_response(generator, mock_api_client):
    """A tool execution error is caught; response is still returned."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response(),
        _text_response("Recovered response"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = Exception("search failed")

    result = generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Recovered response"
    assert mock_api_client.messages.create.call_count == 2


def test_tool_error_string_passed_to_claude(generator, mock_api_client):
    """When a tool errors, the error message is passed as tool_result content."""
    mock_api_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "err_tool", {"query": "test"}),
        _text_response("Done"),
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = Exception("search failed")

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_call_kwargs = mock_api_client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]

    error_found = False
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        assert block["content"].startswith("Tool error:")
                        error_found = True

    assert error_found, "Tool error message not found in second API call messages"
