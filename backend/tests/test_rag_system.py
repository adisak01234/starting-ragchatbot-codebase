import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "test-model"
    cfg.CHROMA_PATH = "/tmp/test_chroma"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return cfg


@pytest.fixture
def rag_system(mock_config):
    """
    RAGSystem with VectorStore, AIGenerator, and DocumentProcessor all mocked.
    The real SessionManager and ToolManager (+ tools) are used.
    """
    with (
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator"),
        patch("rag_system.DocumentProcessor"),
    ):
        from rag_system import RAGSystem

        system = RAGSystem(mock_config)
        # Make ai_generator.generate_response return a plain string by default
        system.ai_generator.generate_response.return_value = "Default response"
        yield system


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_returns_response_and_sources_tuple(rag_system):
    """query() must return a 2-tuple of (str, list)."""
    result = rag_system.query("What is Python?")

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_query_sources_come_from_tool_manager(rag_system):
    """Sources returned by query() are taken from tool_manager.get_last_sources()."""
    expected = [{"label": "Course A - Lesson 1", "url": "http://example.com"}]
    rag_system.tool_manager.get_last_sources = MagicMock(return_value=expected)
    rag_system.ai_generator.generate_response.return_value = "Some response"

    _, sources = rag_system.query("test")

    assert sources == expected


def test_query_resets_sources_after_retrieval(rag_system):
    """tool_manager.reset_sources() is called after sources are retrieved."""
    rag_system.tool_manager.reset_sources = MagicMock()
    rag_system.ai_generator.generate_response.return_value = "response"

    rag_system.query("test")

    rag_system.tool_manager.reset_sources.assert_called_once()


def test_query_updates_session_history(rag_system):
    """session_manager.add_exchange() is called with the query and AI response."""
    rag_system.session_manager = MagicMock()
    rag_system.ai_generator.generate_response.return_value = "AI answer"

    rag_system.query("user question", session_id="session_1")

    rag_system.session_manager.add_exchange.assert_called_once_with(
        "session_1", "user question", "AI answer"
    )


def test_query_pipeline_with_zero_max_results_search_error():
    """
    Full pipeline bug test.

    With config.MAX_RESULTS=0 (the bug), VectorStore.search() passes n_results=0
    to ChromaDB, which raises ValueError.  The exception is caught and converted
    to SearchResults.empty("Search error: ..."), so CourseSearchTool.execute()
    returns an error string.  No sources are populated.

    FAILS before fix  (real_config.MAX_RESULTS == 0 → sources list is empty).
    PASSES after fix  (real_config.MAX_RESULTS == 5 → search succeeds → sources populated).
    """
    from config import config as real_config
    from vector_store import VectorStore as RealVS

    # ChromaDB collection: raises for n_results=0, returns data for n_results >= 1
    mock_collection = MagicMock()

    def query_side_effect(*args, **kwargs):
        if kwargs.get("n_results", 1) == 0:
            raise ValueError(
                "Number of requested results 0 is less than the smallest sample size 1"
            )
        return {
            "documents": [["Python is a versatile programming language."]],
            "metadatas": [
                [{"course_title": "Python Fundamentals", "lesson_number": 1}]
            ],
            "distances": [[0.1]],
        }

    mock_collection.query.side_effect = query_side_effect

    # VectorStore-like object that uses the REAL search() but with our mock ChromaDB
    vs = MagicMock()
    vs.max_results = real_config.MAX_RESULTS  # 0 now (bug); 5 after fix
    vs.course_content = mock_collection
    vs._build_filter.return_value = None
    vs.get_existing_course_titles.return_value = []
    vs.get_lesson_link.return_value = "http://example.com/lesson"
    vs.get_course_link.return_value = "http://example.com/course"
    vs.search = lambda query, course_name=None, lesson_number=None, limit=None: (
        RealVS.search(vs, query, course_name, lesson_number, limit)
    )

    # Mock Anthropic API: first call → tool_use, second call → end_turn text
    mock_api_client = MagicMock()

    tool_resp = MagicMock()
    tool_resp.stop_reason = "tool_use"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tu_001"
    tool_block.input = {"query": "python"}
    tool_resp.content = [tool_block]

    final_resp = MagicMock()
    final_resp.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Python is a versatile language."
    final_resp.content = [text_block]

    mock_api_client.messages.create.side_effect = [tool_resp, final_resp]

    with (
        patch("rag_system.VectorStore", return_value=vs),
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.SessionManager"),
        patch("ai_generator.anthropic.Anthropic", return_value=mock_api_client),
    ):
        from rag_system import RAGSystem

        system = RAGSystem(real_config)
        _, sources = system.query("What is Python?")

    # With MAX_RESULTS=0 (bug):  search fails → no sources → assertion FAILS
    # With MAX_RESULTS=5 (fix):  search works → sources populated → assertion PASSES
    assert len(sources) > 0, (
        "Expected sources to be populated after a successful search. "
        "Fix: set MAX_RESULTS = 5 in backend/config.py (currently 0)."
    )
