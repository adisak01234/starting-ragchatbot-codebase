import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults
from search_tools import CourseSearchTool


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_search_results(docs, course_title="Test Course", lesson_num=1):
    """Build a successful SearchResults from a list of document strings."""
    metadata = [{"course_title": course_title, "lesson_number": lesson_num} for _ in docs]
    distances = [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metadata, distances=distances)


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "http://example.com/lesson"
    store.get_course_link.return_value = "http://example.com/course"
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_text_on_success(tool, mock_store):
    """Happy path: result contains course/lesson header and document text."""
    mock_store.search.return_value = make_search_results(
        ["Some lesson content"], "My Course", 2
    )
    result = tool.execute(query="test query")

    assert "My Course" in result
    assert "Lesson 2" in result
    assert "Some lesson content" in result


def test_execute_returns_search_error_when_search_has_error(tool, mock_store):
    """SearchResults.error is returned verbatim as the tool output."""
    mock_store.search.return_value = SearchResults.empty("Search error: n_results=0")
    result = tool.execute(query="test query")

    assert result == "Search error: n_results=0"


def test_execute_returns_no_content_message_when_empty(tool, mock_store):
    """Empty SearchResults (no error) → 'No relevant content found' message."""
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    result = tool.execute(query="test query")

    assert "No relevant content found" in result


def test_execute_populates_last_sources_after_success(tool, mock_store):
    """last_sources has correct label and url after a successful search."""
    mock_store.search.return_value = make_search_results(["content"], "Course A", 3)
    mock_store.get_lesson_link.return_value = "http://example.com/lesson3"

    tool.execute(query="test query")

    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["label"] == "Course A - Lesson 3"
    assert tool.last_sources[0]["url"] == "http://example.com/lesson3"


def test_execute_does_not_populate_sources_on_error(tool, mock_store):
    """last_sources stays empty when the search returns an error."""
    mock_store.search.return_value = SearchResults.empty("Search error: failure")

    tool.execute(query="test query")

    assert tool.last_sources == []


def test_execute_passes_course_name_filter_to_store(tool, mock_store):
    """store.search is called with the course_name keyword argument."""
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="test", course_name="My Course")

    mock_store.search.assert_called_once_with(
        query="test", course_name="My Course", lesson_number=None
    )


def test_execute_passes_lesson_number_filter_to_store(tool, mock_store):
    """store.search is called with the lesson_number keyword argument."""
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="test", lesson_number=5)

    mock_store.search.assert_called_once_with(
        query="test", course_name=None, lesson_number=5
    )


def test_vector_store_search_returns_error_when_limit_is_zero():
    """
    THE BUG: VectorStore.search() with max_results=0 passes n_results=0 to
    ChromaDB, which raises ValueError.  The exception is caught and the method
    returns SearchResults with a non-None error field.

    This test PASSES now (confirming the bug exists) and continues to pass after
    the fix because the error-handling path still works correctly.
    """
    from vector_store import VectorStore

    # Build a plain MagicMock (no spec) so instance attributes like
    # course_content are accessible without restriction.
    mock_self = MagicMock()
    mock_self.max_results = 0
    mock_self._build_filter.return_value = None
    mock_self.course_content.query.side_effect = ValueError(
        "Number of requested results 0 is less than the smallest sample size 1"
    )

    result = VectorStore.search(mock_self, query="test query")

    assert result.error is not None
    assert "Search error" in result.error
