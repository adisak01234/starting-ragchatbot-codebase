"""
Tests for the FastAPI endpoints defined in app.py.

A lightweight test app is created in the `api_client` fixture (conftest.py)
that mirrors the real routes without mounting static files, so these tests
work out-of-the-box without a built frontend directory.
"""

import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_returns_200_with_answer(api_client, mock_rag_system):
    """Successful query returns 200 and the answer from the RAG system."""
    mock_rag_system.query.return_value = ("Python is a language.", [])

    response = api_client.post("/api/query", json={"query": "What is Python?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Python is a language."


def test_query_response_contains_session_id(api_client, mock_rag_system):
    """Response body always includes a session_id field."""
    mock_rag_system.query.return_value = ("Answer", [])

    response = api_client.post("/api/query", json={"query": "test"})

    assert "session_id" in response.json()


def test_query_creates_session_when_none_provided(api_client, mock_rag_system):
    """When session_id is omitted, a new one is created via session_manager."""
    mock_rag_system.session_manager.create_session.return_value = "new-session-abc"
    mock_rag_system.query.return_value = ("Answer", [])

    response = api_client.post("/api/query", json={"query": "test"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "new-session-abc"
    mock_rag_system.session_manager.create_session.assert_called_once()


def test_query_uses_provided_session_id(api_client, mock_rag_system):
    """When session_id is supplied, it is forwarded to RAG and no new session is created."""
    mock_rag_system.query.return_value = ("Answer", [])

    response = api_client.post(
        "/api/query", json={"query": "test", "session_id": "existing-session"}
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "existing-session"
    mock_rag_system.session_manager.create_session.assert_not_called()


def test_query_passes_query_to_rag_system(api_client, mock_rag_system):
    """The raw query string is forwarded to rag_system.query()."""
    mock_rag_system.query.return_value = ("Answer", [])

    api_client.post("/api/query", json={"query": "Tell me about ML"})

    call_args = mock_rag_system.query.call_args
    assert call_args[0][0] == "Tell me about ML"


def test_query_includes_sources_in_response(api_client, mock_rag_system):
    """Sources returned by the RAG system are included in the response."""
    sources = [{"label": "Python Basics - Lesson 1", "url": "http://example.com/l1"}]
    mock_rag_system.query.return_value = ("Answer with sources", sources)

    response = api_client.post("/api/query", json={"query": "test"})

    assert response.status_code == 200
    assert response.json()["sources"] == sources


def test_query_returns_empty_sources_list_when_no_sources(api_client, mock_rag_system):
    """A query with no matching sources returns an empty sources list, not null."""
    mock_rag_system.query.return_value = ("Direct answer", [])

    response = api_client.post("/api/query", json={"query": "test"})

    assert response.json()["sources"] == []


def test_query_returns_422_when_query_field_missing(api_client):
    """Omitting the required 'query' field results in a 422 Unprocessable Entity."""
    response = api_client.post("/api/query", json={})

    assert response.status_code == 422


def test_query_returns_500_on_rag_exception(api_client, mock_rag_system):
    """An unhandled exception inside the RAG system yields a 500 response."""
    mock_rag_system.query.side_effect = Exception("ChromaDB unavailable")

    response = api_client.post("/api/query", json={"query": "test"})

    assert response.status_code == 500
    assert "ChromaDB unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

def test_courses_returns_200_with_stats(api_client, mock_rag_system):
    """Successful call returns 200 with course count and titles."""
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"],
    }

    response = api_client.get("/api/courses")

    assert response.status_code == 200
    data = response.json()
    assert data["total_courses"] == 3
    assert data["course_titles"] == ["Course A", "Course B", "Course C"]


def test_courses_returns_zero_when_no_courses_loaded(api_client, mock_rag_system):
    """Empty catalog is represented as total_courses=0 and an empty list."""
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }

    response = api_client.get("/api/courses")

    assert response.status_code == 200
    assert response.json()["total_courses"] == 0
    assert response.json()["course_titles"] == []


def test_courses_returns_500_on_analytics_exception(api_client, mock_rag_system):
    """An exception from get_course_analytics produces a 500 response."""
    mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

    response = api_client.get("/api/courses")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

def test_delete_session_returns_200_with_ok_status(api_client, mock_rag_system):
    """DELETE /api/session/{id} returns 200 and {status: ok}."""
    response = api_client.delete("/api/session/my-session-id")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_delete_session_calls_clear_session_with_correct_id(api_client, mock_rag_system):
    """session_manager.clear_session is called with the path-parameter session_id."""
    api_client.delete("/api/session/session-xyz")

    mock_rag_system.session_manager.clear_session.assert_called_once_with("session-xyz")
