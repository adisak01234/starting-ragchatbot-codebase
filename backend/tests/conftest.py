import sys
import os

# Add backend/ to sys.path so that `import ai_generator`, `import vector_store`, etc. work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared fixtures
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
def mock_rag_system():
    """Fully mocked RAGSystem for API-level tests."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = ("Test answer", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Advanced AI"],
    }
    return rag


@pytest.fixture
def api_client(mock_rag_system):
    """
    TestClient backed by a minimal FastAPI app that mirrors app.py's API
    routes but omits the static-file mount (which requires ../frontend to
    exist on disk).  The RAGSystem is replaced by mock_rag_system so no
    real ChromaDB or Anthropic calls are made.
    """

    # --- Pydantic models (mirror of app.py) ---
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        label: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # --- Build a fresh app with only the API routes ---
    test_app = FastAPI()

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag_system.session_manager.clear_session(session_id)
        return {"status": "ok"}

    return TestClient(test_app)
