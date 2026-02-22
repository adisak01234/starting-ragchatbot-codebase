# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Install dependencies
uv sync

# Start the server (from repo root, using Git Bash on Windows)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the repo root:
```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

On startup, the server auto-ingests `.txt` files from `docs/` into ChromaDB and skips already-processed courses.

## Architecture

This is a RAG chatbot with a FastAPI backend and vanilla JS frontend. All backend code lives in `backend/`; the frontend is served as static files from `frontend/`.

### Request Flow

1. Frontend (`script.js`) POSTs `{ query, session_id }` to `/api/query`
2. `app.py` creates a session if needed, delegates to `RAGSystem.query()`
3. `RAGSystem` fetches conversation history from `SessionManager`, then calls `AIGenerator.generate_response()` with the query, history, and tool definitions
4. **Claude API call #1**: Claude decides whether to call `search_course_content` (the `CourseSearchTool`) or answer directly
5. If tool use: `CourseSearchTool` runs a ChromaDB similarity search on `course_content`, optionally filtered by course/lesson. Course name fuzzy-matching is done via a separate semantic search on `course_catalog`
6. **Claude API call #2**: Claude synthesizes a final answer using the retrieved chunks
7. Sources (course + lesson labels) are extracted from `ToolManager` and returned alongside the answer

### Key Architectural Decisions

- **Two ChromaDB collections**: `course_catalog` stores course-level metadata (used for fuzzy course name resolution); `course_content` stores the actual text chunks (used for semantic search)
- **Tool calling drives retrieval**: Claude decides when and how to search — the system does not pre-fetch context before the LLM call. One search per query is enforced in the system prompt
- **Conversation history is injected into the system prompt** as a formatted string, not as additional messages in the `messages` array
- **Session state is in-memory only** — sessions are lost on server restart
- **Embeddings**: `all-MiniLM-L6-v2` via SentenceTransformers, used for both collections

### Adding a New Tool

Implement the `Tool` ABC in `search_tools.py` (requires `get_tool_definition()` returning an Anthropic tool schema and `execute(**kwargs)`), then register it via `tool_manager.register_tool()` in `RAGSystem.__init__()`. Sources are tracked by adding a `last_sources` list attribute to the tool — `ToolManager` picks this up automatically.

### Document Format

Course `.txt` files in `docs/` must follow this structure for proper parsing:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <title>
...
```

`DocumentProcessor` chunks lesson text at sentence boundaries (~800 chars, 100 char overlap) and prepends lesson/course context to the first chunk of each lesson.

### Configuration

All tunable values are in `backend/config.py`: model name, chunk size/overlap, max search results, max conversation history, and ChromaDB path.
