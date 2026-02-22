#!/bin/bash

# Code quality checks for the RAG chatbot project
# Usage: ./check.sh

set -e

echo "=== Running code quality checks ==="
echo ""

echo "--- Checking formatting with black ---"
uv run black --check backend/
echo "Formatting: OK"
echo ""

echo "--- Running tests ---"
uv run pytest backend/tests/
echo ""

echo "=== All checks passed ==="
