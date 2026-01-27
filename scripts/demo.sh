#!/bin/bash

# Demo script for Repo-Trace
# Starts the backend server and provides instructions for opening the UI

set -e

echo "=========================================="
echo "  Repo-Trace Demo"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "✓ Using Python: $PYTHON_CMD"
echo ""

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo ""
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
echo ""

echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Start the backend server
echo "=========================================="
echo "  Starting Backend Server"
echo "=========================================="
echo ""
echo "🚀 Backend will run on: http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "----------------------------------------"
echo "  To use the UI:"
echo "----------------------------------------"
echo "1. Open frontend/index.html in your browser"
echo "   OR"
echo "2. Run: open frontend/index.html (macOS)"
echo "   Run: start frontend/index.html (Windows)"
echo "   Run: xdg-open frontend/index.html (Linux)"
echo ""
echo "----------------------------------------"
echo "  API Endpoints:"
echo "----------------------------------------"
echo "• Health:  GET  http://127.0.0.1:8000/health"
echo "• Ingest:  POST http://127.0.0.1:8000/ingest"
echo "• Status:  GET  http://127.0.0.1:8000/status/{repo_id}"
echo "• Ask:     POST http://127.0.0.1:8000/ask"
echo ""
echo "=========================================="
echo ""

# Start uvicorn
uvicorn app.main:app --reload
