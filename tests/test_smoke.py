"""
Smoke tests for Repo-Trace API.

Tests the basic ingest -> wait -> ask workflow.
"""
import os
import time
import httpx
import pytest


# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TEST_REPO_URL = "https://github.com/octocat/Hello-World"
TEST_BRANCH = "master"
POLL_INTERVAL = 2  # seconds
TIMEOUT = 180  # seconds


@pytest.fixture
def client():
    """Create an httpx client for testing."""
    return httpx.Client(base_url=BACKEND_URL, timeout=30.0)


def test_health(client):
    """Test that the health endpoint responds."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert data["ok"] is True


def test_ingest_and_ask(client):
    """
    Integration test: ingest a small repo, wait for completion, then ask a question.
    """
    # Step 1: POST /ingest
    ingest_payload = {
        "github_url": TEST_REPO_URL,
        "branch": TEST_BRANCH
    }
    response = client.post("/ingest", json=ingest_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "repo_id" in data
    assert "status" in data
    
    repo_id = data["repo_id"]
    assert isinstance(repo_id, str)
    assert len(repo_id) > 0
    assert data["status"] == "indexing"
    
    print(f"\n✓ Ingestion started for repo_id: {repo_id}")
    
    # Step 2: Poll /status until ready or failed
    start_time = time.time()
    status = "indexing"
    
    while status not in ["completed", "error"]:
        if time.time() - start_time > TIMEOUT:
            pytest.fail(f"Timeout waiting for ingestion to complete after {TIMEOUT}s")
        
        time.sleep(POLL_INTERVAL)
        
        response = client.get(f"/status/{repo_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "commits_indexed" in data
        assert "chunks" in data
        
        status = data["status"]
        print(f"  Status: {status}, Commits: {data['commits_indexed']}, Chunks: {data['chunks']}")
    
    # Verify final status
    assert status == "completed", f"Ingestion failed: {data.get('error', 'unknown error')}"
    assert data["commits_indexed"] > 0
    assert data["chunks"] > 0
    
    print(f"✓ Ingestion completed: {data['commits_indexed']} commits, {data['chunks']} chunks")
    
    # Step 3: POST /ask
    ask_payload = {
        "question": "What does this repository do?",
        "repo_id": repo_id
    }
    response = client.post("/ask", json=ask_payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Assert required keys exist
    assert "answer" in data
    assert "citations" in data
    
    # Assert correct types
    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    
    # Answer should not be empty
    assert len(data["answer"]) > 0
    
    print(f"✓ Ask endpoint returned answer: {data['answer'][:100]}...")
    print(f"✓ Citations count: {len(data['citations'])}")


if __name__ == "__main__":
    # Allow running directly for manual testing
    pytest.main([__file__, "-v", "-s"])
