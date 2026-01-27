import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import git

# Initialize FastAPI app
app = FastAPI(title="RepoTrace API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
DATA_DIR = Path("data")
STATE_DIR = DATA_DIR / "state"
REPOS_DIR = DATA_DIR / "repos"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)


# Models
class IngestRequest(BaseModel):
    github_url: str
    branch: Optional[str] = None


class IngestResponse(BaseModel):
    repo_id: str
    status: str


class StatusResponse(BaseModel):
    repo_id: str
    status: str
    commits_indexed: int
    chunks: int
    error: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    repo_id: Optional[str] = None


class Citation(BaseModel):
    commit: str
    file: str
    line: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


# State management
def get_state_file(repo_id: str) -> Path:
    return STATE_DIR / f"{repo_id}.json"


def load_state(repo_id: str) -> dict:
    state_file = get_state_file(repo_id)
    if state_file.exists():
        return json.loads(state_file.read_text())
    return None


def save_state(repo_id: str, state: dict):
    state_file = get_state_file(repo_id)
    state_file.write_text(json.dumps(state, indent=2))


# Background task for indexing
async def index_repo(repo_id: str, github_url: str, branch: Optional[str]):
    """Background task to clone and index a repository."""
    try:
        # Update state to indexing
        state = load_state(repo_id)
        state["status"] = "indexing"
        state["started_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)
        
        # Clone the repository
        repo_path = REPOS_DIR / repo_id
        if repo_path.exists():
            # Remove existing clone
            import shutil
            shutil.rmtree(repo_path)
        
        # Clone repo
        repo = git.Repo.clone_from(github_url, repo_path, branch=branch)
        
        # Simulate indexing (dummy implementation)
        commits = list(repo.iter_commits())
        commits_indexed = len(commits)
        
        # Simulate some processing time
        await asyncio.sleep(2)
        
        # Update state to completed
        state["status"] = "completed"
        state["commits_indexed"] = commits_indexed
        state["chunks"] = commits_indexed * 5  # Dummy calculation
        state["completed_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)
        
    except Exception as e:
        # Update state to error
        state = load_state(repo_id)
        state["status"] = "error"
        state["error"] = str(e)
        state["failed_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)


# Endpoints
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Create a new repository indexing job."""
    # Generate unique repo_id
    repo_id = str(uuid.uuid4())
    
    # Initialize state
    initial_state = {
        "repo_id": repo_id,
        "github_url": request.github_url,
        "branch": request.branch or "main",
        "status": "indexing",
        "commits_indexed": 0,
        "chunks": 0,
        "error": None,
        "created_at": datetime.utcnow().isoformat()
    }
    save_state(repo_id, initial_state)
    
    # Kick off background task
    background_tasks.add_task(index_repo, repo_id, request.github_url, request.branch)
    
    return IngestResponse(repo_id=repo_id, status="indexing")


@app.get("/status/{repo_id}", response_model=StatusResponse)
def get_status(repo_id: str):
    """Get the status of a repository indexing job."""
    state = load_state(repo_id)
    
    if not state:
        return StatusResponse(
            repo_id=repo_id,
            status="not_found",
            commits_indexed=0,
            chunks=0,
            error="Repository not found"
        )
    
    return StatusResponse(
        repo_id=state["repo_id"],
        status=state["status"],
        commits_indexed=state.get("commits_indexed", 0),
        chunks=state.get("chunks", 0),
        error=state.get("error")
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Ask a question about a repository."""
    # Placeholder implementation
    return AskResponse(
        answer="This is a placeholder answer. The actual RAG system is not yet implemented.",
        citations=[]
    )