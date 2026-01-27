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
INDEXES_DIR = DATA_DIR / "indexes"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)


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
        
        # Create indexes directory for this repo
        index_dir = INDEXES_DIR / repo_id
        index_dir.mkdir(parents=True, exist_ok=True)
        docs_file = index_dir / "docs.jsonl"
        
        # Extract commits and diffs (limit to last 2000 commits)
        commits = list(repo.iter_commits(max_count=2000))
        commits_indexed = 0
        chunks = 0
        
        with open(docs_file, 'w', encoding='utf-8') as f:
            for commit in commits:
                try:
                    # Extract commit metadata
                    commit_hash = commit.hexsha
                    author = commit.author.name if commit.author else "Unknown"
                    date = commit.committed_datetime.isoformat()
                    message = commit.message.strip()
                    
                    # Write commit message document
                    commit_doc = {
                        "id": f"{commit_hash}_msg",
                        "type": "commit_message",
                        "text": message,
                        "meta": {
                            "commit": commit_hash,
                            "author": author,
                            "date": date,
                            "files": [],
                            "path": "",
                            "hunk_header": ""
                        }
                    }
                    f.write(json.dumps(commit_doc, ensure_ascii=False) + '\n')
                    chunks += 1
                    
                    # Extract diff hunks
                    # Get parent commit for diff
                    if commit.parents:
                        parent = commit.parents[0]
                        diffs = parent.diff(commit, create_patch=True)
                        
                        for diff_item in diffs:
                            try:
                                # Get file path
                                file_path = diff_item.b_path if diff_item.b_path else diff_item.a_path
                                if not file_path:
                                    continue
                                
                                # Get the unified diff
                                if diff_item.diff:
                                    diff_text = diff_item.diff.decode('utf-8', errors='replace')
                                    
                                    # Split diff into hunks (sections starting with @@)
                                    lines = diff_text.split('\n')
                                    current_hunk = []
                                    hunk_header = ""
                                    hunk_count = 0
                                    
                                    for line in lines:
                                        if line.startswith('@@'):
                                            # Save previous hunk if exists
                                            if current_hunk and hunk_header:
                                                hunk_text = '\n'.join(current_hunk)
                                                # Limit context to 3 lines before and after
                                                if len(hunk_text.strip()) > 0:
                                                    hunk_doc = {
                                                        "id": f"{commit_hash}_{file_path}_{hunk_count}",
                                                        "type": "diff_hunk",
                                                        "text": hunk_text,
                                                        "meta": {
                                                            "commit": commit_hash,
                                                            "author": author,
                                                            "date": date,
                                                            "files": [file_path],
                                                            "path": file_path,
                                                            "hunk_header": hunk_header
                                                        }
                                                    }
                                                    f.write(json.dumps(hunk_doc, ensure_ascii=False) + '\n')
                                                    chunks += 1
                                                    hunk_count += 1
                                            
                                            # Start new hunk
                                            hunk_header = line
                                            current_hunk = [line]
                                        elif current_hunk:
                                            current_hunk.append(line)
                                    
                                    # Save last hunk
                                    if current_hunk and hunk_header:
                                        hunk_text = '\n'.join(current_hunk)
                                        if len(hunk_text.strip()) > 0:
                                            hunk_doc = {
                                                "id": f"{commit_hash}_{file_path}_{hunk_count}",
                                                "type": "diff_hunk",
                                                "text": hunk_text,
                                                "meta": {
                                                    "commit": commit_hash,
                                                    "author": author,
                                                    "date": date,
                                                    "files": [file_path],
                                                    "path": file_path,
                                                    "hunk_header": hunk_header
                                                }
                                            }
                                            f.write(json.dumps(hunk_doc, ensure_ascii=False) + '\n')
                                            chunks += 1
                                            
                            except Exception as e:
                                # Skip individual diff processing errors
                                print(f"Error processing diff for {file_path}: {e}")
                                continue
                    
                    commits_indexed += 1
                    
                    # Update state periodically
                    if commits_indexed % 100 == 0:
                        state["commits_indexed"] = commits_indexed
                        state["chunks"] = chunks
                        save_state(repo_id, state)
                        
                except Exception as e:
                    # Skip individual commit processing errors
                    print(f"Error processing commit {commit.hexsha}: {e}")
                    continue
        
        # Update state to completed
        state["status"] = "completed"
        state["commits_indexed"] = commits_indexed
        state["chunks"] = chunks
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