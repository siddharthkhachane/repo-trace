import json
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Set
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import git
import faiss
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


load_env_file()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

_openai_client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _openai_client = None


app = FastAPI(title="RepoTrace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
STATE_DIR = DATA_DIR / "state"
REPOS_DIR = DATA_DIR / "repos"
INDEXES_DIR = DATA_DIR / "indexes"

STATE_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = int(EMBEDDING_MODEL.get_sentence_embedding_dimension() or 0)

INDEX_CACHE: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {}

app.state.ask_internal_cache = {}
app.state.ask_internal_cache_max = 200

# Q&A Caching
QA_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "total_requests": 0
}
CACHE_TTL_HOURS = 24


def get_cache_key(repo_id: str, question: str) -> str:
    """Generate a cache key from repo_id and question."""
    import hashlib
    key_data = f"{repo_id}:{question.strip().lower()}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached response if it exists and hasn't expired."""
    if cache_key not in QA_CACHE:
        return None

    cached_item = QA_CACHE[cache_key]
    cached_time = cached_item.get("timestamp", 0)
    current_time = datetime.utcnow().timestamp()

    # Check if cache has expired (24 hours)
    if current_time - cached_time > (CACHE_TTL_HOURS * 3600):
        del QA_CACHE[cache_key]  # Remove expired cache
        return None

    return cached_item


def set_cached_response(cache_key: str, repo_id: str, question: str, answer: str, citations: List[Dict[str, Any]], referenced_files: Optional[List[Dict[str, Any]]] = None):
    """Store response in cache."""
    QA_CACHE[cache_key] = {
        "repo_id": repo_id,
        "question": question,
        "answer": answer,
        "citations": citations,
        "referenced_files": referenced_files or [],
        "timestamp": datetime.utcnow().timestamp()
    }

    # Clean up old cache entries if cache gets too large
    if len(QA_CACHE) > 1000:  # Arbitrary limit
        # Remove oldest entries
        sorted_keys = sorted(QA_CACHE.keys(), key=lambda k: QA_CACHE[k]["timestamp"])
        for old_key in sorted_keys[:100]:  # Remove 10% oldest
            del QA_CACHE[old_key]


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
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0


class AskRequest(BaseModel):
    question: str
    repo_id: Optional[str] = None


class Citation(BaseModel):
    commit: str
    author: str
    date: str
    files: list[str]


class FileReference(BaseModel):
    file_path: str
    relevance_score: float
    line_numbers: Optional[list[int]] = None
    preview_snippet: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    referenced_files: list[FileReference]


def get_state_file(repo_id: str) -> Path:
    return STATE_DIR / f"{repo_id}.json"


def load_state(repo_id: str) -> Optional[dict]:
    state_file = get_state_file(repo_id)
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(repo_id: str, state: dict):
    state_file = get_state_file(repo_id)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_vector_index(repo_id: str, docs_file: Path) -> Tuple[int, int]:
    index_dir = INDEXES_DIR / repo_id
    faiss_index_file = index_dir / "faiss.index"
    meta_file = index_dir / "meta.jsonl"

    docs: List[Dict[str, Any]] = []
    with open(docs_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    if not docs:
        return 0, EMBEDDING_DIM

    texts = [str(doc.get("text") or "") for doc in docs]

    embeddings = EMBEDDING_MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings.astype("float32"))  # type: ignore[arg-type]

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_file))

    with open(meta_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            meta_entry = {
                "vector_id": i,
                "doc_id": doc.get("id"),
                "type": doc.get("type"),
                "meta": doc.get("meta", {}) or {},
            }
            f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

    INDEX_CACHE.pop(repo_id, None)
    return int(index.ntotal), EMBEDDING_DIM


def load_index(repo_id: str) -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, Any]]], bool]:
    index_dir = INDEXES_DIR / repo_id
    faiss_index_file = index_dir / "faiss.index"
    meta_file = index_dir / "meta.jsonl"

    if not faiss_index_file.exists() or not meta_file.exists():
        return None, None, False

    try:
        index = faiss.read_index(str(faiss_index_file))

        metadata: List[Dict[str, Any]] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    metadata.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return index, metadata, True
    except Exception:
        return None, None, False


def load_docs_by_id(docs_file: Path, needed_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    if not docs_file.exists() or not needed_ids:
        return docs
    try:
        with open(docs_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get("id")
                if isinstance(doc_id, str) and doc_id in needed_ids:
                    docs[doc_id] = doc
                    if len(docs) == len(needed_ids):
                        break
    except Exception:
        return docs
    return docs


def format_hunk_snippet(hunk_text: str, max_lines: int = 4) -> str:
    if not hunk_text:
        return ""
    lines = [line for line in hunk_text.splitlines() if line.strip()]
    if not lines:
        return ""
    header = lines[0] if lines[0].startswith("@@") else ""
    body_lines = lines[1:] if header else lines
    snippet_lines: List[str] = []
    if header:
        snippet_lines.append(header)
    snippet_lines.extend(body_lines[:max_lines])
    return " | ".join(snippet_lines)


def commit_message_has_revert_keywords(message: Any) -> bool:
    if isinstance(message, (bytes, bytearray)):
        try:
            text = message.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    else:
        text = str(message or "")
    lowered = text.lower()
    return any(token in lowered for token in ("revert", "rollback", "undo"))


def build_commit_file_deltas(commit: git.Commit) -> Dict[str, Dict[str, int]]:
    if not commit.parents:
        return {}

    deltas: Dict[str, Dict[str, int]] = {}
    try:
        diffs = commit.parents[0].diff(commit, create_patch=True)
    except Exception:
        return deltas

    for diff_item in diffs:
        file_path = diff_item.b_path if diff_item.b_path else diff_item.a_path
        if not file_path:
            continue

        patch = diff_item.diff
        if not patch:
            continue

        if isinstance(patch, (bytes, bytearray)):
            diff_text = patch.decode("utf-8", errors="replace")
        else:
            diff_text = str(patch)

        added = 0
        removed = 0
        for line in diff_text.splitlines():
            if not line:
                continue
            if line.startswith(("+++", "---", "@@")):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1

        if added == 0 and removed == 0:
            continue

        entry = deltas.setdefault(file_path, {"added": 0, "removed": 0, "net": 0})
        entry["added"] += added
        entry["removed"] += removed
        entry["net"] += added - removed

    return deltas


def detect_history_attempts(repo_id: str, top_commits: List[str]) -> Dict[str, Any]:
    history_attempts: Dict[str, Any] = {"revert_pairs": [], "revert_markers": []}
    repo_path = REPOS_DIR / repo_id
    if not repo_path.exists():
        return history_attempts

    try:
        repo = git.Repo(repo_path)
        commits = list(repo.iter_commits())
    except Exception:
        return history_attempts

    if not commits:
        return history_attempts

    index_by_hash = {c.hexsha: i for i, c in enumerate(commits)}
    deltas_cache: Dict[str, Dict[str, Dict[str, int]]] = {}
    markers: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()

    def get_deltas(commit_obj: git.Commit) -> Dict[str, Dict[str, int]]:
        cached = deltas_cache.get(commit_obj.hexsha)
        if cached is not None:
            return cached
        deltas = build_commit_file_deltas(commit_obj)
        deltas_cache[commit_obj.hexsha] = deltas
        return deltas

    def opposite_direction(net_a: int, net_b: int) -> bool:
        if net_a == 0 or net_b == 0:
            return False
        return (net_a > 0 and net_b < 0) or (net_a < 0 and net_b > 0)

    for attempt_hash in top_commits:
        idx = index_by_hash.get(attempt_hash)
        if idx is None:
            continue

        attempt_commit = commits[idx]
        if commit_message_has_revert_keywords(attempt_commit.message):
            markers.add(attempt_hash)

        attempt_deltas = get_deltas(attempt_commit)
        if not attempt_deltas:
            continue

        window_start = max(0, idx - 50)
        later_commits = commits[window_start:idx]

        for later_commit in later_commits:
            if commit_message_has_revert_keywords(later_commit.message):
                markers.add(later_commit.hexsha)

            later_deltas = get_deltas(later_commit)
            if not later_deltas:
                continue

            for file_path, delta in attempt_deltas.items():
                later_delta = later_deltas.get(file_path)
                if not later_delta:
                    continue
                if opposite_direction(int(delta.get("net", 0)), int(later_delta.get("net", 0))):
                    pairs.add((attempt_hash, later_commit.hexsha))
                    break
            else:
                continue
            break

    history_attempts["revert_pairs"] = [
        {"attempt": attempt, "reverted_by": reverted_by} for attempt, reverted_by in sorted(pairs)
    ]
    history_attempts["revert_markers"] = sorted(markers)
    return history_attempts


def compose_answer_with_gpt(
    question: str,
    commit_context: List[Dict[str, Any]],
    history_attempts: Optional[Dict[str, Any]] = None,
) -> str:
    if _openai_client is None:
        return ""

    payload = {
        "question": question,
        "top_commits": commit_context,
        "history_attempts": history_attempts or {"revert_pairs": [], "revert_markers": []},
        "requirements": [
            "Explain likely reason using commit messages and diff context",
            "Mention who and when",
            "Mention files affected",
            "Do not invent details not present in provided context",
        ],
    }

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You summarize relevant git commits and diffs to answer questions."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def stream_answer_with_gpt(
    question: str,
    commit_context: List[Dict[str, Any]],
    history_attempts: Optional[Dict[str, Any]] = None,
):
    if _openai_client is None:
        return None

    payload = {
        "question": question,
        "top_commits": commit_context,
        "history_attempts": history_attempts or {"revert_pairs": [], "revert_markers": []},
        "requirements": [
            "Explain likely reason using commit messages and diff context",
            "Mention who and when",
            "Mention files affected",
            "Do not invent details not present in provided context",
        ],
    }

    try:
        return _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You summarize relevant git commits and diffs to answer questions."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            stream=True,
        )
    except Exception:
        return None


def build_deterministic_answer(
    commit_context: List[Dict[str, Any]],
    history_attempts: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = ["Likely reason (from commit messages and diff context):"]
    for c in commit_context:
        commit = str(c.get("commit") or "")
        author = str(c.get("author") or "Unknown")
        date = str(c.get("date") or "")
        msg = str(c.get("commit_message") or "No commit message available.")
        files = c.get("files") or []
        diffs = c.get("diff_context") or []

        lines.append(f"- {commit[:7]} by {author} on {date}: {msg}")
        if files:
            lines.append(f"  Files affected: {', '.join(files)}")
        if diffs:
            lines.append(f"  Diff context: {' || '.join(diffs)}")

    if history_attempts:
        pairs = history_attempts.get("revert_pairs") or []
        if pairs:
            lines.append("Potential revert activity:")
            for pair in pairs:
                attempt = str(pair.get("attempt") or "")
                reverted_by = str(pair.get("reverted_by") or "")
                if attempt and reverted_by:
                    lines.append(f"- {attempt[:7]} reverted by {reverted_by[:7]}")

    return "\n".join(lines).strip()


def store_ask_internal(trace_id: str, payload: Dict[str, Any]) -> None:
    cache: Dict[str, Any] = app.state.ask_internal_cache
    cache[trace_id] = payload
    max_size = int(app.state.ask_internal_cache_max or 200)
    if len(cache) > max_size:
        try:
            to_drop = list(cache.keys())[: max(1, len(cache) - max_size)]
            for k in to_drop:
                cache.pop(k, None)
        except Exception:
            pass


async def index_repo(repo_id: str, github_url: str, branch: Optional[str]):
    try:
        state = load_state(repo_id) or {"repo_id": repo_id}
        state["status"] = "indexing"
        state["started_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)

        repo_path = REPOS_DIR / repo_id
        if repo_path.exists():
            import shutil

            shutil.rmtree(repo_path)

        repo = git.Repo.clone_from(github_url, repo_path, branch=branch)

        index_dir = INDEXES_DIR / repo_id
        index_dir.mkdir(parents=True, exist_ok=True)
        docs_file = index_dir / "docs.jsonl"

        commits = list(repo.iter_commits(max_count=2000))
        commits_indexed = 0
        chunks = 0

        with open(docs_file, "w", encoding="utf-8") as f:
            for commit in commits:
                try:
                    commit_hash = commit.hexsha
                    author = commit.author.name if commit.author else "Unknown"
                    date = commit.committed_datetime.isoformat()
                    message = (commit.message or "").strip()

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
                            "hunk_header": "",
                        },
                    }
                    f.write(json.dumps(commit_doc, ensure_ascii=False) + "\n")
                    chunks += 1

                    if commit.parents:
                        parent = commit.parents[0]
                        diffs = parent.diff(commit, create_patch=True)

                        for diff_item in diffs:
                            file_path = diff_item.b_path if diff_item.b_path else diff_item.a_path
                            if not file_path:
                                continue

                            patch = diff_item.diff
                            if not patch:
                                continue

                            if isinstance(patch, (bytes, bytearray)):
                                diff_text = patch.decode("utf-8", errors="replace")
                            else:
                                diff_text = str(patch)

                            lines = diff_text.split("\n")
                            current_hunk: List[str] = []
                            hunk_header = ""
                            hunk_count = 0

                            for line in lines:
                                if line.startswith("@@"):
                                    if current_hunk and hunk_header:
                                        hunk_text = "\n".join(current_hunk)
                                        if hunk_text.strip():
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
                                                    "hunk_header": hunk_header,
                                                },
                                            }
                                            f.write(json.dumps(hunk_doc, ensure_ascii=False) + "\n")
                                            chunks += 1
                                            hunk_count += 1

                                    hunk_header = line
                                    current_hunk = [line]
                                elif current_hunk:
                                    current_hunk.append(line)

                            if current_hunk and hunk_header:
                                hunk_text = "\n".join(current_hunk)
                                if hunk_text.strip():
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
                                            "hunk_header": hunk_header,
                                        },
                                    }
                                    f.write(json.dumps(hunk_doc, ensure_ascii=False) + "\n")
                                    chunks += 1

                    commits_indexed += 1
                    if commits_indexed % 100 == 0:
                        state["commits_indexed"] = commits_indexed
                        state["chunks"] = chunks
                        save_state(repo_id, state)
                except Exception:
                    continue

        state["status"] = "building_index"
        state["commits_indexed"] = commits_indexed
        state["chunks"] = chunks
        save_state(repo_id, state)

        num_vectors, embed_dim = build_vector_index(repo_id, docs_file)

        state["status"] = "ready"
        state["commits_indexed"] = commits_indexed
        state["chunks"] = chunks
        state["vectors"] = num_vectors
        state["embedding_dim"] = embed_dim
        state["completed_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)
    except Exception as e:
        state = load_state(repo_id) or {"repo_id": repo_id}
        state["status"] = "error"
        state["error"] = str(e)
        state["failed_at"] = datetime.utcnow().isoformat()
        save_state(repo_id, state)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    repo_id = str(uuid.uuid4())

    initial_state = {
        "repo_id": repo_id,
        "github_url": request.github_url,
        "branch": request.branch or "main",
        "status": "indexing",
        "commits_indexed": 0,
        "chunks": 0,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_state(repo_id, initial_state)

    background_tasks.add_task(index_repo, repo_id, request.github_url, request.branch)
    return IngestResponse(repo_id=repo_id, status="indexing")


@app.get("/status/{repo_id}", response_model=StatusResponse)
def get_status(repo_id: str):
    state = load_state(repo_id)
    if not state:
        return StatusResponse(
            repo_id=repo_id,
            status="not_found",
            commits_indexed=0,
            chunks=0,
            error="Repository not found",
            cache_hits=CACHE_STATS["hits"],
            cache_misses=CACHE_STATS["misses"],
            cache_hit_rate=0.0,
        )

    total_requests = CACHE_STATS["total_requests"]
    hit_rate = (CACHE_STATS["hits"] / total_requests * 100) if total_requests > 0 else 0.0

    return StatusResponse(
        repo_id=str(state.get("repo_id", repo_id)),
        status=str(state.get("status", "unknown")),
        commits_indexed=int(state.get("commits_indexed", 0) or 0),
        chunks=int(state.get("chunks", 0) or 0),
        error=state.get("error"),
        cache_hits=CACHE_STATS["hits"],
        cache_misses=CACHE_STATS["misses"],
        cache_hit_rate=round(hit_rate, 1),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request):
    trace_id = str(uuid.uuid4())
    CACHE_STATS["total_requests"] += 1

    try:
        repo_id = request.repo_id
        question = request.question

        if not repo_id:
            CACHE_STATS["misses"] += 1
            store_ask_internal(
                trace_id,
                {
                    "trace_id": trace_id,
                    "repo_id": repo_id,
                    "question": question,
                    "error": "missing_repo_id",
                    "history_attempts": {"revert_pairs": [], "revert_markers": []},
                },
            )
            return AskResponse(
                answer="Missing repo_id. Please provide a valid repo_id from /ingest.",
                citations=[],
                referenced_files=[],
            )

        if not question or not question.strip():
            CACHE_STATS["misses"] += 1
            store_ask_internal(
                trace_id,
                {
                    "trace_id": trace_id,
                    "repo_id": repo_id,
                    "question": question,
                    "error": "missing_question",
                    "history_attempts": {"revert_pairs": [], "revert_markers": []},
                },
            )
            return AskResponse(
                answer="Missing question. Please provide a non-empty question.",
                citations=[],
                referenced_files=[],
            )

        # Check cache first
        cache_key = get_cache_key(repo_id, question)
        cached_response = get_cached_response(cache_key)
        if cached_response:
            CACHE_STATS["hits"] += 1
            # Return cached response
            citations = [Citation(**c) for c in cached_response["citations"]]
            referenced_files = [FileReference(**f) for f in cached_response.get("referenced_files", [])]
            return AskResponse(
                answer=cached_response["answer"],
                citations=citations,
                referenced_files=referenced_files,
            )

        CACHE_STATS["misses"] += 1

        cached = INDEX_CACHE.get(repo_id)
        if cached:
            index, metadata = cached
        else:
            index, metadata, ok = load_index(repo_id)
            if not ok or index is None or metadata is None:
                store_ask_internal(
                    trace_id,
                    {
                        "trace_id": trace_id,
                        "repo_id": repo_id,
                        "question": question,
                        "error": "index_not_found",
                        "history_attempts": {"revert_pairs": [], "revert_markers": []},
                    },
                )
                return AskResponse(
                    answer=(
                        f"No index found for repo_id '{repo_id}'. "
                        "Please run /ingest and wait for indexing to complete."
                    ),
                    citations=[],
                    referenced_files=[],
                )
            INDEX_CACHE[repo_id] = (index, metadata)

        if int(index.ntotal) == 0:
            store_ask_internal(
                trace_id,
                {
                    "trace_id": trace_id,
                    "repo_id": repo_id,
                    "question": question,
                    "error": "empty_index",
                    "history_attempts": {"revert_pairs": [], "revert_markers": []},
                },
            )
            return AskResponse(
                answer=f"Index for repo_id '{repo_id}' is empty. Try re-ingesting the repository.",
                citations=[],
                referenced_files=[],
            )
        
        # Create query vector from question
        query_vec = EMBEDDING_MODEL.encode([question], convert_to_numpy=True)[0]
        top_k = 10
        
        distances, indices = index.search(query_vec.reshape(1, -1).astype("float32"), top_k)  # type: ignore[call-arg]

        meta_by_vector: Dict[int, Dict[str, Any]] = {}
        for m in metadata:
            vid_raw = m.get("vector_id")
            if vid_raw is None:
                continue
            try:
                vid = int(vid_raw)
            except Exception:
                continue
            meta_by_vector[vid] = m

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if int(idx) == -1:
                continue
            meta_entry = meta_by_vector.get(int(idx))
            if not meta_entry:
                continue
            results.append(
                {
                    "distance": float(dist),
                    "doc_id": meta_entry.get("doc_id"),
                    "type": meta_entry.get("type"),
                    "meta": meta_entry.get("meta", {}) or {},
                }
            )

        if not results:
            store_ask_internal(
                trace_id,
                {
                    "trace_id": trace_id,
                    "repo_id": repo_id,
                    "question": question,
                    "error": "no_results",
                    "history_attempts": {"revert_pairs": [], "revert_markers": []},
                },
            )
            return AskResponse(answer="No relevant commits found for this question.", citations=[], referenced_files=[])

        index_dir = INDEXES_DIR / repo_id
        docs_file = index_dir / "docs.jsonl"
        doc_ids: Set[str] = {str(r["doc_id"]) for r in results if r.get("doc_id") is not None}
        docs_by_id = load_docs_by_id(docs_file, doc_ids)

        commits: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta = r.get("meta") or {}
            commit = meta.get("commit")
            if not isinstance(commit, str) or not commit:
                continue

            entry = commits.setdefault(
                commit,
                {
                    "best_distance": float(r["distance"]),
                    "results": [],
                    "author": meta.get("author", "Unknown"),
                    "date": meta.get("date", ""),
                    "files": set(),
                },
            )
            entry["best_distance"] = min(float(entry["best_distance"]), float(r["distance"]))
            entry["results"].append(r)

            for file_path in (meta.get("files") or []):
                if file_path:
                    entry["files"].add(str(file_path))

        if not commits:
            store_ask_internal(
                trace_id,
                {
                    "trace_id": trace_id,
                    "repo_id": repo_id,
                    "question": question,
                    "error": "no_commits",
                    "history_attempts": {"revert_pairs": [], "revert_markers": []},
                },
            )
            return AskResponse(answer="No relevant commits found for this question.", citations=[], referenced_files=[])

        top_commits = sorted(commits.items(), key=lambda item: float(item[1]["best_distance"]))[:3]

        commit_context: List[Dict[str, Any]] = []
        citations: List[Citation] = []
        file_references: Dict[str, Dict[str, Any]] = {}

        for commit, data in top_commits:
            author = str(data.get("author", "Unknown"))
            date = str(data.get("date", ""))
            files = sorted(list(data.get("files", set())))

            # Track file references with relevance scores
            for file_path in files:
                if file_path not in file_references:
                    file_references[file_path] = {
                        "relevance_score": 1.0 - float(data["best_distance"]),  # Convert distance to relevance (higher is better)
                        "line_numbers": [],
                        "preview_snippet": None
                    }
                else:
                    # Update relevance score (take the maximum)
                    file_references[file_path]["relevance_score"] = max(
                        file_references[file_path]["relevance_score"],
                        1.0 - float(data["best_distance"])
                    )

            commit_message = ""
            hunk_snippets: List[str] = []

            for r in data["results"]:
                doc_id = r.get("doc_id")
                if doc_id is None:
                    continue
                doc = docs_by_id.get(str(doc_id), {}) or {}
                rtype = r.get("type")

                if rtype == "commit_message" and not commit_message:
                    commit_message = str(doc.get("text") or "").strip()
                elif rtype == "diff_hunk" and len(hunk_snippets) < 2:
                    snippet = format_hunk_snippet(str(doc.get("text") or ""))
                    if snippet:
                        hunk_snippets.append(snippet)

            if not commit_message:
                commit_message = "No commit message available."

            commit_context.append(
                {
                    "commit": commit,
                    "author": author,
                    "date": date,
                    "files": files,
                    "commit_message": commit_message,
                    "diff_context": hunk_snippets,
                }
            )
            citations.append(Citation(commit=commit, author=author, date=date, files=files))

        # Convert file references to list and sort by relevance
        referenced_files_list = [
            FileReference(
                file_path=file_path,
                relevance_score=round(data["relevance_score"], 3),
                line_numbers=data.get("line_numbers", []),
                preview_snippet=data.get("preview_snippet")
            )
            for file_path, data in file_references.items()
        ]
        referenced_files_list.sort(key=lambda x: x.relevance_score, reverse=True)
        # Limit to top 5 files
        referenced_files_list = referenced_files_list[:5]

        history_attempts = detect_history_attempts(
            repo_id,
            [str(c.get("commit") or "") for c in commit_context if c.get("commit")],
        )

        answer = compose_answer_with_gpt(question, commit_context, history_attempts)
        if not answer:
            answer = build_deterministic_answer(commit_context, history_attempts)

        ask_internal = {
            "trace_id": trace_id,
            "repo_id": repo_id,
            "question": question,
            "commit_context": commit_context,
            "history_attempts": history_attempts,
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "referenced_files": [f.model_dump() for f in referenced_files_list],
        }
        store_ask_internal(trace_id, ask_internal)
        http_request.state.ask_internal = ask_internal

        # Cache the response
        set_cached_response(cache_key, repo_id, question, answer, [c.model_dump() for c in citations], [f.model_dump() for f in referenced_files_list])

        return AskResponse(
            answer=answer,
            citations=citations,
            referenced_files=referenced_files_list
        )

    except Exception as e:
        store_ask_internal(
            trace_id,
            {
                "trace_id": trace_id,
                "repo_id": getattr(request, "repo_id", None),
                "question": getattr(request, "question", None),
                "error": f"{type(e).__name__}: {e}",
                "history_attempts": {"revert_pairs": [], "revert_markers": []},
            },
        )
        return AskResponse(
            answer=(
                "I couldn't answer due to an internal error while querying the repo index. "
                f"Details: {type(e).__name__}: {e}"
            ),
            citations=[],
            referenced_files=[],
        )


@app.get("/ask")
async def ask_stream(request: Request, repo_id: Optional[str] = None, question: Optional[str] = None):
    trace_id = str(uuid.uuid4())
    CACHE_STATS["total_requests"] += 1

    async def event_generator():
        def sse(payload: Dict[str, Any]) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            if not repo_id:
                CACHE_STATS["misses"] += 1
                yield sse({"type": "error", "message": "Missing repo_id. Please provide a valid repo_id from /ingest."})
                return

            if not question or not question.strip():
                CACHE_STATS["misses"] += 1
                yield sse({"type": "error", "message": "Missing question. Please provide a non-empty question."})
                return

            cache_key = get_cache_key(repo_id, question)
            cached_response = get_cached_response(cache_key)
            if cached_response:
                CACHE_STATS["hits"] += 1
                for word in str(cached_response.get("answer", "")).split():
                    if await request.is_disconnected():
                        return
                    yield sse({"type": "token", "content": word + " "})
                yield sse(
                    {
                        "type": "done",
                        "answer": cached_response.get("answer", ""),
                        "citations": cached_response.get("citations", []),
                        "referenced_files": cached_response.get("referenced_files", []),
                    }
                )
                return

            CACHE_STATS["misses"] += 1

            cached = INDEX_CACHE.get(repo_id)
            if cached:
                index, metadata = cached
            else:
                index, metadata, ok = load_index(repo_id)
                if not ok or index is None or metadata is None:
                    yield sse({"type": "error", "message": f"No index found for repo_id '{repo_id}'. Please run /ingest and wait for indexing to complete."})
                    return
                INDEX_CACHE[repo_id] = (index, metadata)

            if int(index.ntotal) == 0:
                yield sse({"type": "error", "message": f"Index for repo_id '{repo_id}' is empty. Try re-ingesting the repository."})
                return

            query_vec = EMBEDDING_MODEL.encode([question], convert_to_numpy=True)[0]
            top_k = 10

            distances, indices = index.search(query_vec.reshape(1, -1).astype("float32"), top_k)  # type: ignore[call-arg]

            meta_by_vector: Dict[int, Dict[str, Any]] = {}
            for m in metadata:
                vid_raw = m.get("vector_id")
                if vid_raw is None:
                    continue
                try:
                    vid = int(vid_raw)
                except Exception:
                    continue
                meta_by_vector[vid] = m

            results: List[Dict[str, Any]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if int(idx) == -1:
                    continue
                meta_entry = meta_by_vector.get(int(idx))
                if not meta_entry:
                    continue
                results.append(
                    {
                        "distance": float(dist),
                        "doc_id": meta_entry.get("doc_id"),
                        "type": meta_entry.get("type"),
                        "meta": meta_entry.get("meta", {}) or {},
                    }
                )

            if not results:
                yield sse({"type": "error", "message": "No relevant commits found for this question."})
                return

            index_dir = INDEXES_DIR / repo_id
            docs_file = index_dir / "docs.jsonl"
            doc_ids: Set[str] = {str(r["doc_id"]) for r in results if r.get("doc_id") is not None}
            docs_by_id = load_docs_by_id(docs_file, doc_ids)

            commits: Dict[str, Dict[str, Any]] = {}
            for r in results:
                meta = r.get("meta") or {}
                commit = meta.get("commit")
                if not isinstance(commit, str) or not commit:
                    continue

                entry = commits.setdefault(
                    commit,
                    {
                        "best_distance": float(r["distance"]),
                        "results": [],
                        "author": meta.get("author", "Unknown"),
                        "date": meta.get("date", ""),
                        "files": set(),
                    },
                )
                entry["best_distance"] = min(float(entry["best_distance"]), float(r["distance"]))
                entry["results"].append(r)

                for file_path in (meta.get("files") or []):
                    if file_path:
                        entry["files"].add(str(file_path))

            if not commits:
                yield sse({"type": "error", "message": "No relevant commits found for this question."})
                return

            top_commits = sorted(commits.items(), key=lambda item: float(item[1]["best_distance"]))[:3]

            commit_context: List[Dict[str, Any]] = []
            citations: List[Citation] = []
            file_references: Dict[str, Dict[str, Any]] = {}

            for commit, data in top_commits:
                author = str(data.get("author", "Unknown"))
                date = str(data.get("date", ""))
                files = sorted(list(data.get("files", set())))

                for file_path in files:
                    if file_path not in file_references:
                        file_references[file_path] = {
                            "relevance_score": 1.0 - float(data["best_distance"]),
                            "line_numbers": [],
                            "preview_snippet": None,
                        }
                    else:
                        file_references[file_path]["relevance_score"] = max(
                            file_references[file_path]["relevance_score"],
                            1.0 - float(data["best_distance"])
                        )

                commit_message = ""
                hunk_snippets: List[str] = []

                for r in data["results"]:
                    doc_id = r.get("doc_id")
                    if doc_id is None:
                        continue
                    doc = docs_by_id.get(str(doc_id), {}) or {}
                    rtype = r.get("type")

                    if rtype == "commit_message" and not commit_message:
                        commit_message = str(doc.get("text") or "").strip()
                    elif rtype == "diff_hunk" and len(hunk_snippets) < 2:
                        snippet = format_hunk_snippet(str(doc.get("text") or ""))
                        if snippet:
                            hunk_snippets.append(snippet)

                if not commit_message:
                    commit_message = "No commit message available."

                commit_context.append(
                    {
                        "commit": commit,
                        "author": author,
                        "date": date,
                        "files": files,
                        "commit_message": commit_message,
                        "diff_context": hunk_snippets,
                    }
                )
                citations.append(Citation(commit=commit, author=author, date=date, files=files))

            referenced_files_list = [
                FileReference(
                    file_path=file_path,
                    relevance_score=round(data["relevance_score"], 3),
                    line_numbers=data.get("line_numbers", []),
                    preview_snippet=data.get("preview_snippet"),
                )
                for file_path, data in file_references.items()
            ]
            referenced_files_list.sort(key=lambda x: x.relevance_score, reverse=True)
            referenced_files_list = referenced_files_list[:5]

            history_attempts = detect_history_attempts(
                repo_id,
                [str(c.get("commit") or "") for c in commit_context if c.get("commit")],
            )

            answer_parts: List[str] = []
            stream = stream_answer_with_gpt(question, commit_context, history_attempts)
            if stream is not None:
                for event in stream:
                    if await request.is_disconnected():
                        return
                    delta = event.choices[0].delta.content if event.choices else None
                    if not delta:
                        continue
                    answer_parts.append(delta)
                    yield sse({"type": "token", "content": delta})

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = build_deterministic_answer(commit_context, history_attempts)
                for word in answer.split():
                    if await request.is_disconnected():
                        return
                    yield sse({"type": "token", "content": word + " "})

            ask_internal = {
                "trace_id": trace_id,
                "repo_id": repo_id,
                "question": question,
                "commit_context": commit_context,
                "history_attempts": history_attempts,
                "answer": answer,
                "citations": [c.model_dump() for c in citations],
                "referenced_files": [f.model_dump() for f in referenced_files_list],
            }
            store_ask_internal(trace_id, ask_internal)

            set_cached_response(
                cache_key,
                repo_id,
                question,
                answer,
                [c.model_dump() for c in citations],
                [f.model_dump() for f in referenced_files_list],
            )

            yield sse(
                {
                    "type": "done",
                    "answer": answer,
                    "citations": [c.model_dump() for c in citations],
                    "referenced_files": [f.model_dump() for f in referenced_files_list],
                }
            )

        except Exception as e:
            yield sse({"type": "error", "message": f"I couldn't answer due to an internal error. Details: {type(e).__name__}: {e}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
