# repo-trace

RepoTrace is a commit-aware retrieval-augmented generation (RAG) system for GitHub repositories.  
It answers questions about codebases by grounding responses in commits, diffs, and pull requests, with explicit citations back to source evidence.

The primary design goal is traceability. Every material claim in an answer must be attributable to a specific commit, file, or diff hunk.

---

## Core Capabilities

- Commit- and diff-grounded question answering
- Pull request summarization with source citations
- Change origin attribution (where a change was introduced)
- Hybrid retrieval over commit history and code artifacts
- Trace inspection for debugging retrieval and generation

---

## High-Level Architecture

- **Backend**: FastAPI deployed as serverless functions on Vercel
- **Client Application**: Separate frontend application consuming the API
- **Database**: Supabase Postgres with pgvector
- **Storage**: Supabase Storage for large artifacts
- **Indexing**: Background jobs (e.g., GitHub Actions) for repository ingestion
- **LLM**: External model provider (e.g., OpenAI)

Vercel handles request-response workloads only. Repository ingestion and indexing run outside the request path.

---

## Repository Structure (planned)


## Local Development

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the API server:
   - `uvicorn app.main:app --reload`
4. Health check:
   - `GET http://127.0.0.1:8000/health`