from fastapi import FastAPI

app = FastAPI(title="RepoTrace API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}