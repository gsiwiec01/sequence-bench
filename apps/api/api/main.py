from fastapi import FastAPI

app = FastAPI(title="sequence-bench API")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
