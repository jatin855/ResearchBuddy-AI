# ResearchBuddy AI Service

FastAPI service for PDF extraction, section detection, chunking, embeddings, Chroma-backed retrieval, grounded summaries, QA, and paper comparison.

Run locally:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Optional ML/vector dependencies:

```bash
pip install -r requirements-ml.txt
```

Use Python 3.11 or 3.12 for the optional Chroma install on Windows, or install Microsoft C++ Build Tools first. Without those optional packages, the service still runs with deterministic fallback embeddings and in-process retrieval.
