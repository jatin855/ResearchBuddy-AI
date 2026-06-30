# ResearchBuddy AI Service

FastAPI service for PDF extraction, section detection, chunking, embeddings, Chroma-backed retrieval, grounded summaries, QA, and paper comparison.

Run locally:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Required ML/vector dependencies:

```bash
pip install -r requirements-ml.txt
```

Use Python 3.11 or 3.12 for the Chroma install on Windows, or install Microsoft C++ Build Tools first. The service validates all components (Gemini API, SentenceTransformer model, and ChromaDB) during startup and will fail fast to prevent running in a misconfigured state.
