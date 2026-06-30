# ResearchBuddy AI

ResearchBuddy AI is a RAG-based research paper claim summarizer with a React frontend, Spring Boot backend, FastAPI AI service, Chroma vector storage, Sentence Transformers embeddings, and PostgreSQL metadata storage.

## Services

- Frontend: `frontend`, served by Vite on `http://localhost:5173`
- Backend: `backend`, Spring Boot API on `http://localhost:8080`
- AI service: `ai-service`, FastAPI on `http://localhost:8000`
- Database: PostgreSQL on `localhost:5432`

## Run Locally

Prerequisites:

- Node.js 20+
- Python 3.11+ for default service, preferably Python 3.11 or 3.12 for optional Chroma on Windows
- Docker Desktop
- JDK 17+
- Apache Maven 3.9+

Check Java and Maven:

```bash
java -version
mvn -v
```

On Windows, make sure `mvn` points to Apache Maven, not an Anaconda Python package. If `Get-Command mvn` shows an Anaconda path, remove that conflicting package or put Apache Maven's `bin` folder earlier in PATH.

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Start the AI service:

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Required real embedding/vector packages:

```bash
pip install -r requirements-ml.txt
```

On Windows, use Python 3.11 or 3.12 for the Chroma install, or install Microsoft C++ Build Tools. These dependencies are mandatory for the AI service.

Start the backend:

```bash
cd backend
mvn spring-boot:run
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Shape

Backend:

- `POST /api/papers` uploads a PDF
- `GET /api/papers` lists uploaded papers
- `GET /api/papers/{id}/summary` returns grounded summary sections
- `POST /api/papers/{id}/qa` answers a question from retrieved paper chunks
- `POST /api/papers/compare` compares two uploaded papers

AI service:

- `POST /papers/upload`
- `GET /papers/{paper_id}/summary`
- `GET /papers/{paper_id}/claims`
- `GET /papers/{paper_id}/limitations`
- `GET /papers/{paper_id}/future-work`
- `GET /papers/{paper_id}/datasets-metrics`
- `POST /papers/{paper_id}/qa`
- `POST /papers/compare`
- `POST /tools/extract`
- `POST /tools/chunk`
- `POST /tools/embed`
- `POST /tools/retrieve`

The AI service uses real PDF extraction, section detection, chunking, Sentence Transformers embeddings (all-MiniLM-L6-v2), and ChromaDB (researchbuddy_chunks). It validates all components during startup and exits immediately if any dependency is unavailable.
