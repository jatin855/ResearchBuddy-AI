from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import math
import re

try:
    import fitz
except Exception:
    fitz = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
except Exception:
    chromadb = None

app = FastAPI(title="ResearchBuddy AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
pdf_dir = data_dir / "pdfs"
paper_dir = data_dir / "papers"
vector_dir = data_dir / "vectors"
for folder in [pdf_dir, paper_dir, vector_dir]:
    folder.mkdir(parents=True, exist_ok=True)

model = None
client = None
collection = None
memory_vectors = {}

section_names = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "methodology",
    "method",
    "methods",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "future work",
]


class QuestionRequest(BaseModel):
    question: str


class CompareRequest(BaseModel):
    paperIds: list[str]


class TextRequest(BaseModel):
    text: str


class RetrievalRequest(BaseModel):
    paperId: str
    query: str
    topK: int = 5


def paper_path(paper_id):
    return paper_dir / f"{paper_id}.json"


def load_paper(paper_id):
    path = paper_path(paper_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")
    return json.loads(path.read_text(encoding="utf-8"))


def save_paper(paper):
    paper_path(paper["id"]).write_text(json.dumps(paper, indent=2), encoding="utf-8")


def get_model():
    global model
    if model is None and SentenceTransformer is not None:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            model = False
    return model


def get_collection():
    global client, collection
    if collection is not None:
        return collection
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=str(vector_dir))
        collection = client.get_or_create_collection("researchbuddy_chunks")
        return collection
    except Exception:
        return None


def clean_text(value):
    value = re.sub(r"-\n", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_pdf(path):
    if fitz is None:
        return [{"page": 1, "text": ""}]
    doc = fitz.open(path)
    pages = []
    for index, page in enumerate(doc):
        pages.append({"page": index + 1, "text": clean_text(page.get_text("text"))})
    doc.close()
    return pages


def detect_sections(pages):
    sections = []
    current = {"name": "Unknown", "page": 1, "text": ""}
    pattern = re.compile(r"(?i)\b(" + "|".join(re.escape(name) for name in section_names) + r")\b")
    for page in pages:
        text = page["text"]
        matches = list(pattern.finditer(text))
        if not matches:
            current["text"] += " " + text
            continue
        start = 0
        for match in matches:
            before = text[start:match.start()].strip()
            if before:
                current["text"] += " " + before
            if current["text"].strip():
                current["text"] = clean_text(current["text"])
                sections.append(current)
            current = {"name": match.group(1).title(), "page": page["page"], "text": ""}
            start = match.end()
        current["text"] += " " + text[start:]
    if current["text"].strip():
        current["text"] = clean_text(current["text"])
        sections.append(current)
    return merge_sections(sections)


def merge_sections(sections):
    merged = []
    for section in sections:
        existing = next((item for item in merged if item["name"].lower() == section["name"].lower()), None)
        if existing:
            existing["text"] = clean_text(existing["text"] + " " + section["text"])
        else:
            merged.append(section)
    return merged


def chunk_sections(sections, size=900, overlap=140):
    chunks = []
    for section in sections:
        words = section["text"].split()
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + size]).strip()
            if part:
                chunks.append({
                    "id": str(uuid4()),
                    "text": part,
                    "section": section["name"],
                    "page": section["page"],
                })
    return chunks


def fallback_embedding(text, dimensions=384):
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        pos = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1 if digest[2] % 2 == 0 else -1
        values[pos] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def embed_texts(texts):
    loaded = get_model()
    if loaded:
        return loaded.encode(texts).tolist()
    return [fallback_embedding(text) for text in texts]


def cosine(left, right):
    total = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return total / (left_norm * right_norm)


def index_chunks(paper_id, chunks):
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    store = get_collection()
    ids = []
    metas = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        ids.append(chunk["id"])
        metas.append({"paperId": paper_id, "section": chunk["section"], "page": chunk["page"]})
    if store is not None and ids:
        try:
            store.add(
                ids=ids,
                documents=[chunk["text"] for chunk in chunks],
                embeddings=embeddings,
                metadatas=metas,
            )
            return
        except Exception:
            pass
    memory_vectors[paper_id] = chunks


def retrieve(paper, query, top_k=5):
    store = get_collection()
    query_embedding = embed_texts([query])[0]
    if store is not None:
        try:
            results = store.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"paperId": paper["id"]},
            )
            found = []
            for index, text in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][index]
                found.append({
                    "text": text,
                    "section": meta.get("section", "Unknown"),
                    "page": meta.get("page", 1),
                    "score": 1.0,
                })
            if found:
                return found
        except Exception:
            pass
    chunks = paper.get("chunks", [])
    ranked = []
    for chunk in chunks:
        embedding = chunk.get("embedding") or fallback_embedding(chunk["text"])
        ranked.append((cosine(query_embedding, embedding), chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "text": chunk["text"],
            "section": chunk["section"],
            "page": chunk["page"],
            "score": round(score, 4),
        }
        for score, chunk in ranked[:top_k]
    ]


def sentences(text):
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if len(item.strip()) > 40]


def pick_sentences(chunks, keywords, limit=4):
    selected = []
    terms = [term.lower() for term in keywords]
    for chunk in chunks:
        for sentence in sentences(chunk["text"]):
            lowered = sentence.lower()
            if any(term in lowered for term in terms):
                selected.append({
                    "text": sentence,
                    "section": chunk["section"],
                    "page": chunk["page"],
                })
            if len(selected) >= limit:
                return selected
    if not selected:
        for chunk in chunks[:limit]:
            text = sentences(chunk["text"])
            if text:
                selected.append({"text": text[0], "section": chunk["section"], "page": chunk["page"]})
    return selected[:limit]


def cite(item):
    return {"section": item.get("section", "Unknown"), "page": item.get("page", 1)}


def build_summary(paper):
    chunks = paper.get("chunks", [])
    contributions = pick_sentences(chunks, ["propose", "present", "introduce", "contribution", "show"], 5)
    limitations = pick_sentences(chunks, ["limitation", "future", "fail", "cannot", "however"], 4)
    datasets = pick_sentences(chunks, ["dataset", "benchmark", "corpus", "accuracy", "f1", "bleu", "rouge", "metric"], 5)
    methods = pick_sentences(chunks, ["method", "model", "approach", "training", "architecture"], 4)
    future = pick_sentences(chunks, ["future work", "future", "extend", "further"], 3)
    claims = pick_sentences(chunks, ["outperform", "improve", "achieve", "state-of-the-art", "significant"], 5)
    overview_items = pick_sentences(chunks, ["abstract", "paper", "study", "propose", "result"], 3)
    return {
        "title": paper["title"],
        "overview": [{"text": item["text"], "source": cite(item)} for item in overview_items],
        "keyContributions": [{"text": item["text"], "source": cite(item)} for item in contributions],
        "claims": [{"text": item["text"], "source": cite(item)} for item in claims],
        "methodology": [{"text": item["text"], "source": cite(item)} for item in methods],
        "limitations": [{"text": item["text"], "source": cite(item)} for item in limitations],
        "futureWork": [{"text": item["text"], "source": cite(item)} for item in future],
        "datasetsAndMetrics": [{"text": item["text"], "source": cite(item)} for item in datasets],
    }


def answer_question(paper, question):
    found = retrieve(paper, question, 5)
    if not found:
        return {"answer": "No grounded answer was found in the paper.", "sources": []}
    answer = " ".join(sentences(item["text"])[0] if sentences(item["text"]) else item["text"][:260] for item in found[:2])
    return {
        "answer": answer,
        "sources": [{"section": item["section"], "page": item["page"], "score": item["score"]} for item in found],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    paper_id = str(uuid4())
    target = pdf_dir / f"{paper_id}.pdf"
    target.write_bytes(await file.read())
    pages = extract_pdf(target)
    text = clean_text(" ".join(page["text"] for page in pages))
    sections = detect_sections(pages)
    chunks = chunk_sections(sections)
    paper = {
        "id": paper_id,
        "title": Path(file.filename or "paper.pdf").stem,
        "fileName": file.filename,
        "pages": pages,
        "text": text,
        "sections": sections,
        "chunks": chunks,
    }
    index_chunks(paper_id, chunks)
    save_paper(paper)
    return {
        "paperId": paper_id,
        "title": paper["title"],
        "fileName": paper["fileName"],
        "sectionCount": len(sections),
        "chunkCount": len(chunks),
    }


@app.get("/papers/{paper_id}")
def get_paper(paper_id: str):
    paper = load_paper(paper_id)
    return {
        "id": paper["id"],
        "title": paper["title"],
        "fileName": paper["fileName"],
        "sections": [{"name": item["name"], "page": item["page"]} for item in paper.get("sections", [])],
        "chunkCount": len(paper.get("chunks", [])),
    }


@app.get("/papers/{paper_id}/summary")
def summarize_paper(paper_id: str):
    return build_summary(load_paper(paper_id))


@app.get("/papers/{paper_id}/claims")
def claims(paper_id: str):
    paper = load_paper(paper_id)
    return {"claims": build_summary(paper)["claims"]}


@app.get("/papers/{paper_id}/limitations")
def limitations(paper_id: str):
    paper = load_paper(paper_id)
    return {"limitations": build_summary(paper)["limitations"]}


@app.get("/papers/{paper_id}/future-work")
def future_work(paper_id: str):
    paper = load_paper(paper_id)
    return {"futureWork": build_summary(paper)["futureWork"]}


@app.get("/papers/{paper_id}/datasets-metrics")
def datasets_metrics(paper_id: str):
    paper = load_paper(paper_id)
    return {"datasetsAndMetrics": build_summary(paper)["datasetsAndMetrics"]}


@app.post("/papers/{paper_id}/qa")
def qa(paper_id: str, request: QuestionRequest):
    return answer_question(load_paper(paper_id), request.question)


@app.post("/papers/compare")
def compare(request: CompareRequest):
    if len(request.paperIds) != 2:
        raise HTTPException(status_code=400, detail="Provide exactly two paper ids")
    left = load_paper(request.paperIds[0])
    right = load_paper(request.paperIds[1])
    left_summary = build_summary(left)
    right_summary = build_summary(right)
    return {
        "papers": [
            {"id": left["id"], "title": left["title"]},
            {"id": right["id"], "title": right["title"]},
        ],
        "sharedThemes": [
            item for item in ["model", "dataset", "training", "evaluation", "architecture"]
            if item in left["text"].lower() and item in right["text"].lower()
        ],
        "paperA": {
            "contributions": left_summary["keyContributions"][:3],
            "limitations": left_summary["limitations"][:2],
            "datasetsAndMetrics": left_summary["datasetsAndMetrics"][:3],
        },
        "paperB": {
            "contributions": right_summary["keyContributions"][:3],
            "limitations": right_summary["limitations"][:2],
            "datasetsAndMetrics": right_summary["datasetsAndMetrics"][:3],
        },
    }


@app.post("/tools/extract")
async def extract_tool(file: UploadFile = File(...)):
    target = pdf_dir / f"tool-{uuid4()}.pdf"
    target.write_bytes(await file.read())
    pages = extract_pdf(target)
    return {"pages": pages, "sections": detect_sections(pages)}


@app.post("/tools/chunk")
def chunk_tool(request: TextRequest):
    sections = [{"name": "Text", "page": 1, "text": request.text}]
    return {"chunks": chunk_sections(sections)}


@app.post("/tools/embed")
def embed_tool(request: TextRequest):
    return {"embedding": embed_texts([request.text])[0]}


@app.post("/tools/retrieve")
def retrieve_tool(request: RetrievalRequest):
    paper = load_paper(request.paperId)
    return {"results": retrieve(paper, request.query, request.topK)}
