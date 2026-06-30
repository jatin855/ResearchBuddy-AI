import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir.parent))

from Data_injestion import ingest_pdf_file
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import hashlib
import json
import math
import re
import chromadb
import llm_client

# Startup Verification & Configuration Banner
# Print model configuration banner
print("===================================")
print("LLM : Gemini")
print(f"Model : {llm_client.GEMINI_MODEL}")
print("Embeddings : all-MiniLM-L6-v2")
print("Vector DB : ChromaDB")
print("===================================\n")

def safe_print_status(success: bool, message: str):
    symbol = "✓" if success else "✗"
    fallback_symbol = "[OK]" if success else "[ERROR]"
    try:
        print(f"{symbol} {message}\n")
    except UnicodeEncodeError:
        print(f"{fallback_symbol} {message}\n")

# Verify required components
print("Loading Gemini...")
if not llm_client.GEMINI_API_KEY:
    safe_print_status(False, "Missing GEMINI_API_KEY environment variable.")
    sys.exit(1)

try:
    # A simple call to Gemini to verify connectivity
    llm_client.client.models.generate_content(
        model=llm_client.GEMINI_MODEL,
        contents="ping",
    )
    safe_print_status(True, "Gemini OK")
except Exception as e:
    safe_print_status(False, f"Gemini Connection Failed: {e}")
    sys.exit(1)

print("Loading SentenceTransformer...")
try:
    from src import Embedding_Manager
    embedding_manager = Embedding_Manager()
    safe_print_status(True, "all-MiniLM-L6-v2 loaded")
except Exception as e:
    safe_print_status(False, f"Failed to load SentenceTransformer: {e}")
    sys.exit(1)

print("Connecting to ChromaDB...")
try:
    client = chromadb.PersistentClient(path=str(base_dir.parent / "vector_dB"))
    safe_print_status(True, "Connected")
except Exception as e:
    safe_print_status(False, f"Failed to connect to ChromaDB: {e}")
    sys.exit(1)

print("Loading Vector Collection...")
try:
    collection = client.get_or_create_collection("researchbuddy_chunks")
    safe_print_status(True, "researchbuddy_chunks loaded")
except Exception as e:
    safe_print_status(False, f"Failed to load collection: {e}")
    sys.exit(1)

print("AI Service Ready\n")

app = FastAPI(title="ResearchBuddy AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_dir = base_dir / "data"
pdf_dir = data_dir / "pdfs"
paper_dir = data_dir / "papers"
for folder in [pdf_dir, paper_dir]:
    folder.mkdir(parents=True, exist_ok=True)

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

def get_collection():
    return collection

def get_embedding_manager():
    return embedding_manager

def clean_text(value):
    value = re.sub(r"-\n", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

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


def retrieve(paper, query, top_k=5):
    store = get_collection()
    em = get_embedding_manager()
    
    query_embedding = em.gen_embeddings([query])[0]
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
        
    results = store.query(
        query_embeddings=[query_embedding],
        n_results=100,
    )
    found = []
    file_name = paper.get("fileName", "")
    paper_id = paper.get("id", "")
    for index, text in enumerate(results.get("documents", [[]])[0]):
        meta = results.get("metadatas", [[]])[0][index]
        source = meta.get("source", "")
        if (file_name and file_name in source) or (paper_id and paper_id in source):
            found.append({
                "text": text,
                "section": meta.get("section", "Unknown"),
                "page": meta.get("page", 0) + 1,
                "score": 1.0,
            })
    return found[:top_k]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    paper_id = str(uuid4())
    target = pdf_dir / f"{paper_id}.pdf"
    target.write_bytes(await file.read())
    
    docs, chunk_docs, embeddings = ingest_pdf_file(str(target), str(base_dir.parent / "vector_dB"))
    
    pages = []
    for d in docs:
        pages.append({
            "page": d.metadata.get("page", 0) + 1,
            "text": clean_text(d.page_content)
        })
    
    text = clean_text(" ".join(d.page_content for d in docs))
    sections = detect_sections(pages)
    
    json_chunks = []
    for i, (chunk_doc, emb) in enumerate(zip(chunk_docs, embeddings)):
        page_num = chunk_doc.metadata.get("page", 0) + 1
        chunk_text = clean_text(chunk_doc.page_content)
        
        section_name = "General"
        for sec in sections:
            if sec["text"] and chunk_text[:50].lower() in sec["text"].lower():
                section_name = sec["name"]
                break
        
        json_chunks.append({
            "id": f"chunk_{i}_{uuid4()}",
            "text": chunk_text,
            "section": section_name,
            "page": page_num,
            "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb)
        })
        
    paper = {
        "id": paper_id,
        "title": Path(file.filename or "paper.pdf").stem,
        "fileName": file.filename,
        "pages": pages,
        "text": text,
        "sections": sections,
        "chunks": json_chunks,
    }
    
    save_paper(paper)
    
    return {
        "paperId": paper_id,
        "title": paper["title"],
        "fileName": paper["fileName"],
        "sectionCount": len(sections),
        "chunkCount": len(json_chunks),
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
    paper = load_paper(paper_id)
    try:
        return llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/claims")
def claims(paper_id: str):
    paper = load_paper(paper_id)
    try:
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        return {"claims": summary.get("claims", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/limitations")
def limitations(paper_id: str):
    paper = load_paper(paper_id)
    try:
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        return {"limitations": summary.get("limitations", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/future-work")
def future_work(paper_id: str):
    paper = load_paper(paper_id)
    try:
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        return {"futureWork": summary.get("futureWork", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/datasets-metrics")
def datasets_metrics(paper_id: str):
    paper = load_paper(paper_id)
    try:
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        return {"datasetsAndMetrics": summary.get("datasetsAndMetrics", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/papers/{paper_id}/qa")
def qa(paper_id: str, request: QuestionRequest):
    paper = load_paper(paper_id)
    try:
        chunks = retrieve(paper, request.question, 5)
        return llm_client.get_grounded_qa(request.question, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/papers/compare")
def compare(request: CompareRequest):
    if len(request.paperIds) != 2:
        raise HTTPException(status_code=400, detail="Provide exactly two paper ids")
    left = load_paper(request.paperIds[0])
    right = load_paper(request.paperIds[1])
    try:
        left_summary = llm_client.get_grounded_summary(left["title"], left.get("chunks", []))
        right_summary = llm_client.get_grounded_summary(right["title"], right.get("chunks", []))
        comparison = llm_client.get_comparison(left["title"], left_summary, right["title"], right_summary)
        comparison["papers"] = [
            {"id": left["id"], "title": left["title"]},
            {"id": right["id"], "title": right["title"]},
        ]
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/extract")
async def extract_tool(file: UploadFile = File(...)):
    target = pdf_dir / f"tool-{uuid4()}.pdf"
    target.write_bytes(await file.read())
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(str(target))
    docs = loader.load()
    pages = [{"page": d.metadata.get("page", 0) + 1, "text": clean_text(d.page_content)} for d in docs]
    return {"pages": pages, "sections": detect_sections(pages)}

@app.post("/tools/chunk")
def chunk_tool(request: TextRequest):
    from langchain_core.documents import Document
    doc = Document(page_content=request.text)
    from src import create_chunks
    chunks = create_chunks([doc])
    return {"chunks": [{"text": c.page_content} for c in chunks]}

@app.post("/tools/embed")
def embed_tool(request: TextRequest):
    em = get_embedding_manager()
    emb = em.gen_embeddings([request.text])[0]
    return {"embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb)}

@app.post("/tools/retrieve")
def retrieve_tool(request: RetrievalRequest):
    paper = load_paper(request.paperId)
    return {"results": retrieve(paper, request.query, request.topK)}
