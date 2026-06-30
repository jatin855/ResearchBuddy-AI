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

client = None
collection = None

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
    global client, collection
    if collection is not None:
        return collection
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=str(base_dir.parent / "vector_dB"))
        collection = client.get_or_create_collection("pdf_vectordB")
        return collection
    except Exception:
        return None

def get_embedding_manager():
    try:
        from src import Embedding_Manager
        return Embedding_Manager()
    except Exception:
        return None

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

def fallback_embedding(text, dimensions=384):
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        pos = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1 if digest[2] % 2 == 0 else -1
        values[pos] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]

def cosine(left, right):
    total = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return total / (left_norm * right_norm)

def retrieve(paper, query, top_k=5):
    store = get_collection()
    em = get_embedding_manager()
    if em:
        try:
            query_embedding = em.gen_embeddings([query])[0]
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()
        except Exception:
            query_embedding = fallback_embedding(query)
    else:
        query_embedding = fallback_embedding(query)
        
    if store is not None:
        try:
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
            if found:
                return found[:top_k]
        except Exception as e:
            print(f"Chroma retrieval failed: {e}")
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
    if llm_client.is_gemini_available():
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        if summary and "overview" in summary:
            return summary
    return build_summary(paper)

@app.get("/papers/{paper_id}/claims")
def claims(paper_id: str):
    paper = load_paper(paper_id)
    if llm_client.is_gemini_available():
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        if summary and "claims" in summary:
            return {"claims": summary["claims"]}
    return {"claims": build_summary(paper)["claims"]}

@app.get("/papers/{paper_id}/limitations")
def limitations(paper_id: str):
    paper = load_paper(paper_id)
    if llm_client.is_gemini_available():
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        if summary and "limitations" in summary:
            return {"limitations": summary["limitations"]}
    return {"limitations": build_summary(paper)["limitations"]}

@app.get("/papers/{paper_id}/future-work")
def future_work(paper_id: str):
    paper = load_paper(paper_id)
    if llm_client.is_gemini_available():
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        if summary and "futureWork" in summary:
            return {"futureWork": summary["futureWork"]}
    return {"futureWork": build_summary(paper)["futureWork"]}

@app.get("/papers/{paper_id}/datasets-metrics")
def datasets_metrics(paper_id: str):
    paper = load_paper(paper_id)
    if llm_client.is_gemini_available():
        summary = llm_client.get_grounded_summary(paper["title"], paper.get("chunks", []))
        if summary and "datasetsAndMetrics" in summary:
            return {"datasetsAndMetrics": summary["datasetsAndMetrics"]}
    return {"datasetsAndMetrics": build_summary(paper)["datasetsAndMetrics"]}

@app.post("/papers/{paper_id}/qa")
def qa(paper_id: str, request: QuestionRequest):
    paper = load_paper(paper_id)
    if llm_client.is_gemini_available():
        chunks = retrieve(paper, request.question, 5)
        ans = llm_client.get_grounded_qa(request.question, chunks)
        if ans and "answer" in ans:
            return ans
    return answer_question(paper, request.question)

@app.post("/papers/compare")
def compare(request: CompareRequest):
    if len(request.paperIds) != 2:
        raise HTTPException(status_code=400, detail="Provide exactly two paper ids")
    left = load_paper(request.paperIds[0])
    right = load_paper(request.paperIds[1])
    
    left_summary = None
    right_summary = None
    
    if llm_client.is_gemini_available():
        left_summary = llm_client.get_grounded_summary(left["title"], left.get("chunks", []))
        right_summary = llm_client.get_grounded_summary(right["title"], right.get("chunks", []))
        if left_summary and right_summary:
            comparison = llm_client.get_comparison(left["title"], left_summary, right["title"], right_summary)
            if comparison and "paperA" in comparison:
                comparison["papers"] = [
                    {"id": left["id"], "title": left["title"]},
                    {"id": right["id"], "title": right["title"]},
                ]
                return comparison
                
    if not left_summary:
        left_summary = build_summary(left)
    if not right_summary:
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
    if em:
        emb = em.gen_embeddings([request.text])[0]
        return {"embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb)}
    return {"embedding": fallback_embedding(request.text)}

@app.post("/tools/retrieve")
def retrieve_tool(request: RetrievalRequest):
    paper = load_paper(request.paperId)
    return {"results": retrieve(paper, request.query, request.topK)}
