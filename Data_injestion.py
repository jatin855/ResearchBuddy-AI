import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src import load_document, create_chunks, Embedding_Manager, Vector_Store

def ingest_pdf_file(file_path: str, persistent_directory: str = "vector_dB"):
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    chunks = create_chunks(docs)
    
    embedding_manager = Embedding_Manager()
    documents_content = [i.page_content for i in chunks]
    embeddings = embedding_manager.gen_embeddings(documents_content)
    
    vector_db = Vector_Store(persistent_Directory=persistent_directory)
    vector_db.add_document(chunks, embeddings)
    return docs, chunks, embeddings

def ingest_directory(directory_path: str, persistent_directory: str = "vector_dB"):
    docs = load_document(director_path=directory_path)
    chunks = create_chunks(docs)
    
    embedding_manager = Embedding_Manager()
    documents_content = [i.page_content for i in chunks]
    embeddings = embedding_manager.gen_embeddings(documents_content)
    
    vector_db = Vector_Store(persistent_Directory=persistent_directory)
    vector_db.add_document(chunks, embeddings)
    return docs, chunks, embeddings

if __name__ == "__main__":
    ingest_directory("PDFs", "vector_dB")
