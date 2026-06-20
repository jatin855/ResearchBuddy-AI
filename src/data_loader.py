import os
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List,Dict,Any,Tuple
""" 
Convertin PDFs files into Langchain document and then spliting those Langchain
documents into Chunks.

"""

def load_document(director_path : str = "../PDFs"):
    try:
        os.makedirs(director_path,exist_ok=True)
        PDF_Loader = DirectoryLoader(
            path = director_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader
        ) 
        pdf_files = PDF_Loader.load()
        print(f"{len(pdf_files)} Documents are convert into Lanchain document successfully")
    except Exception as e:
        print(f"Error while loading documents : {e}")
        raise

def create_chunks(document ,chunk_size: int = 500 , chunk_overlap:int = 200 ):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n","\n"," ",""],
        length_function = len
        )
    chunks = text_splitter.split_documents(document)
    print(f"successfully {len(chunks)} from {len(document)} documents ")
    

    


