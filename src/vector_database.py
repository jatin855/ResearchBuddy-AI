import os
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid 
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity

"""
class for vector Store 

"""

class Vector_Store:
    def __init__(self,persistent_Directory: str , collection_name : str = "researchbuddy_chunks"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persistent_directory = persistent_Directory
        self.client = None
        self.collection = None
        self.inizialise()
    def inizialise(self):
        try :
            # creating client of ChromadB
            os.makedirs(self.persistent_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persistent_directory)
            
            # creating collection of vector dB
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"description":f"collection of pdf present in {self.persistent_directory}"}
            )
            print(f"vector store initialized succesfully : {self.collection_name}")
        except Exception as e:
            print(f"Error while inizialising client and collection : {e}")
            raise e
        
    def add_document(self,document : List[Any] , Embedding : np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(document) != len(Embedding):
            raise ValueError("Number of documents and embeddings is not same")
        
        # preperaing data for injecting in VectordB
        
        ids = []
        document_content = []
        document_metadata = []
        embeddings = []
        for i,(doc,embed) in enumerate(zip(document,Embedding)):
            doc_id = f"doc_{uuid.uuid4().hex}_{i}"
            ids.append(doc_id)
            embeddings.append(embed.tolist())
            document_content.append(doc.page_content)
            metadata = dict(doc.metadata)
            metadata["document_index"] = i
            document_metadata.append(metadata)

        # Dumping all the data in collection
        try:
            self.collection.add(
                ids = ids,
                metadatas=document_metadata,
                documents=document_content,
                embeddings=embeddings
            )
            print(f"Successfull {self.collection.count()} documents are stored in vector database")

        except Exception as e: 
            print(f"Error storing data in Vector_dB : {e}")
            raise       

        
        

