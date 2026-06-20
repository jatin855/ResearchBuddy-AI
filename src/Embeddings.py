import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List,Dict,Tuple,Any
from sklearn.metrics.pairwise import cosine_similarity
"""
Class for objects to create embedding from texts

"""
class Embedding_Manager:
    def __init__(self,model_name: str = "all-MiniLM-L6-v2"):
        """
        Args: 
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self.inizialize()
    def inizialize(self):
        """
        inizialize the model for embedding

        """
        try:
            print(f"loading {self.model_name} model")
            self.model = SentenceTransformer(self.model_name)
            print(f"{self.model_name} has loaded Successfully")
            print(f"Embedding dimension : {self.model.get_embedding_dimension()}")

        except Exception as e :
            print(f"Error While loading model : {e}")
            raise
    
    def gen_embeddings(self,Texts : List[str]) -> np.ndarray :
        """
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)

        """
        if self.model:
            print("model is not loaded")
            raise 
        try:
            print(f"{len(Texts)} Texts detected")
            print(f"generating Embeddings for {len(Texts)}")
            embeddings = self.model.encode(Texts,show_progress_bar=True)
            print(f"Embeddings Created : {embeddings.shape}")
        except Exception as e:
            print(f"Error while generating Embeddings : {e}")
            raise
    

