import time
import hashlib
import numpy as np
from typing import List
from utils.config import Config, logger


class MockEmbedder:
    """Mock embedder that generates deterministic embeddings without API calls"""
    
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        logger.info("Using Mock Embedder (offline mode)")
    
    def _hash_to_vector(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash"""
        # Create hash of text
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to float values
        vector = []
        for i in range(0, len(hash_hex), 2):
            hex_pair = hash_hex[i:i+2]
            val = int(hex_pair, 16) / 255.0
            vector.append(val)
        
        # Pad or truncate to desired dimension
        if len(vector) < self.embedding_dim:
            vector.extend([0.0] * (self.embedding_dim - len(vector)))
        else:
            vector = vector[:self.embedding_dim]
        
        return vector
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return np.array(self._hash_to_vector(text), dtype=np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        logger.info(f"Generating mock embeddings for {len(texts)} texts...")
        embeddings = [self._hash_to_vector(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @embedding_dim.setter
    def embedding_dim(self, value: int):
        self._embedding_dim = value


class OpenAIEmbedder:
    """OpenAI API embedder"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or Config.OPENAI_API_KEY
        self._embedding_dim = 1536  # Default for ada-002
        self.client = None
        self._init_client()
        
    def _init_client(self):
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbedder")
        try:
            import openai
            self.client = openai.Client(api_key=self.api_key)
            logger.info(f"OpenAIEmbedder client initialized for {self.model_name}")
        except ImportError:
            logger.error("openai package not installed")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        logger.info(f"Generating OpenAI embeddings for {len(texts)} texts...")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            embeddings.extend([item.embedding for item in response.data])
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class LocalEmbedder:
    """Local embedder using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._embedding_dim = 384  # Default for all-MiniLM-L6-v2, will be updated on load
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding model loaded. Dimension: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        logger.info(f"Generating local embeddings for {len(texts)} texts...")
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def get_embedder(model_name: str = None, embedder_type: str = 'local'):
    """Factory function to get embedder"""
    if embedder_type == 'openai' and Config.OPENAI_API_KEY:
        try:
            return OpenAIEmbedder(model_name=model_name or "text-embedding-ada-002")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbedder: {e}. Falling back to local/mock.")
    
    if embedder_type == 'local' or (embedder_type == 'openai' and not Config.OPENAI_API_KEY):
        try:
            # Default local model
            local_model = model_name if model_name and model_name != "text-embedding-ada-002" else 'all-MiniLM-L6-v2'
            return LocalEmbedder(model_name=local_model)
        except Exception as e:
            logger.error(f"Failed to initialize LocalEmbedder: {e}. Falling back to mock embedder.")
            
    return MockEmbedder(embedding_dim=Config.EMBEDDING_DIM)
