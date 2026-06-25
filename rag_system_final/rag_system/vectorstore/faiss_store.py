"""
FAISS vector storage implementation
"""

import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, embedding_dim: int, index_type: str = 'flat'):
        """Initialize vector store"""
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunks = []
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index"""
        try:
            import faiss
            
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == 'ivfflat':
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    min(100, self.embedding_dim),
                    faiss.METRIC_L2
                )
            
            logger.info(f"Created {self.index_type} FAISS index")
        
        except ImportError:
            logger.error("FAISS not installed")
            raise
    
    def add(self, embeddings: np.ndarray, chunks: List):
        """Add embeddings to index"""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks count mismatch")
        
        embeddings = embeddings.astype(np.float32)
        
        # Train IVF index if needed
        if hasattr(self.index, 'train') and self.index.ntotal == 0:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar embeddings"""
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return results
    
    def get_chunk(self, idx: int):
        """Get chunk by index"""
        if 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return None
    
    def save(self, path: str):
        """Save index to disk"""
        import faiss
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save chunks
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index from disk"""
        import faiss
        
        path = Path(path)
        
        # Load index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load chunks
        with open(path.with_suffix('.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded index from {path}")