"""
Cross-encoder based reranking
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker"""
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        """Initialize reranker"""
        self.model_name = model_name
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load reranker model"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {self.model_name}...")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker loaded")
        
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
    
    def rerank(self, query: str, chunks: List,
              scores: List[float] = None) -> Tuple[List, List[float]]:
        """Rerank chunks"""
        if not self.model or not chunks:
            return chunks, scores or [1.0] * len(chunks)
        
        try:
            # Prepare pairs
            passages = [chunk.content for chunk in chunks]
            pairs = [[query, passage] for passage in passages]
            
            # Get scores
            reranked_scores = self.model.predict(pairs)
            
            # Sort by score
            sorted_items = sorted(
                zip(chunks, reranked_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_chunks = [item[0] for item in sorted_items]
            reranked_scores_list = [float(item[1]) for item in sorted_items]
            
            return reranked_chunks, reranked_scores_list
        
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return chunks, scores or [1.0] * len(chunks)