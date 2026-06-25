"""
Retrieval pipeline combining vector and lexical search
"""

from typing import List, Dict, Tuple
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based lexical retrieval"""
    
    def __init__(self, documents: List[str]):
        """Initialize BM25 retriever"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search using BM25 (TF-IDF)"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(-similarities)[:k]
        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]
        
        return results


class HybridRetriever:
    """Hybrid retriever combining vector and lexical search"""
    
    def __init__(self, vector_store, embedder, chunks,
                 reranker=None, top_k: int = 5):
        """Initialize retriever"""
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks
        self.reranker = reranker
        self.top_k = top_k
        
        # Initialize BM25
        chunk_texts = [chunk.content for chunk in chunks]
        self.bm25 = BM25Retriever(chunk_texts)
        
        logger.info("HybridRetriever initialized")
    
    def retrieve(self, question: str, k: int = None,
                use_reranking: bool = True) -> Dict:
        """Retrieve relevant chunks"""
        k = k or self.top_k
        start_time = time.time()
        
        # Vector search
        query_emb = self.embedder.embed_text(question)
        vector_results = self.vector_store.search(query_emb, k=k*2)
        
        # BM25 search
        bm25_results = self.bm25.search(question, k=k*2)
        
        # Combine results
        combined = {}
        for idx, dist in vector_results:
            combined[idx] = {'vector': 1 - (dist / 10)}
        
        for idx, score in bm25_results:
            if idx not in combined:
                combined[idx] = {}
            combined[idx]['bm25'] = score
        
        # Average scores
        results = []
        for idx, scores in combined.items():
            avg_score = 0
            if 'vector' in scores:
                avg_score += scores['vector'] * 0.5
            if 'bm25' in scores:
                avg_score += scores['bm25'] * 0.5
            results.append((idx, avg_score))
        
        # Sort and get top-k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:k]
        
        # Get chunks
        chunks = [self.chunks[idx] for idx, _ in top_results]
        scores = [score for _, score in top_results]
        
        # Rerank if enabled
        if use_reranking and self.reranker:
            chunks, scores = self.reranker.rerank(question, chunks, scores)
        
        elapsed = time.time() - start_time
        
        return {
            'chunks': chunks,
            'scores': scores,
            'time': elapsed
        }