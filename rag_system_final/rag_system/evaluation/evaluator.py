"""
RAG system evaluation
"""

from typing import List, Dict
import logging
import numpy as np
from evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_faithfulness
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, retriever, llm, embedder):
        """Initialize evaluator"""
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.test_cases = []
    
    def add_test_case(self, question: str, answer: str,
                     relevant_chunk_ids: List[str]):
        """Add a test case"""
        self.test_cases.append({
            'question': question,
            'answer': answer,
            'relevant_chunks': relevant_chunk_ids
        })
    
    def evaluate(self) -> Dict:
        """Run evaluation"""
        metrics = {
            'recall_at_3': [],
            'recall_at_5': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'mrr': [],
            'faithfulness': []
        }
        
        for test_case in self.test_cases:
            question = test_case['question']
            expected_answer = test_case['answer']
            relevant_ids = test_case['relevant_chunks']
            
            # Retrieve
            result = self.retriever.retrieve(question, k=5, use_reranking=True)
            retrieved_ids = [chunk.chunk_id for chunk in result['chunks']]
            
            # Calculate retrieval metrics
            recall_3 = calculate_recall_at_k(retrieved_ids, relevant_ids, 3)
            recall_5 = calculate_recall_at_k(retrieved_ids, relevant_ids, 5)
            precision_3 = calculate_precision_at_k(retrieved_ids, relevant_ids, 3)
            precision_5 = calculate_precision_at_k(retrieved_ids, relevant_ids, 5)
            mrr = calculate_mrr(retrieved_ids, relevant_ids)
            
            metrics['recall_at_3'].append(recall_3)
            metrics['recall_at_5'].append(recall_5)
            metrics['precision_at_3'].append(precision_3)
            metrics['precision_at_5'].append(precision_5)
            metrics['mrr'].append(mrr)
            
            # Generate and evaluate
            answer = self.llm.generate(question, result['chunks'])
            context_text = ' '.join([c.content for c in result['chunks']])
            faith = calculate_faithfulness(answer, context_text)
            metrics['faithfulness'].append(faith)
        
        # Average metrics
        return {
            'recall_at_3': np.mean(metrics['recall_at_3']),
            'recall_at_5': np.mean(metrics['recall_at_5']),
            'precision_at_3': np.mean(metrics['precision_at_3']),
            'precision_at_5': np.mean(metrics['precision_at_5']),
            'mrr': np.mean(metrics['mrr']),
            'faithfulness': np.mean(metrics['faithfulness'])
        }