"""
Evaluation metrics calculation
"""

import numpy as np
from typing import List, Dict


def calculate_recall_at_k(retrieved_indices: List[int],
                         relevant_indices: List[int],
                         k: int) -> float:
    """Calculate Recall@K"""
    if not relevant_indices:
        return 1.0
    
    retrieved_at_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def calculate_precision_at_k(retrieved_indices: List[int],
                            relevant_indices: List[int],
                            k: int) -> float:
    """Calculate Precision@K"""
    retrieved_at_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    if not retrieved_at_k:
        return 0.0
    
    return len(retrieved_at_k & relevant_set) / len(retrieved_at_k)


def calculate_mrr(retrieved_indices: List[int],
                 relevant_indices: List[int]) -> float:
    """Calculate Mean Reciprocal Rank"""
    relevant_set = set(relevant_indices)
    
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_set:
            return 1.0 / rank
    
    return 0.0


def calculate_faithfulness(answer: str, context: str) -> float:
    """Calculate faithfulness (0-1)"""
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    
    if not answer_tokens:
        return 0.0
    
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)