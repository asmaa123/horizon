"""
RAG System Configuration
Configuration settings for the RAG system
"""

import os
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Global configuration settings"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    INDEX_DIR = DATA_DIR / "index"
    
    # API Keys (set in environment or .env file)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-ada-002"  # or "gemini-embedding" for Google
    EMBEDDING_DIM = 1536  # OpenAI ada-002 dimension
    
    # LLM settings
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 500
    
    # Vector store settings
    INDEX_TYPE = "flat"  # or "ivf" for approximate search
    INDEX_PATH = INDEX_DIR / "faiss_index.bin"


class RAGConfig:
    """RAG system specific configuration"""
    
    def __init__(
        self,
        pdf_path: str = None,
        embedding_model: str = None,
        chunk_size_min: int = 100,
        chunk_size_max: int = 500,
        chunk_overlap: int = 50,
        embedding_batch_size: int = 100,
        top_k: int = 5,
        use_reranking: bool = True,
        use_mock_llm: bool = False,
        index_type: str = "flat",
        index_path: str = None,
        embedder_type: str = "local"
    ):
        self.pdf_path = pdf_path or str(Config.RAW_DIR / "knowledge_base.pdf")
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size
        self.top_k = top_k
        self.use_reranking = use_reranking
        self.use_mock_llm = use_mock_llm
        self.index_type = index_type
        self.index_path = Path(index_path) if index_path else Config.INDEX_PATH
        self.embedder_type = embedder_type