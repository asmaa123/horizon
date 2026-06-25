"""
RAG System - Main Entry Point
Production-ready RAG application for document retrieval and QA
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_loader import PDFLoader
from processing.chunking import HybridChunker
from embeddings.embedder import get_embedder
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm import get_llm
from evaluation.evaluator import RAGEvaluator
from utils.config import RAGConfig
from utils.helpers import setup_logging, print_header

logger = setup_logging(__name__)


class RAGSystem:
    """Main RAG System Orchestrator"""
    
    def __init__(self, config: RAGConfig = None):
        """Initialize RAG system with configuration"""
        self.config = config or RAGConfig()
        
        logger.info("Initializing RAG System...")
        
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.evaluator = None
        self.chunks = []
        
        self._build_system()
    
    def _build_system(self):
        """Build all system components"""
        try:
            print_header("BUILDING RAG SYSTEM")
            
            # 1. Initialize embedder
            logger.info("[1/7] Loading embedding model...")
            self.embedder = get_embedder(
                model_name=self.config.embedding_model,
                embedder_type=self.config.embedder_type
            )
            
            # 2. Load PDF
            logger.info("[2/7] Loading PDF document...")
            loader = PDFLoader(self.config.pdf_path)
            document = loader.load()
            logger.info(f"   Loaded {len(document.pages)} pages, {len(document.tables)} tables")
            
            # 3. Chunk document
            logger.info("[3/7] Chunking document...")
            chunker = HybridChunker(
                token_limit_min=self.config.chunk_size_min,
                token_limit_max=self.config.chunk_size_max,
                overlap=self.config.chunk_overlap
            )
            self.chunks = chunker.chunk_document(document)
            logger.info(f"   Created {len(self.chunks)} chunks")
            
            # 4. Embed chunks
            logger.info("[4/7] Embedding chunks...")
            chunk_texts = [chunk.content for chunk in self.chunks]
            embeddings = self.embedder.embed_batch(
                chunk_texts,
                batch_size=self.config.embedding_batch_size
            )
            logger.info(f"   Embedding shape: {embeddings.shape}")
            
            # 5. Create vector store
            logger.info("[5/7] Creating FAISS vector store...")
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedder.embedding_dim,
                index_type=self.config.index_type
            )
            self.vector_store.add(embeddings, self.chunks)
            
            # 6. Initialize retriever
            logger.info("[6/7] Initializing retriever...")
            reranker = Reranker() if self.config.use_reranking else None
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                chunks=self.chunks,
                reranker=reranker,
                top_k=self.config.top_k
            )
            
            # 7. Initialize LLM
            logger.info("[7/7] Initializing LLM...")
            self.llm = get_llm(use_mock=self.config.use_mock_llm)
            
            # Initialize evaluator
            self.evaluator = RAGEvaluator(
                retriever=self.retriever,
                llm=self.llm,
                embedder=self.embedder
            )
            
            print_header("SYSTEM READY", char="✓")
            logger.info("System initialization complete!")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise
    
    def query(self, question: str, k: int = None, use_reranking: bool = True) -> Dict:
        """
        Answer a question using the RAG system
        
        Args:
            question: User query
            k: Number of results (uses config default if None)
            use_reranking: Apply reranking
            
        Returns:
            Dictionary with answer and sources
        """
        k = k or self.config.top_k
        
        try:
            logger.info(f"Processing query: {question[:80]}...")
            
            # Retrieve
            retrieval_result = self.retriever.retrieve(
                question=question,
                k=k,
                use_reranking=use_reranking
            )
            
            # Generate
            answer = self.llm.generate(
                question=question,
                context_chunks=retrieval_result['chunks']
            )
            
            return {
                'question': question,
                'answer': answer,
                'sources': [
                    {
                        'chunk_id': chunk.chunk_id,
                        'page': chunk.page_num,
                        'section': chunk.section_title,
                        'score': score
                    }
                    for chunk, score in zip(
                        retrieval_result['chunks'],
                        retrieval_result['scores']
                    )
                ],
                'retrieval_time': retrieval_result['time']
            }
        
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def interactive_mode(self):
        """Run interactive Q&A mode"""
        print_header("INTERACTIVE MODE")
        print("\nEnter your questions (type 'exit' to quit):\n")
        
        while True:
            try:
                question = input("\n❓ Q: ").strip()
                
                if question.lower() == 'exit':
                    print("\n👋 Exiting...\n")
                    break
                
                if not question:
                    continue
                
                result = self.query(question)
                
                print(f"\n✅ A: {result['answer']}\n")
                print(f"📍 Sources ({len(result['sources'])} results):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. Page {source['page']}: {source['section']} (score: {source['score']:.3f})")
                
                print(f"⏱️ Retrieval time: {result['retrieval_time']:.3f}s")
            
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def batch_evaluate(self, test_cases: List[Dict]):
        """Evaluate system on test cases"""
        print_header("EVALUATION")
        
        for test_case in test_cases:
            self.evaluator.add_test_case(**test_case)
        
        results = self.evaluator.evaluate()
        
        print("\n📊 Results:")
        print(f"  Recall@3: {results['recall_at_3']:.3f}")
        print(f"  Recall@5: {results['recall_at_5']:.3f}")
        print(f"  MRR: {results['mrr']:.3f}")
        print(f"  Faithfulness: {results['faithfulness']:.3f}\n")
        
        return results
    
    def save(self, path: str = None):
        """Save system to disk"""
        path = path or str(self.config.index_path)
        self.vector_store.save(path)
        logger.info(f"System saved to {path}")
    
    def load(self, path: str = None):
        """Load system from disk"""
        path = path or str(self.config.index_path)
        self.vector_store.load(path)
        logger.info(f"System loaded from {path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG System for document QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive
  python main.py --query "What is the price?"
  python main.py --pdf data/raw/document.pdf --interactive
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        default='data/raw/knowledge_base.pdf',
        help='Path to PDF file'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM (for testing without API key)'
    )
    
    parser.add_argument(
        '--embedder-type',
        type=str,
        default='local',
        choices=['local', 'openai', 'mock'],
        help='Embedder type (local, openai, mock)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RAGConfig(
        pdf_path=args.pdf,
        use_mock_llm=args.mock,
        embedder_type=args.embedder_type
    )
    
    # Initialize system
    try:
        rag = RAGSystem(config)
        
        if args.interactive:
            rag.interactive_mode()
        elif args.query:
            result = rag.query(args.query)
            
            print_header("RESULT")
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}\n")
            
            print(f"📍 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. Page {source['page']}: {source['section']}")
            
            print(f"\n⏱️ Time: {result['retrieval_time']:.3f}s\n")
        else:
            # Demo mode
            demo_queries = [
                "What is the price of the Cairo package?",
                "What is the cancellation policy?",
            ]
            
            print_header("DEMO MODE")
            
            for query in demo_queries:
                print(f"\n❓ Q: {query}")
                try:
                    result = rag.query(query)
                    print(f"✅ A: {result['answer'][:200]}...\n")
                except Exception as e:
                    print(f"❌ Error: {str(e)}\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"\nPlease ensure the PDF file exists at: {config.pdf_path}")
        print("You can change the path with: python main.py --pdf <path>\n")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}\n")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())