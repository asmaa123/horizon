"""
LLM response generation
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM response generator"""
    
    SYSTEM_PROMPT = """You are an expert assistant answering questions based on provided documents.
Answer questions ONLY using the provided context.
If the answer is not in the context, respond: "Not found in the provided documents"
Be concise and helpful."""
    
    def __init__(self, model: str = 'gpt-3.5-turbo',
                 api_key: str = None,
                 max_tokens: int = 500):
        """Initialize LLM generator"""
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.max_tokens = max_tokens
        self.client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.warning("No OpenAI API key found")
            return
        
        try:
            import openai
            self.client = openai.Client(api_key=self.api_key)
            logger.info(f"OpenAI client initialized for {self.model}")
        except ImportError:
            logger.warning("openai not installed")
    
    def generate(self, question: str, context_chunks: List) -> str:
        """Generate answer"""
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Build prompt
        user_prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""
        
        if not self.client:
            return self._generate_fallback(question, context_chunks)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.2
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return self._generate_fallback(question, context_chunks)
    
    @staticmethod
    def _format_context(chunks: List) -> str:
        """Format context chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}: Page {chunk.page_num}]"
            content = chunk.content[:500]
            context_parts.append(f"{header}\n{content}")
        
        return "\n\n".join(context_parts)
    
    @staticmethod
    def _generate_fallback(question: str, chunks: List) -> str:
        """Generate fallback response"""
        if not chunks:
            return "Not found in the provided documents"
        
        response = "Based on the provided context:\n\n"
        for chunk in chunks[:2]:
            response += f"- {chunk.section_title}: {chunk.content[:200]}...\n"
        
        return response


class MockLLMGenerator(LLMGenerator):
    """Mock LLM for testing"""
    
    def generate(self, question: str, context_chunks: List) -> str:
        """Generate mock response"""
        return self._generate_fallback(question, context_chunks)


def get_llm(model: str = 'gpt-3.5-turbo',
            use_mock: bool = False,
            max_tokens: int = 500) -> LLMGenerator:
    """Get LLM generator instance"""
    if use_mock:
        logger.info("Using MockLLMGenerator")
        return MockLLMGenerator()
    
    return LLMGenerator(model=model, max_tokens=max_tokens)