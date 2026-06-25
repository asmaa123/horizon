"""
Hybrid document chunking strategy
"""

import re
from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk"""
    chunk_id: str
    content: str
    chunk_type: str  # 'text', 'table', 'semantic'
    page_num: int
    section_title: str
    token_count: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridChunker:
    """Hybrid chunking combining semantic and token-based approaches"""
    
    def __init__(self, token_limit_min: int = 300,
                 token_limit_max: int = 800,
                 overlap: int = 75):
        """Initialize chunker"""
        self.token_limit_min = token_limit_min
        self.token_limit_max = token_limit_max
        self.overlap = overlap
        self.chunk_counter = 0
    
    def chunk_document(self, document) -> List[Chunk]:
        """Create chunks from document"""
        chunks = []
        
        # Semantic chunks from text
        semantic_chunks = self._chunk_semantic(document.pages)
        chunks.extend(semantic_chunks)
        logger.info(f"Created {len(semantic_chunks)} semantic chunks")
        
        # Table chunks
        table_chunks = self._chunk_tables(document.tables)
        chunks.extend(table_chunks)
        logger.info(f"Created {len(table_chunks)} table chunks")
        
        # Token-based fallback if needed
        if len(chunks) < 50:
            token_chunks = self._chunk_token_based(document.pages)
            chunks.extend(token_chunks)
            logger.info(f"Created {len(token_chunks)} token-based chunks")
        
        return chunks
    
    def _chunk_semantic(self, pages) -> List[Chunk]:
        """Create semantic chunks from document sections"""
        chunks = []
        
        for page in pages:
            sections = self._extract_sections(page.text)
            
            for section in sections:
                content = section['content'].strip()
                if len(content) < 50:
                    continue
                
                chunk = Chunk(
                    chunk_id=f"chunk_{self.chunk_counter:06d}",
                    content=content,
                    chunk_type='semantic',
                    page_num=page.page_num,
                    section_title=section['title'],
                    token_count=len(content.split()),
                    metadata={'source': 'semantic'}
                )
                chunks.append(chunk)
                self.chunk_counter += 1
        
        return chunks
    
    def _chunk_tables(self, tables) -> List[Chunk]:
        """Create chunks for each table"""
        chunks = []
        
        for table in tables:
            # Convert table to text format
            content = self._table_to_text(table)
            
            chunk = Chunk(
                chunk_id=f"chunk_{self.chunk_counter:06d}",
                content=content,
                chunk_type='table',
                page_num=table.page_num,
                section_title=f"Table {table.table_num}",
                token_count=len(content.split()),
                metadata={'table_num': table.table_num, 'source': 'table'}
            )
            chunks.append(chunk)
            self.chunk_counter += 1
        
        return chunks
    
    def _chunk_token_based(self, pages) -> List[Chunk]:
        """Token-based chunking as fallback"""
        chunks = []
        
        for page in pages:
            tokens = page.text.split()
            
            for i in range(0, len(tokens), self.token_limit_max - self.overlap):
                chunk_tokens = tokens[i:i + self.token_limit_max]
                content = ' '.join(chunk_tokens)
                
                if len(content) < self.token_limit_min:
                    continue
                
                chunk = Chunk(
                    chunk_id=f"chunk_{self.chunk_counter:06d}",
                    content=content,
                    chunk_type='text',
                    page_num=page.page_num,
                    section_title='General Content',
                    token_count=len(chunk_tokens),
                    metadata={'source': 'token_based'}
                )
                chunks.append(chunk)
                self.chunk_counter += 1
        
        return chunks
    
    @staticmethod
    def _extract_sections(text: str) -> List[dict]:
        """Extract semantic sections from text"""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a section header
            if re.match(r'^[A-Z][A-Z\s]+$', line.strip()) or \
               re.match(r'SECTION \d+:', line.strip()):
                
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content)
                    })
                
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content)
            })
        
        return sections
    
    @staticmethod
    def _table_to_text(table) -> str:
        """Convert table to readable text format"""
        lines = []
        
        # Headers
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("|" + "|".join(["-" * 10] * len(table.headers)) + "|")
        
        # Rows
        for row in table.rows:
            values = [str(row.get(h, "")) for h in table.headers]
            lines.append("| " + " | ".join(values) + " |")
        
        return "\n".join(lines)