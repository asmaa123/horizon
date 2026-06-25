"""
PDF document loading and extraction
"""

from pathlib import Path
from typing import List, Tuple
import pdfplumber
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Represents a PDF page"""
    page_num: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)


@dataclass
class Table:
    """Represents an extracted table"""
    page_num: int
    table_num: int
    headers: List[str]
    rows: List[dict]


@dataclass
class PDFDocument:
    """Complete PDF document"""
    path: str
    pages: List[Page]
    tables: List[Table]
    total_pages: int


class PDFLoader:
    """Load and extract content from PDF files"""
    
    def __init__(self, pdf_path: str):
        """Initialize PDF loader"""
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"PDFLoader initialized for: {pdf_path}")
    
    def load(self) -> PDFDocument:
        """Load PDF and extract all content"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                pages = []
                tables = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    page_tables = page.extract_tables() or []
                    
                    pages.append(Page(
                        page_num=page_num,
                        text=text,
                        tables=page_tables
                    ))
                    
                    # Process tables
                    for table_num, table_data in enumerate(page_tables):
                        table = self._process_table(page_num, table_num, table_data)
                        tables.append(table)
                
                logger.info(f"Loaded {len(pages)} pages, {len(tables)} tables")
                
                return PDFDocument(
                    path=str(self.pdf_path),
                    pages=pages,
                    tables=tables,
                    total_pages=len(pdf.pages)
                )
        
        except Exception as e:
            logger.error(f"PDF loading failed: {str(e)}")
            raise
    
    @staticmethod
    def _process_table(page_num: int, table_num: int,
                      table_data: List[List[str]]) -> Table:
        """Convert raw table data to structured format"""
        if not table_data or len(table_data) < 2:
            return Table(page_num, table_num, [], [])
        
        headers = table_data[0]
        rows = []
        
        for row_data in table_data[1:]:
            row_dict = {header: row_data[i] if i < len(row_data) else ""
                       for i, header in enumerate(headers)}
            rows.append(row_dict)
        
        return Table(page_num, table_num, headers, rows)