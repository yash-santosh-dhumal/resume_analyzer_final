"""
PDF Text Extraction Module
Handles extraction of text from PDF files using PyMuPDF and pdfplumber
"""

import fitz  # PyMuPDF
import pdfplumber
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDF files using multiple methods for robustness"""
    
    def __init__(self):
        self.methods = ['pymupdf', 'pdfplumber']
    
    def extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fitz)"""
        try:
            text = ""
            doc = fitz.open(file_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            
            doc.close()
            return text.strip()
        
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {file_path}: {str(e)}")
            return ""
    
    def extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {str(e)}")
            return ""
    
    def extract_text(self, file_path: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Extract text from PDF with fallback methods
        
        Args:
            file_path: Path to PDF file
            method: Extraction method ('auto', 'pymupdf', 'pdfplumber')
        
        Returns:
            Dictionary with extracted text and metadata
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        result = {
            'text': '',
            'method_used': '',
            'page_count': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Get page count first
            doc = fitz.open(file_path)
            result['page_count'] = doc.page_count
            doc.close()
            
            if method == 'auto':
                # Try PyMuPDF first, then pdfplumber as fallback
                text = self.extract_text_pymupdf(file_path)
                if text and len(text.strip()) > 10:
                    result['text'] = text
                    result['method_used'] = 'pymupdf'
                    result['success'] = True
                else:
                    text = self.extract_text_pdfplumber(file_path)
                    if text:
                        result['text'] = text
                        result['method_used'] = 'pdfplumber'
                        result['success'] = True
            
            elif method == 'pymupdf':
                text = self.extract_text_pymupdf(file_path)
                result['text'] = text
                result['method_used'] = 'pymupdf'
                result['success'] = bool(text)
            
            elif method == 'pdfplumber':
                text = self.extract_text_pdfplumber(file_path)
                result['text'] = text
                result['method_used'] = 'pdfplumber'
                result['success'] = bool(text)
            
            else:
                raise ValueError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        
        return result
    
    def extract_structured_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structured content from PDF including tables and metadata
        
        Returns:
            Dictionary with text, tables, and document metadata
        """
        result = {
            'text': '',
            'tables': [],
            'metadata': {},
            'success': False,
            'error': None
        }
        
        try:
            # Extract basic text
            text_result = self.extract_text(file_path)
            result['text'] = text_result['text']
            result['success'] = text_result['success']
            
            # Extract metadata using PyMuPDF
            doc = fitz.open(file_path)
            result['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': doc.page_count
            }
            doc.close()
            
            # Extract tables using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables):
                            result['tables'].append({
                                'page': page_num + 1,
                                'table_id': table_num + 1,
                                'data': table
                            })
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Structured PDF extraction failed for {file_path}: {str(e)}")
        
        return result