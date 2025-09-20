"""
Document Parser - Main interface for text extraction from various file formats
Combines PDF and DOCX extractors with text normalization
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import re
import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from pdf_extractor import PDFExtractor
from docx_extractor import DOCXExtractor
from text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Main document parser that handles multiple file formats
    and provides unified interface for text extraction
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize document parser with all extractors
        
        Args:
            spacy_model: spaCy model for text processing
        """
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DOCXExtractor()
        self.text_normalizer = TextNormalizer(spacy_model)
        
        self.supported_formats = {'.pdf', '.docx', '.doc'}
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported
        
        Args:
            file_path: Path to file
        
        Returns:
            True if format is supported
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def extract_raw_text(self, file_path: Union[str, Path], 
                        method: str = 'auto') -> Dict[str, Any]:
        """
        Extract raw text from document
        
        Args:
            file_path: Path to document file
            method: Extraction method preference
        
        Returns:
            Dictionary with extraction results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                result = self.pdf_extractor.extract_text(str(file_path), method)
            elif file_extension in ['.docx', '.doc']:
                result = self.docx_extractor.extract_text(str(file_path), method)
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
            
            # Add file metadata
            result['file_path'] = str(file_path)
            result['file_name'] = file_path.name
            result['file_size'] = file_path.stat().st_size
            result['file_extension'] = file_extension
            
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'file_name': file_path.name
            }
    
    def extract_structured_content(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract structured content including tables and metadata
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dictionary with structured content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                result = self.pdf_extractor.extract_structured_content(str(file_path))
            elif file_extension in ['.docx', '.doc']:
                result = self.docx_extractor.extract_structured_content(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
            
            # Add file metadata
            result['file_path'] = str(file_path)
            result['file_name'] = file_path.name
            result['file_size'] = file_path.stat().st_size
            result['file_extension'] = file_extension
            
            return result
            
        except Exception as e:
            logger.error(f"Structured extraction failed for {file_path}: {str(e)}")
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'file_name': file_path.name
            }
    
    def parse_document(self, file_path: Union[str, Path], 
                      include_structured: bool = True,
                      normalize_text: bool = True) -> Dict[str, Any]:
        """
        Complete document parsing with normalization and structure extraction
        
        Args:
            file_path: Path to document file
            include_structured: Whether to extract structured content
            normalize_text: Whether to normalize and process text
        
        Returns:
            Complete parsed document data
        """
        try:
            # Extract raw content
            if include_structured:
                extraction_result = self.extract_structured_content(file_path)
            else:
                extraction_result = self.extract_raw_text(file_path)
            
            if not extraction_result['success']:
                return extraction_result
            
            # Process text if normalization is requested
            if normalize_text and extraction_result['text']:
                processed = self.text_normalizer.process_document(extraction_result['text'])
                extraction_result.update(processed)
            
            # Add parsing metadata
            extraction_result['parsing_successful'] = True
            extraction_result['parsing_method'] = 'structured' if include_structured else 'basic'
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Document parsing failed for {file_path}: {str(e)}")
            return {
                'text': '',
                'success': False,
                'parsing_successful': False,
                'error': str(e),
                'file_path': str(file_path),
                'file_name': Path(file_path).name
            }
    
    def parse_resume(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse resume with specific processing for resume content
        
        Args:
            file_path: Path to resume file
        
        Returns:
            Parsed resume data with extracted sections
        """
        result = self.parse_document(file_path, include_structured=True, normalize_text=True)
        
        if result['success']:
            # Add resume-specific metadata
            result['document_type'] = 'resume'
            
            # Extract specific resume information
            if 'sections' in result:
                sections = result['sections']
                result['resume_sections'] = {
                    'contact_info': sections.get('contact', ''),
                    'summary': sections.get('summary', ''),
                    'experience': sections.get('experience', ''),
                    'education': sections.get('education', ''),
                    'skills': sections.get('skills', ''),
                    'projects': sections.get('projects', ''),
                    'certifications': sections.get('certifications', ''),
                    'achievements': sections.get('achievements', '')
                }
            
            # Extract skills specifically
            if 'entities' in result and 'skills' in result['entities']:
                result['extracted_skills'] = result['entities']['skills']
            
        return result
    
    def parse_job_description(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse job description with specific processing for JD content
        
        Args:
            file_path: Path to job description file
        
        Returns:
            Parsed JD data with extracted requirements
        """
        result = self.parse_document(file_path, include_structured=True, normalize_text=True)
        
        if result['success']:
            # Add JD-specific metadata
            result['document_type'] = 'job_description'
            
            # Extract JD-specific information
            text = result.get('cleaned_text', '')
            
            # Extract requirements patterns
            requirements_patterns = {
                'required_skills': r'(?:required|must have|essential).*?(?:skills?|experience|knowledge)',
                'preferred_skills': r'(?:preferred|nice to have|desired|plus).*?(?:skills?|experience|knowledge)',
                'education': r'(?:degree|education|qualification|bachelor|master|phd)',
                'experience': r'(?:\d+\+?\s*years?.*?experience|experience.*?\d+\+?\s*years?)',
                'responsibilities': r'(?:responsibilities?|duties|role|tasks?)',
                'benefits': r'(?:benefits?|perks|compensation|salary)'
            }
            
            jd_sections = {}
            text_lower = text.lower()
            
            for section, pattern in requirements_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
                jd_sections[section] = matches
            
            result['jd_sections'] = jd_sections
            
            # Extract skills from JD
            if 'entities' in result and 'skills' in result['entities']:
                result['required_skills'] = result['entities']['skills']
        
        return result
    
    def batch_parse(self, file_paths: list, document_type: str = 'auto') -> Dict[str, Any]:
        """
        Parse multiple documents in batch
        
        Args:
            file_paths: List of file paths to parse
            document_type: Type of documents ('resume', 'job_description', 'auto')
        
        Returns:
            Dictionary with results for each file
        """
        results = {
            'successful': [],
            'failed': [],
            'summary': {
                'total': len(file_paths),
                'successful_count': 0,
                'failed_count': 0
            }
        }
        
        for file_path in file_paths:
            try:
                if document_type == 'resume':
                    result = self.parse_resume(file_path)
                elif document_type == 'job_description':
                    result = self.parse_job_description(file_path)
                else:
                    result = self.parse_document(file_path)
                
                if result['success']:
                    results['successful'].append(result)
                    results['summary']['successful_count'] += 1
                else:
                    results['failed'].append(result)
                    results['summary']['failed_count'] += 1
                    
            except Exception as e:
                error_result = {
                    'file_path': str(file_path),
                    'file_name': Path(file_path).name,
                    'success': False,
                    'error': str(e)
                }
                results['failed'].append(error_result)
                results['summary']['failed_count'] += 1
                logger.error(f"Batch parsing failed for {file_path}: {str(e)}")
        
        return results