"""
Text Normalization and Processing Module
Handles cleaning, normalization, and preprocessing of extracted text
"""

import re
import spacy
import nltk
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
import logging

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Normalize and preprocess text for better matching and analysis"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize text normalizer with spaCy model
        
        Args:
            spacy_model: spaCy model name to use
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load spaCy model
        self._load_spacy_model()
        
        # Common resume sections patterns
        self.section_patterns = {
            'contact': r'(contact|phone|email|address|linkedin|github)',
            'summary': r'(summary|objective|profile|about)',
            'experience': r'(experience|work|employment|career|job)',
            'education': r'(education|qualification|degree|university|college)',
            'skills': r'(skills|technical|competencies|proficiencies)',
            'projects': r'(projects|portfolio|work)',
            'certifications': r'(certifications?|certificates?|licenses?)',
            'achievements': r'(achievements?|awards?|honors?|accomplishments?)',
            'languages': r'(languages?|linguistic)'
        }
        
        # Common technical skills keywords
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'figma', 'photoshop']
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            self.nlp = spacy.load(self.spacy_model)
        except OSError:
            logger.warning(f"spaCy model {self.spacy_model} not found. Using basic processing.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\-]{2,}', '-', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        text = re.sub(r'([\.,:;!?])\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from resume/JD text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with extracted sections
        """
        sections = {}
        text_lower = text.lower()
        
        # Split text into lines for section detection
        lines = text.split('\n')
        current_section = 'other'
        section_content = {section: [] for section in self.section_patterns.keys()}
        section_content['other'] = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            line_lower = line_clean.lower()
            
            # Check if line is a section header
            section_detected = False
            for section, pattern in self.section_patterns.items():
                if re.search(pattern, line_lower) and len(line_clean) < 50:
                    current_section = section
                    section_detected = True
                    break
            
            if not section_detected:
                section_content[current_section].append(line_clean)
        
        # Convert lists to strings
        for section, content in section_content.items():
            sections[section] = '\n'.join(content)
        
        return sections
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract technical skills from text
        
        Args:
            text: Input text
        
        Returns:
            List of identified skills
        """
        skills_found = []
        text_lower = text.lower()
        
        # Extract skills from all categories
        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                if skill.lower() in text_lower:
                    skills_found.append(skill)
        
        # Use spaCy for entity recognition if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT'] and len(ent.text) > 2:
                    # Could be a technology/tool name
                    skills_found.append(ent.text)
        
        return list(set(skills_found))  # Remove duplicates
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of entities by type
        """
        entities = {
            'organizations': [],
            'persons': [],
            'locations': [],
            'dates': [],
            'skills': []
        }
        
        if not self.nlp:
            # Fallback to basic pattern matching
            entities['skills'] = self.extract_skills(text)
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
        
        entities['skills'] = self.extract_skills(text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def normalize_text(self, text: str, remove_stopwords: bool = True, 
                      lemmatize: bool = True) -> str:
        """
        Comprehensive text normalization
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Clean text first
        text = self.clean_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords if requested
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            if self.nlp:
                # Use spaCy for lemmatization
                doc = self.nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc]
            else:
                # Fallback to NLTK
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Args:
            text: Raw document text
        
        Returns:
            Dictionary with processed content
        """
        result = {
            'raw_text': text,
            'cleaned_text': self.clean_text(text),
            'normalized_text': self.normalize_text(text),
            'sections': self.extract_sections(text),
            'entities': self.extract_entities(text),
            'word_count': 0,
            'sentence_count': 0
        }
        
        # Add statistics
        if result['cleaned_text']:
            result['word_count'] = len(word_tokenize(result['cleaned_text']))
            result['sentence_count'] = len(sent_tokenize(result['cleaned_text']))
        
        return result