"""
Soft Matching Engine
Implements semantic similarity matching using embeddings and vector databases
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import logging
import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from embedding_generator import EmbeddingGenerator
from vector_database import VectorDBManager

logger = logging.getLogger(__name__)

class SoftMatcher:
    """
    Semantic similarity matcher using embeddings and vector databases
    """
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 vector_db_type: str = 'chroma',
                 persist_directory: str = './data/vector_db'):
        """
        Initialize soft matcher
        
        Args:
            embedding_model: Sentence transformer model name
            vector_db_type: Vector database type ('chroma', 'faiss')
            persist_directory: Directory for persistent storage
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Initialize vector database
        db_config = {
            'persist_directory': persist_directory,
            'dimension': self.embedding_generator.embedding_dimension
        }
        
        if vector_db_type == 'chroma':
            db_config['collection_name'] = 'resume_jd_soft_matching'
        elif vector_db_type == 'faiss':
            db_config['index_type'] = 'FlatIP'  # Inner product for cosine similarity
        
        self.vector_db = VectorDBManager(vector_db_type, **db_config)
        
        # Section weights for document embedding
        self.section_weights = {
            'summary': 0.20,
            'experience': 0.30,
            'skills': 0.25,
            'education': 0.15,
            'projects': 0.10
        }
    
    def analyze_semantic_similarity(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Analyze semantic similarity between resume and job description
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
        
        Returns:
            Comprehensive semantic similarity analysis
        """
        results = {
            'overall_similarity': 0.0,
            'section_similarities': {},
            'semantic_features': {},
            'detailed_analysis': {}
        }
        
        try:
            # 1. Overall document similarity
            resume_embedding = self.embedding_generator.encode_text(resume_text)
            jd_embedding = self.embedding_generator.encode_text(jd_text)
            
            overall_similarity = self.embedding_generator.calculate_similarity(
                resume_embedding, jd_embedding
            )
            results['overall_similarity'] = round(overall_similarity * 100, 2)
            
            # 2. Section-wise similarity analysis
            resume_sections = self._extract_document_sections(resume_text, 'resume')
            jd_sections = self._extract_document_sections(jd_text, 'job_description')
            
            section_similarities = self._calculate_section_similarities(
                resume_sections, jd_sections
            )
            results['section_similarities'] = section_similarities
            
            # 3. Weighted section similarity
            weighted_similarity = self._calculate_weighted_similarity(section_similarities)
            results['weighted_section_similarity'] = round(weighted_similarity, 2)
            
            # 4. Semantic feature analysis
            semantic_features = self._analyze_semantic_features(
                resume_embedding, jd_embedding, resume_text, jd_text
            )
            results['semantic_features'] = semantic_features
            
            # 5. Context-aware similarity
            context_similarity = self._calculate_context_similarity(
                resume_sections, jd_sections
            )
            results['context_similarity'] = context_similarity
            
            # 6. Final combined score
            combined_score = (
                results['overall_similarity'] * 0.3 +
                results['weighted_section_similarity'] * 0.4 +
                context_similarity['overall_score'] * 0.3
            )
            results['combined_semantic_score'] = round(combined_score, 2)
            
            # 7. Generate semantic insights
            results['insights'] = self._generate_semantic_insights(results)
            
        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _extract_document_sections(self, text: str, doc_type: str) -> Dict[str, str]:
        """Extract and structure document sections"""
        # Use simple section extraction for now
        # In a real implementation, this could use the text normalizer
        sections = {
            'summary': '',
            'experience': '',
            'skills': '',
            'education': '',
            'projects': ''
        }
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        current_section = 'summary'
        section_content = {section: [] for section in sections.keys()}
        
        # Simple section detection based on keywords
        section_keywords = {
            'summary': ['summary', 'objective', 'profile', 'about'],
            'experience': ['experience', 'work', 'employment', 'career'],
            'skills': ['skills', 'technical', 'competencies'],
            'education': ['education', 'qualification', 'degree'],
            'projects': ['projects', 'portfolio', 'work samples']
        }
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            line_lower = line_clean.lower()
            
            # Check if line indicates a new section
            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords) and len(line_clean) < 100:
                    current_section = section
                    break
            else:
                section_content[current_section].append(line_clean)
        
        # Convert to strings
        for section in sections:
            sections[section] = '\n'.join(section_content[section])
        
        return sections
    
    def _calculate_section_similarities(self, resume_sections: Dict[str, str], 
                                      jd_sections: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Calculate similarity for each section pair"""
        similarities = {}
        
        for section in resume_sections.keys():
            if resume_sections[section] and jd_sections[section]:
                # Encode sections
                resume_emb = self.embedding_generator.encode_text(resume_sections[section])
                jd_emb = self.embedding_generator.encode_text(jd_sections[section])
                
                # Calculate similarity
                similarity = self.embedding_generator.calculate_similarity(resume_emb, jd_emb)
                
                similarities[section] = {
                    'similarity': round(similarity * 100, 2),
                    'resume_length': len(resume_sections[section]),
                    'jd_length': len(jd_sections[section])
                }
            else:
                similarities[section] = {
                    'similarity': 0.0,
                    'resume_length': len(resume_sections[section]),
                    'jd_length': len(jd_sections[section])
                }
        
        return similarities
    
    def _calculate_weighted_similarity(self, section_similarities: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted average of section similarities"""
        total_weight = 0
        weighted_sum = 0
        
        for section, weight in self.section_weights.items():
            if section in section_similarities:
                similarity = section_similarities[section]['similarity']
                weighted_sum += similarity * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _analyze_semantic_features(self, resume_emb: np.ndarray, jd_emb: np.ndarray,
                                 resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Analyze semantic features of both documents"""
        features = {
            'embedding_dimensions': {
                'resume': resume_emb.shape,
                'jd': jd_emb.shape
            },
            'content_analysis': {},
            'similarity_metrics': {}
        }
        
        # Content analysis
        features['content_analysis'] = {
            'resume_word_count': len(resume_text.split()),
            'jd_word_count': len(jd_text.split()),
            'length_ratio': len(resume_text) / len(jd_text) if jd_text else 0
        }
        
        # Different similarity metrics
        similarities = {}
        similarities['cosine'] = self.embedding_generator.calculate_similarity(
            resume_emb, jd_emb, method='cosine'
        )
        similarities['dot_product'] = self.embedding_generator.calculate_similarity(
            resume_emb, jd_emb, method='dot'
        )
        similarities['euclidean'] = self.embedding_generator.calculate_similarity(
            resume_emb, jd_emb, method='euclidean'
        )
        
        features['similarity_metrics'] = {
            k: round(v * 100, 2) for k, v in similarities.items()
        }
        
        return features
    
    def _calculate_context_similarity(self, resume_sections: Dict[str, str], 
                                    jd_sections: Dict[str, str]) -> Dict[str, Any]:
        """Calculate context-aware similarity using cross-section matching"""
        context_results = {
            'cross_section_matches': {},
            'best_matches': [],
            'overall_score': 0
        }
        
        try:
            # Create embeddings for all non-empty sections
            resume_embeddings = {}
            jd_embeddings = {}
            
            for section, content in resume_sections.items():
                if content.strip():
                    resume_embeddings[f"resume_{section}"] = self.embedding_generator.encode_text(content)
            
            for section, content in jd_sections.items():
                if content.strip():
                    jd_embeddings[f"jd_{section}"] = self.embedding_generator.encode_text(content)
            
            # Calculate cross-section similarities
            all_similarities = []
            
            for resume_key, resume_emb in resume_embeddings.items():
                for jd_key, jd_emb in jd_embeddings.items():
                    similarity = self.embedding_generator.calculate_similarity(resume_emb, jd_emb)
                    
                    match_info = {
                        'resume_section': resume_key,
                        'jd_section': jd_key,
                        'similarity': round(similarity * 100, 2)
                    }
                    
                    all_similarities.append(match_info)
                    
                    # Store in cross-section matrix
                    if resume_key not in context_results['cross_section_matches']:
                        context_results['cross_section_matches'][resume_key] = {}
                    context_results['cross_section_matches'][resume_key][jd_key] = round(similarity * 100, 2)
            
            # Find best matches
            all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            context_results['best_matches'] = all_similarities[:10]
            
            # Calculate overall context score
            if all_similarities:
                top_scores = [match['similarity'] for match in all_similarities[:5]]
                context_results['overall_score'] = sum(top_scores) / len(top_scores)
            
        except Exception as e:
            logger.error(f"Context similarity calculation failed: {str(e)}")
            context_results['error'] = str(e)
        
        return context_results
    
    def _generate_semantic_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights based on semantic analysis"""
        insights = []
        
        overall_score = results.get('combined_semantic_score', 0)
        
        # Overall similarity insights
        if overall_score >= 80:
            insights.append("Excellent semantic alignment with job requirements")
        elif overall_score >= 60:
            insights.append("Good semantic match with room for improvement")
        elif overall_score >= 40:
            insights.append("Moderate semantic alignment - consider enhancing content")
        else:
            insights.append("Low semantic similarity - significant content revision needed")
        
        # Section-specific insights
        section_sims = results.get('section_similarities', {})
        
        # Find strongest sections
        strong_sections = [
            section for section, data in section_sims.items()
            if data['similarity'] >= 70
        ]
        
        if strong_sections:
            insights.append(f"Strong sections: {', '.join(strong_sections)}")
        
        # Find weak sections
        weak_sections = [
            section for section, data in section_sims.items()
            if data['similarity'] < 40 and data['jd_length'] > 0
        ]
        
        if weak_sections:
            insights.append(f"Sections needing improvement: {', '.join(weak_sections)}")
        
        # Context insights
        context_score = results.get('context_similarity', {}).get('overall_score', 0)
        if context_score < 50:
            insights.append("Consider reorganizing content to better match job description structure")
        
        return insights
    
    def store_document_embeddings(self, documents: List[Dict[str, Any]]):
        """
        Store document embeddings in vector database for future similarity searches
        
        Args:
            documents: List of documents with 'text' and metadata
        """
        try:
            embeddings = []
            doc_metadata = []
            
            for doc in documents:
                if 'text' in doc:
                    # Generate embedding
                    embedding = self.embedding_generator.encode_text(doc['text'])
                    embeddings.append(embedding)
                    
                    # Prepare metadata (exclude text to save space)
                    metadata = {k: v for k, v in doc.items() if k != 'text'}
                    doc_metadata.append(metadata)
            
            # Store in vector database
            self.vector_db.add_documents(doc_metadata, embeddings)
            
            logger.info(f"Stored {len(embeddings)} document embeddings")
            
        except Exception as e:
            logger.error(f"Failed to store document embeddings: {str(e)}")
            raise
    
    def find_similar_documents(self, query_text: str, top_k: int = 5, 
                             document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find documents similar to query text
        
        Args:
            query_text: Query text
            top_k: Number of top results
            document_type: Optional filter by document type
        
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_text(query_text)
            
            # Search vector database
            filter_criteria = None
            if document_type:
                filter_criteria = {'document_type': document_type}
            
            results = self.vector_db.search_similar(
                query_embedding, top_k, filter_criteria
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Similar document search failed: {str(e)}")
            return []
    
    def batch_analyze_resumes(self, resume_texts: List[str], jd_text: str) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes against a single job description
        
        Args:
            resume_texts: List of resume texts
            jd_text: Job description text
        
        Returns:
            List of analysis results for each resume
        """
        results = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                analysis = self.analyze_semantic_similarity(resume_text, jd_text)
                analysis['resume_index'] = i
                results.append(analysis)
            except Exception as e:
                logger.error(f"Analysis failed for resume {i}: {str(e)}")
                results.append({
                    'resume_index': i,
                    'error': str(e),
                    'combined_semantic_score': 0
                })
        
        # Sort by combined semantic score
        results.sort(key=lambda x: x.get('combined_semantic_score', 0), reverse=True)
        
        return results