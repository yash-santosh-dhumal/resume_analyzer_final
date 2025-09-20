"""
Embedding Generator Module
Creates and manages text embeddings using Sentence Transformers
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate and manage text embeddings using Sentence Transformers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence Transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension by encoding a test string
            test_embedding = self.model.encode(["test"])
            self.embedding_dimension = test_embedding.shape[1]
            logger.info(f"Loaded embedding model: {self.model_name}, dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], 
                   normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode text into embeddings
        
        Args:
            text: Single text string or list of text strings
            normalize_embeddings: Whether to normalize embeddings to unit length
        
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = self.model.encode(
                text,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise
    
    def encode_documents(self, documents: List[Dict[str, str]], 
                        text_field: str = 'text') -> Dict[str, np.ndarray]:
        """
        Encode multiple documents with metadata
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to encode
        
        Returns:
            Dictionary with document IDs and their embeddings
        """
        embeddings = {}
        texts_to_encode = []
        doc_ids = []
        
        for i, doc in enumerate(documents):
            if text_field in doc and doc[text_field]:
                texts_to_encode.append(doc[text_field])
                doc_id = doc.get('id', f'doc_{i}')
                doc_ids.append(doc_id)
        
        if texts_to_encode:
            encoded_embeddings = self.encode_text(texts_to_encode)
            
            for doc_id, embedding in zip(doc_ids, encoded_embeddings):
                embeddings[doc_id] = embedding
        
        return embeddings
    
    def encode_sections(self, text_sections: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Encode different sections of a document separately
        
        Args:
            text_sections: Dictionary with section names and their text content
        
        Returns:
            Dictionary with section embeddings
        """
        section_embeddings = {}
        
        for section_name, section_text in text_sections.items():
            if section_text and section_text.strip():
                try:
                    embedding = self.encode_text(section_text)
                    section_embeddings[section_name] = embedding
                except Exception as e:
                    logger.warning(f"Failed to encode section '{section_name}': {str(e)}")
        
        return section_embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray, 
                           method: str = 'cosine') -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ('cosine', 'dot', 'euclidean')
        
        Returns:
            Similarity score
        """
        try:
            if method == 'cosine':
                # Cosine similarity
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
            elif method == 'dot':
                # Dot product (assumes normalized embeddings)
                similarity = np.dot(embedding1, embedding2)
            elif method == 'euclidean':
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                similarity = 1 / (1 + distance)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def calculate_similarity_matrix(self, embeddings1: List[np.ndarray], 
                                  embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        Calculate similarity matrix between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
        
        Returns:
            Similarity matrix
        """
        similarities = []
        
        for emb1 in embeddings1:
            row_similarities = []
            for emb2 in embeddings2:
                similarity = self.calculate_similarity(emb1, emb2)
                row_similarities.append(similarity)
            similarities.append(row_similarities)
        
        return np.array(similarities)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
        
        Returns:
            List of similarity results
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def create_document_embedding(self, document_sections: Dict[str, str],
                                 section_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create a weighted document embedding from multiple sections
        
        Args:
            document_sections: Dictionary with section names and text
            section_weights: Weights for each section (default: equal weights)
        
        Returns:
            Weighted document embedding
        """
        if not document_sections:
            return np.zeros(self.embedding_dimension)
        
        section_embeddings = self.encode_sections(document_sections)
        
        if not section_embeddings:
            return np.zeros(self.embedding_dimension)
        
        # Default weights if not provided
        if section_weights is None:
            section_weights = {section: 1.0 for section in section_embeddings.keys()}
        
        # Calculate weighted average
        total_weight = 0
        weighted_embedding = np.zeros(self.embedding_dimension)
        
        for section_name, embedding in section_embeddings.items():
            weight = section_weights.get(section_name, 1.0)
            weighted_embedding += embedding * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_embedding /= total_weight
        
        return weighted_embedding
    
    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix for two batches of texts
        
        Args:
            texts1: First batch of texts
            texts2: Second batch of texts
        
        Returns:
            Similarity matrix
        """
        embeddings1 = self.encode_text(texts1)
        embeddings2 = self.encode_text(texts2)
        
        return self.calculate_similarity_matrix(embeddings1, embeddings2)