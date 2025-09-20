"""
BM25 and TF-IDF Scoring Module
Implements advanced text similarity using BM25 and TF-IDF algorithms
"""

import math
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class BM25Scorer:
    """
    BM25 (Best Matching 25) algorithm implementation for document ranking
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer
        
        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls length normalization (0-1, where 0 = no normalization)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.N = 0
        
    def fit(self, corpus: List[str]):
        """
        Fit BM25 model on corpus
        
        Args:
            corpus: List of documents (strings)
        """
        self.corpus = [doc.lower().split() for doc in corpus]
        self.N = len(corpus)
        
        # Calculate document frequencies and lengths
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / self.N
        
        # Calculate document frequencies for each term
        df = defaultdict(int)
        for doc in self.corpus:
            for word in set(doc):
                df[word] += 1
        
        # Calculate IDF for each term
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5))
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a query against a specific document
        
        Args:
            query: Query string
            doc_idx: Document index in corpus
        
        Returns:
            BM25 score
        """
        query_terms = query.lower().split()
        doc = self.corpus[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0
        doc_term_freqs = Counter(doc)
        
        for term in query_terms:
            if term in doc_term_freqs:
                tf = doc_term_freqs[term]
                idf = self.idf.get(term, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for query against all documents
        
        Args:
            query: Query string
        
        Returns:
            List of scores for each document
        """
        return [self.score(query, i) for i in range(self.N)]
    
    def rank_documents(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents by BM25 score
        
        Args:
            query: Query string
            top_k: Number of top documents to return
        
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k:
            return ranked[:top_k]
        return ranked

class TFIDFScorer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) implementation
    """
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF scorer
        
        Args:
            max_features: Maximum number of features to use
            ngram_range: Range of n-grams to extract
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9+#\-.]{2,}\b'
        )
        self.corpus_vectors = None
        self.feature_names = None
    
    def fit(self, corpus: List[str]):
        """
        Fit TF-IDF model on corpus
        
        Args:
            corpus: List of documents (strings)
        """
        self.corpus_vectors = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors
        
        Args:
            documents: List of documents to transform
        
        Returns:
            TF-IDF matrix
        """
        return self.vectorizer.transform(documents)
    
    def get_similarity_scores(self, query: str) -> List[float]:
        """
        Get cosine similarity scores between query and all corpus documents
        
        Args:
            query: Query string
        
        Returns:
            List of similarity scores
        """
        if self.corpus_vectors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()
        return similarities.tolist()
    
    def rank_documents(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents by TF-IDF similarity
        
        Args:
            query: Query string
            top_k: Number of top documents to return
        
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        similarities = self.get_similarity_scores(query)
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        
        if top_k:
            return ranked[:top_k]
        return ranked
    
    def get_feature_importance(self, document_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Get most important features (terms) for a document
        
        Args:
            document_text: Document text
            top_k: Number of top features to return
        
        Returns:
            List of feature importance dictionaries
        """
        doc_vector = self.vectorizer.transform([document_text])
        
        # Get non-zero features and their scores
        feature_scores = []
        for idx in doc_vector.nonzero()[1]:
            feature_scores.append({
                'feature': self.feature_names[idx],
                'tfidf_score': doc_vector[0, idx],
                'feature_index': idx
            })
        
        # Sort by TF-IDF score
        feature_scores.sort(key=lambda x: x['tfidf_score'], reverse=True)
        
        return feature_scores[:top_k]

class AdvancedTextMatcher:
    """
    Advanced text matching using BM25 and TF-IDF algorithms
    """
    
    def __init__(self):
        """Initialize advanced text matcher"""
        self.bm25 = BM25Scorer()
        self.tfidf = TFIDFScorer()
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """
        Fit both BM25 and TF-IDF models
        
        Args:
            documents: List of documents to fit on
        """
        if not documents:
            raise ValueError("Empty document list provided")
        
        self.bm25.fit(documents)
        self.tfidf.fit(documents)
        self.is_fitted = True
        
        logger.info(f"Fitted models on {len(documents)} documents")
    
    def score_resume_against_jd(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Score resume against job description using multiple algorithms
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Comprehensive scoring results
        """
        # Fit models on both documents
        documents = [resume_text, jd_text]
        self.fit(documents)
        
        # BM25 scoring (query = JD, document = resume)
        bm25_jd_to_resume = self.bm25.score(jd_text, 0)  # JD query against resume (doc 0)
        bm25_resume_to_jd = self.bm25.score(resume_text, 1)  # Resume query against JD (doc 1)
        
        # TF-IDF similarity
        tfidf_similarity = self.tfidf.get_similarity_scores(jd_text)[0]  # JD query against resume
        
        # Get feature importance for both documents
        resume_features = self.tfidf.get_feature_importance(resume_text, top_k=15)
        jd_features = self.tfidf.get_feature_importance(jd_text, top_k=15)
        
        # Calculate combined score
        bm25_avg = (bm25_jd_to_resume + bm25_resume_to_jd) / 2
        combined_score = (bm25_avg * 0.4 + tfidf_similarity * 100 * 0.6)
        
        return {
            'bm25_scores': {
                'jd_to_resume': round(bm25_jd_to_resume, 4),
                'resume_to_jd': round(bm25_resume_to_jd, 4),
                'average': round(bm25_avg, 4)
            },
            'tfidf_similarity': round(tfidf_similarity * 100, 2),  # Convert to percentage
            'combined_score': round(combined_score, 2),
            'resume_important_features': resume_features,
            'jd_important_features': jd_features,
            'feature_overlap': self._calculate_feature_overlap(resume_features, jd_features)
        }
    
    def _calculate_feature_overlap(self, resume_features: List[Dict], jd_features: List[Dict]) -> Dict[str, Any]:
        """
        Calculate overlap between resume and JD important features
        
        Args:
            resume_features: Resume feature importance list
            jd_features: JD feature importance list
        
        Returns:
            Feature overlap analysis
        """
        resume_terms = {f['feature'] for f in resume_features}
        jd_terms = {f['feature'] for f in jd_features}
        
        overlap = resume_terms.intersection(jd_terms)
        overlap_percentage = (len(overlap) / len(jd_terms) * 100) if jd_terms else 0
        
        return {
            'overlapping_features': list(overlap),
            'overlap_count': len(overlap),
            'total_jd_features': len(jd_terms),
            'overlap_percentage': round(overlap_percentage, 2)
        }
    
    def batch_score_resumes(self, resume_texts: List[str], jd_text: str) -> List[Dict[str, Any]]:
        """
        Score multiple resumes against a single job description
        
        Args:
            resume_texts: List of resume texts
            jd_text: Job description text
        
        Returns:
            List of scoring results for each resume
        """
        results = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                score_result = self.score_resume_against_jd(resume_text, jd_text)
                score_result['resume_index'] = i
                results.append(score_result)
            except Exception as e:
                logger.error(f"Scoring failed for resume {i}: {str(e)}")
                results.append({
                    'resume_index': i,
                    'error': str(e),
                    'combined_score': 0
                })
        
        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return results