"""
Hard Matching Engine
Combines keyword matching, BM25, TF-IDF, and fuzzy matching for comprehensive hard matching
"""

from typing import Dict, List, Any, Optional
import logging
import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from keyword_matcher import KeywordMatcher
from tfidf_bm25 import AdvancedTextMatcher

logger = logging.getLogger(__name__)

class HardMatcher:
    """
    Comprehensive hard matching engine that combines multiple matching algorithms
    """
    
    def __init__(self, fuzzy_threshold: int = 80):
        """
        Initialize hard matcher with all components
        
        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching
        """
        self.keyword_matcher = KeywordMatcher(fuzzy_threshold)
        self.text_matcher = AdvancedTextMatcher()
        
        # Weights for combining different matching methods
        self.method_weights = {
            'keyword_exact': 0.25,
            'keyword_fuzzy': 0.15,
            'keyword_weighted': 0.20,
            'tfidf_bm25': 0.40
        }
    
    def analyze_match(self, resume_text: str, jd_text: str, 
                     include_detailed: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive hard matching analysis
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
            include_detailed: Whether to include detailed breakdowns
        
        Returns:
            Complete hard matching results
        """
        results = {
            'overall_score': 0,
            'method_scores': {},
            'summary': {},
            'recommendations': []
        }
        
        try:
            # 1. Keyword-based matching
            resume_keywords = self.keyword_matcher.extract_keywords(resume_text)
            jd_keywords = self.keyword_matcher.extract_keywords(jd_text)
            
            # Exact keyword matching
            exact_match = self.keyword_matcher.exact_match(
                resume_keywords['all_keywords'],
                jd_keywords['all_keywords']
            )
            
            # Fuzzy keyword matching
            fuzzy_match = self.keyword_matcher.fuzzy_match(
                resume_keywords['all_keywords'],
                jd_keywords['all_keywords']
            )
            
            # Weighted keyword matching by categories
            weighted_match = self.keyword_matcher.weighted_keyword_match(
                resume_text, jd_text
            )
            
            # 2. Advanced text matching (TF-IDF + BM25)
            advanced_match = self.text_matcher.score_resume_against_jd(
                resume_text, jd_text
            )
            
            # Store method scores
            results['method_scores'] = {
                'keyword_exact': exact_match['match_percentage'],
                'keyword_fuzzy': fuzzy_match['match_percentage'],
                'keyword_weighted': weighted_match['overall_score'],
                'tfidf_bm25': advanced_match['combined_score']
            }
            
            # Calculate overall weighted score
            overall_score = sum(
                results['method_scores'][method] * weight
                for method, weight in self.method_weights.items()
            )
            results['overall_score'] = round(overall_score, 2)
            
            # Generate summary
            results['summary'] = self._generate_summary(
                exact_match, fuzzy_match, weighted_match, advanced_match
            )
            
            # Include detailed results if requested
            if include_detailed:
                results['detailed_results'] = {
                    'keyword_exact': exact_match,
                    'keyword_fuzzy': fuzzy_match,
                    'keyword_weighted': weighted_match,
                    'advanced_matching': advanced_match,
                    'keyword_analysis': self.keyword_matcher.keyword_frequency_analysis(
                        resume_text, jd_text
                    )
                }
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(
                exact_match, weighted_match, advanced_match
            )
            
        except Exception as e:
            logger.error(f"Hard matching analysis failed: {str(e)}")
            results['error'] = str(e)
            results['overall_score'] = 0
        
        return results
    
    def _generate_summary(self, exact_match: Dict, fuzzy_match: Dict, 
                         weighted_match: Dict, advanced_match: Dict) -> Dict[str, Any]:
        """Generate summary of matching results"""
        
        summary = {
            'total_keywords_matched': exact_match['match_count'],
            'total_jd_keywords': exact_match['total_jd_keywords'],
            'keyword_coverage': exact_match['match_percentage'],
            'fuzzy_matches_found': fuzzy_match['match_count'],
            'average_fuzzy_similarity': fuzzy_match['average_similarity'],
            'top_matching_categories': [],
            'missing_critical_skills': [],
            'strength_areas': []
        }
        
        # Analyze category performance from weighted matching
        if 'category_results' in weighted_match:
            category_scores = []
            for category, data in weighted_match['category_results'].items():
                category_scores.append({
                    'category': category,
                    'score': data['combined_score'],
                    'matched_skills': data['exact_match']['matched_keywords']
                })
            
            # Sort by score and get top categories
            category_scores.sort(key=lambda x: x['score'], reverse=True)
            summary['top_matching_categories'] = category_scores[:3]
            
            # Identify missing critical skills
            for category, data in weighted_match['category_results'].items():
                if data['combined_score'] < 30:  # Low score threshold
                    missing_skills = data['exact_match']['missing_keywords'][:5]
                    if missing_skills:
                        summary['missing_critical_skills'].extend([
                            f"{category}: {', '.join(missing_skills)}"
                        ])
        
        # Identify strength areas
        if advanced_match['feature_overlap']['overlap_percentage'] > 60:
            summary['strength_areas'].append("Strong feature overlap with job requirements")
        
        if exact_match['match_percentage'] > 70:
            summary['strength_areas'].append("Excellent keyword coverage")
        
        if fuzzy_match['average_similarity'] > 85:
            summary['strength_areas'].append("High similarity in technical terminology")
        
        return summary
    
    def _generate_recommendations(self, exact_match: Dict, weighted_match: Dict, 
                                advanced_match: Dict) -> List[str]:
        """Generate actionable recommendations based on matching results"""
        
        recommendations = []
        
        # Keyword coverage recommendations
        if exact_match['match_percentage'] < 50:
            recommendations.append(
                f"Improve keyword coverage: Only {exact_match['match_percentage']:.1f}% "
                f"of job requirements are directly mentioned in your resume"
            )
        
        # Missing skills recommendations
        missing_keywords = exact_match['missing_keywords'][:10]
        if missing_keywords:
            recommendations.append(
                f"Consider adding these relevant skills: {', '.join(missing_keywords[:5])}"
            )
        
        # Category-specific recommendations
        if 'category_results' in weighted_match:
            for category, data in weighted_match['category_results'].items():
                if data['combined_score'] < 40 and data['exact_match']['missing_keywords']:
                    missing = data['exact_match']['missing_keywords'][:3]
                    recommendations.append(
                        f"Strengthen {category.replace('_', ' ')}: "
                        f"Add experience with {', '.join(missing)}"
                    )
        
        # Advanced matching recommendations
        if advanced_match['tfidf_similarity'] < 30:
            recommendations.append(
                "Consider restructuring your resume to better align with job description language"
            )
        
        # Feature overlap recommendations
        overlap_pct = advanced_match['feature_overlap']['overlap_percentage']
        if overlap_pct < 40:
            recommendations.append(
                f"Low feature overlap ({overlap_pct:.1f}%): "
                f"Use more terminology from the job description"
            )
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def compare_multiple_resumes(self, resume_texts: List[str], jd_text: str) -> Dict[str, Any]:
        """
        Compare multiple resumes against a single job description
        
        Args:
            resume_texts: List of resume texts
            jd_text: Job description text
        
        Returns:
            Comparison results with rankings
        """
        results = {
            'rankings': [],
            'summary_stats': {},
            'top_performers': [],
            'detailed_scores': []
        }
        
        try:
            # Analyze each resume
            all_scores = []
            for i, resume_text in enumerate(resume_texts):
                analysis = self.analyze_match(resume_text, jd_text, include_detailed=False)
                analysis['resume_index'] = i
                all_scores.append(analysis)
            
            # Sort by overall score
            all_scores.sort(key=lambda x: x['overall_score'], reverse=True)
            
            # Generate rankings
            for rank, score_data in enumerate(all_scores, 1):
                results['rankings'].append({
                    'rank': rank,
                    'resume_index': score_data['resume_index'],
                    'overall_score': score_data['overall_score'],
                    'key_strengths': score_data['summary'].get('strength_areas', [])[:2]
                })
            
            # Calculate summary statistics
            scores = [item['overall_score'] for item in all_scores]
            results['summary_stats'] = {
                'total_resumes': len(resume_texts),
                'average_score': round(sum(scores) / len(scores), 2),
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'score_range': round(max(scores) - min(scores), 2)
            }
            
            # Identify top performers (top 20% or minimum 3)
            top_count = max(3, len(resume_texts) // 5)
            results['top_performers'] = results['rankings'][:top_count]
            
            results['detailed_scores'] = all_scores
            
        except Exception as e:
            logger.error(f"Multi-resume comparison failed: {str(e)}")
            results['error'] = str(e)
        
        return results