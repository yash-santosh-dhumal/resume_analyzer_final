"""
Scoring Engine
Combines hard matching, soft matching, and LLM analysis into final relevance scores
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MatchLevel(Enum):
    """Enum for match levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ScoreWeights:
    """Configuration for score weights"""
    hard_matching: float = 0.40
    soft_matching: float = 0.30
    llm_analysis: float = 0.30
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.hard_matching + self.soft_matching + self.llm_analysis
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

@dataclass
class RelevanceScore:
    """Final relevance score with details"""
    overall_score: float
    match_level: MatchLevel
    confidence: float
    component_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    explanation: str
    recommendations: List[str]
    risk_factors: List[str]

class ScoringEngine:
    """
    Main scoring engine that combines all analysis components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scoring engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.weights = self._load_weights()
        self.thresholds = self._load_thresholds()
        
    def _load_weights(self) -> ScoreWeights:
        """Load scoring weights from config"""
        weights_config = self.config.get('scoring', {}).get('weights', {})
        
        return ScoreWeights(
            hard_matching=weights_config.get('hard_matching', 0.40),
            soft_matching=weights_config.get('soft_matching', 0.30),
            llm_analysis=weights_config.get('llm_analysis', 0.30)
        )
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load scoring thresholds from config"""
        default_thresholds = {
            'excellent': 80.0,
            'good': 65.0,
            'fair': 45.0,
            'poor': 0.0
        }
        
        return self.config.get('scoring', {}).get('thresholds', default_thresholds)
    
    def calculate_relevance_score(self, 
                                 hard_results: Dict[str, Any],
                                 soft_results: Dict[str, Any],
                                 llm_results: Dict[str, Any]) -> RelevanceScore:
        """
        Calculate final relevance score combining all analysis results
        
        Args:
            hard_results: Hard matching results
            soft_results: Soft matching results  
            llm_results: LLM analysis results
            
        Returns:
            RelevanceScore object with complete scoring details
        """
        try:
            # Extract component scores
            component_scores = self._extract_component_scores(
                hard_results, soft_results, llm_results
            )
            
            # Calculate weighted scores
            weighted_scores = self._calculate_weighted_scores(component_scores)
            
            # Calculate overall score
            overall_score = sum(weighted_scores.values())
            
            # Determine match level
            match_level = self._determine_match_level(overall_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(component_scores, overall_score)
            
            # Generate explanation
            explanation = self._generate_explanation(
                overall_score, component_scores, weighted_scores
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component_scores, llm_results
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                component_scores, hard_results, soft_results, llm_results
            )
            
            return RelevanceScore(
                overall_score=round(overall_score, 2),
                match_level=match_level,
                confidence=round(confidence, 2),
                component_scores=component_scores,
                weighted_scores=weighted_scores,
                explanation=explanation,
                recommendations=recommendations,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {str(e)}")
            # Return default score on error
            return self._create_error_score(str(e))
    
    def _extract_component_scores(self, hard_results: Dict[str, Any],
                                 soft_results: Dict[str, Any],
                                 llm_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize component scores"""
        scores = {}
        
        # Hard matching score
        scores['hard_matching'] = self._normalize_score(
            hard_results.get('overall_score', 0)
        )
        
        # Soft matching score (semantic similarity)
        scores['soft_matching'] = self._normalize_score(
            soft_results.get('combined_semantic_score', 0)
        )
        
        # LLM analysis score
        scores['llm_analysis'] = self._normalize_score(
            llm_results.get('llm_score', 50)
        )
        
        # Additional detailed scores for analysis
        scores['keyword_matching'] = self._normalize_score(
            hard_results.get('keyword_score', 0)
        )
        
        scores['skill_matching'] = self._normalize_score(
            hard_results.get('skills_score', 0)
        )
        
        scores['semantic_similarity'] = self._normalize_score(
            soft_results.get('semantic_score', 0)
        )
        
        scores['embedding_similarity'] = self._normalize_score(
            soft_results.get('embedding_score', 0)
        )
        
        return scores
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-100 range"""
        return max(0.0, min(100.0, float(score)))
    
    def _calculate_weighted_scores(self, component_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted component scores"""
        return {
            'hard_matching': component_scores['hard_matching'] * self.weights.hard_matching,
            'soft_matching': component_scores['soft_matching'] * self.weights.soft_matching,
            'llm_analysis': component_scores['llm_analysis'] * self.weights.llm_analysis
        }
    
    def _determine_match_level(self, overall_score: float) -> MatchLevel:
        """Determine match level based on overall score"""
        if overall_score >= self.thresholds['excellent']:
            return MatchLevel.EXCELLENT
        elif overall_score >= self.thresholds['good']:
            return MatchLevel.GOOD
        elif overall_score >= self.thresholds['fair']:
            return MatchLevel.FAIR
        else:
            return MatchLevel.POOR
    
    def _calculate_confidence(self, component_scores: Dict[str, float], 
                            overall_score: float) -> float:
        """Calculate confidence in the score"""
        # Base confidence on score consistency
        main_scores = [
            component_scores['hard_matching'],
            component_scores['soft_matching'], 
            component_scores['llm_analysis']
        ]
        
        # Calculate variance
        mean_score = sum(main_scores) / len(main_scores)
        variance = sum((score - mean_score) ** 2 for score in main_scores) / len(main_scores)
        
        # Higher variance = lower confidence
        consistency_factor = max(0, 100 - variance)
        
        # Score extremes have higher confidence
        extremity_factor = max(overall_score, 100 - overall_score)
        
        # Combine factors
        confidence = (consistency_factor * 0.7 + extremity_factor * 0.3)
        
        return min(100.0, max(10.0, confidence))
    
    def _generate_explanation(self, overall_score: float, 
                            component_scores: Dict[str, float],
                            weighted_scores: Dict[str, float]) -> str:
        """Generate human-readable explanation of the score"""
        match_level = self._determine_match_level(overall_score)
        
        explanation = f"Overall relevance score: {overall_score:.1f}/100 ({match_level.value.upper()} match). "
        
        # Add component breakdown
        explanation += f"Component scores: Hard matching {component_scores['hard_matching']:.1f} "
        explanation += f"(weighted: {weighted_scores['hard_matching']:.1f}), "
        explanation += f"Semantic fit {component_scores['soft_matching']:.1f} "
        explanation += f"(weighted: {weighted_scores['soft_matching']:.1f}), "
        explanation += f"LLM analysis {component_scores['llm_analysis']:.1f} "
        explanation += f"(weighted: {weighted_scores['llm_analysis']:.1f}). "
        
        # Add qualitative assessment
        if match_level == MatchLevel.EXCELLENT:
            explanation += "Candidate demonstrates strong alignment across all evaluation criteria."
        elif match_level == MatchLevel.GOOD:
            explanation += "Candidate shows good fit with minor gaps in some areas."
        elif match_level == MatchLevel.FAIR:
            explanation += "Candidate has potential but significant improvement needed."
        else:
            explanation += "Candidate does not meet minimum requirements for this position."
        
        return explanation
    
    def _generate_recommendations(self, component_scores: Dict[str, float],
                                llm_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check component performance
        if component_scores['hard_matching'] < 50:
            recommendations.append("Focus on developing specific technical skills mentioned in job requirements")
        
        if component_scores['soft_matching'] < 50:
            recommendations.append("Improve resume content to better align with job context and responsibilities")
        
        if component_scores['llm_analysis'] < 50:
            recommendations.append("Address qualitative gaps identified in experience and competencies")
        
        # Add LLM recommendations if available
        llm_recommendations = llm_results.get('improvement_suggestions', [])
        if llm_recommendations:
            recommendations.extend(llm_recommendations[:3])  # Add top 3 LLM recommendations
        
        # Add general recommendations based on overall performance
        overall_avg = (component_scores['hard_matching'] + 
                      component_scores['soft_matching'] + 
                      component_scores['llm_analysis']) / 3
        
        if overall_avg > 75:
            recommendations.append("Consider for immediate interview - strong candidate")
        elif overall_avg > 50:
            recommendations.append("Schedule phone screening to assess fit")
        else:
            recommendations.append("Significant skill development needed before considering")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _identify_risk_factors(self, component_scores: Dict[str, float],
                             hard_results: Dict[str, Any],
                             soft_results: Dict[str, Any],
                             llm_results: Dict[str, Any]) -> List[str]:
        """Identify potential hiring risk factors"""
        risk_factors = []
        
        # Score-based risk factors
        if component_scores['hard_matching'] < 30:
            risk_factors.append("Critical technical skills gap")
        
        if component_scores['soft_matching'] < 30:
            risk_factors.append("Poor contextual fit for role responsibilities")
        
        # Inconsistency risk
        scores = [component_scores['hard_matching'], 
                 component_scores['soft_matching'],
                 component_scores['llm_analysis']]
        if max(scores) - min(scores) > 40:
            risk_factors.append("Inconsistent performance across evaluation criteria")
        
        # LLM-identified risks
        llm_risks = llm_results.get('risk_factors', [])
        if llm_risks:
            risk_factors.extend(llm_risks[:2])  # Add top 2 LLM risk factors
        
        # Missing critical skills
        missing_skills = hard_results.get('missing_required_skills', [])
        if len(missing_skills) > 2:
            risk_factors.append(f"Multiple missing required skills: {', '.join(missing_skills[:3])}")
        
        return risk_factors[:5]  # Limit to top 5 risk factors
    
    def _create_error_score(self, error_message: str) -> RelevanceScore:
        """Create default score when calculation fails"""
        return RelevanceScore(
            overall_score=0.0,
            match_level=MatchLevel.POOR,
            confidence=10.0,
            component_scores={'hard_matching': 0, 'soft_matching': 0, 'llm_analysis': 0},
            weighted_scores={'hard_matching': 0, 'soft_matching': 0, 'llm_analysis': 0},
            explanation=f"Score calculation failed: {error_message}",
            recommendations=["Manual review required due to scoring error"],
            risk_factors=["Automated scoring failed"]
        )
    
    def batch_score_candidates(self, candidates_results: List[Dict[str, Any]]) -> List[RelevanceScore]:
        """
        Score multiple candidates and rank them
        
        Args:
            candidates_results: List of complete analysis results for each candidate
            
        Returns:
            List of RelevanceScore objects sorted by overall score
        """
        scores = []
        
        for i, candidate_result in enumerate(candidates_results):
            try:
                score = self.calculate_relevance_score(
                    candidate_result.get('hard_matching', {}),
                    candidate_result.get('soft_matching', {}),
                    candidate_result.get('llm_analysis', {})
                )
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Failed to score candidate {i}: {str(e)}")
                scores.append(self._create_error_score(f"Candidate {i} scoring failed"))
        
        # Sort by overall score (descending)
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return scores
    
    def generate_ranking_report(self, scored_candidates: List[RelevanceScore],
                              candidate_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive ranking report
        
        Args:
            scored_candidates: List of scored candidates
            candidate_names: Optional list of candidate names
            
        Returns:
            Ranking report with statistics and insights
        """
        if not scored_candidates:
            return {'error': 'No candidates to rank'}
        
        report = {
            'total_candidates': len(scored_candidates),
            'ranking': [],
            'statistics': {},
            'insights': []
        }
        
        # Create ranking
        for i, score in enumerate(scored_candidates):
            candidate_name = candidate_names[i] if candidate_names and i < len(candidate_names) else f"Candidate {i+1}"
            
            report['ranking'].append({
                'rank': i + 1,
                'candidate': candidate_name,
                'overall_score': score.overall_score,
                'match_level': score.match_level.value,
                'confidence': score.confidence,
                'component_scores': score.component_scores,
                'top_recommendation': score.recommendations[0] if score.recommendations else "No recommendations"
            })
        
        # Calculate statistics
        scores = [candidate.overall_score for candidate in scored_candidates]
        report['statistics'] = {
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'average_score': sum(scores) / len(scores),
            'median_score': sorted(scores)[len(scores) // 2],
            'excellent_candidates': len([s for s in scored_candidates if s.match_level == MatchLevel.EXCELLENT]),
            'good_candidates': len([s for s in scored_candidates if s.match_level == MatchLevel.GOOD]),
            'fair_candidates': len([s for s in scored_candidates if s.match_level == MatchLevel.FAIR]),
            'poor_candidates': len([s for s in scored_candidates if s.match_level == MatchLevel.POOR])
        }
        
        # Generate insights
        report['insights'] = self._generate_ranking_insights(scored_candidates, report['statistics'])
        
        return report
    
    def _generate_ranking_insights(self, candidates: List[RelevanceScore], 
                                  stats: Dict[str, Any]) -> List[str]:
        """Generate insights about the candidate pool"""
        insights = []
        
        total = stats['total_candidates']
        excellent = stats['excellent_candidates']
        good = stats['good_candidates']
        
        # Overall pool quality
        if excellent > 0:
            insights.append(f"{excellent} excellent candidate(s) identified - strong hiring pool")
        elif good > total * 0.3:
            insights.append("Good quality candidate pool with several viable options")
        elif stats['average_score'] < 40:
            insights.append("Weak candidate pool - consider expanding search criteria")
        
        # Score distribution
        score_range = stats['highest_score'] - stats['lowest_score']
        if score_range > 50:
            insights.append("Wide score distribution - clear differentiation between candidates")
        elif score_range < 20:
            insights.append("Narrow score distribution - candidates have similar qualification levels")
        
        # Component analysis
        hard_scores = [c.component_scores.get('hard_matching', 0) for c in candidates]
        soft_scores = [c.component_scores.get('soft_matching', 0) for c in candidates]
        
        avg_hard = sum(hard_scores) / len(hard_scores)
        avg_soft = sum(soft_scores) / len(soft_scores)
        
        if avg_hard > avg_soft + 15:
            insights.append("Candidates generally strong on technical skills but weaker on contextual fit")
        elif avg_soft > avg_hard + 15:
            insights.append("Candidates show good conceptual understanding but may lack specific skills")
        
        return insights[:5]  # Limit to top 5 insights