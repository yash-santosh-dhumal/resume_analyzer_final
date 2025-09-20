"""
LLM Reasoning Engine
Orchestrates LLM-based analysis, gap detection, and feedback generation
"""

from typing import Dict, Any, List, Optional
import logging
import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from llm_client import LLMManager
from langchain_analyzer import LangChainAnalyzer

logger = logging.getLogger(__name__)

class LLMReasoningEngine:
    """
    Main LLM reasoning engine that coordinates different analysis components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM reasoning engine
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.config = config
        self.llm_manager = None
        self.langchain_analyzer = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM components"""
        try:
            # Initialize LLM manager
            self.llm_manager = LLMManager(
                primary_provider="openai",
                fallback_provider="huggingface"
            )
            
            # Initialize LangChain analyzer if API key is available
            api_key = self.config.get('api_keys', {}).get('openai')
            if api_key:
                self.langchain_analyzer = LangChainAnalyzer(
                    api_key=api_key,
                    model=self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo')
                )
            
            logger.info("LLM reasoning engine components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {str(e)}")
    
    def comprehensive_analysis(self, resume_text: str, jd_text: str,
                             hard_match_results: Dict[str, Any],
                             soft_match_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive LLM-based analysis
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
            hard_match_results: Results from hard matching
            soft_match_results: Results from soft matching
        
        Returns:
            Comprehensive LLM analysis results
        """
        results = {
            'llm_verdict': 'medium',
            'llm_score': 50,
            'gap_analysis': {},
            'personalized_feedback': '',
            'improvement_suggestions': [],
            'match_explanation': '',
            'success': False
        }
        
        try:
            # Use LangChain analyzer if available
            if self.langchain_analyzer and self.langchain_analyzer.is_available():
                langchain_results = self._langchain_analysis(resume_text, jd_text)
                results.update(langchain_results)
            else:
                # Fallback to basic LLM analysis
                basic_results = self._basic_llm_analysis(
                    resume_text, jd_text, hard_match_results, soft_match_results
                )
                results.update(basic_results)
            
            # Generate additional insights
            insights = self._generate_insights(
                hard_match_results, soft_match_results, results
            )
            results['insights'] = insights
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"LLM comprehensive analysis failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _langchain_analysis(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Perform analysis using LangChain"""
        try:
            # Main resume analysis
            analysis = self.langchain_analyzer.analyze_resume_match(resume_text, jd_text)
            
            # Gap analysis
            gap_analysis = self.langchain_analyzer.generate_gap_analysis(resume_text, jd_text)
            
            # Extract position title from JD for personalized feedback
            position = self._extract_position_title(jd_text)
            personalized_feedback = self.langchain_analyzer.generate_personalized_feedback(
                resume_text, position, analysis
            )
            
            return {
                'llm_verdict': analysis.get('verdict', 'medium'),
                'llm_score': analysis.get('score', 50),
                'gap_analysis': {'detailed_analysis': gap_analysis},
                'personalized_feedback': personalized_feedback,
                'improvement_suggestions': analysis.get('recommendations', []),
                'match_explanation': self._create_match_explanation(analysis),
                'langchain_usage': analysis.get('llm_usage', {}),
                'strengths': analysis.get('strengths', []),
                'weaknesses': analysis.get('weaknesses', []),
                'missing_skills': analysis.get('missing_skills', [])
            }
            
        except Exception as e:
            logger.error(f"LangChain analysis failed: {str(e)}")
            raise
    
    def _basic_llm_analysis(self, resume_text: str, jd_text: str,
                           hard_match_results: Dict[str, Any],
                           soft_match_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic LLM analysis using direct prompting"""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(
                resume_text, jd_text, hard_match_results, soft_match_results
            )
            
            # Generate analysis
            llm_response = self.llm_manager.generate_text(
                prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            # Parse response
            parsed_results = self._parse_basic_response(llm_response)
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"Basic LLM analysis failed: {str(e)}")
            return {
                'llm_verdict': 'medium',
                'llm_score': 50,
                'gap_analysis': {'error': str(e)},
                'personalized_feedback': f"Analysis failed: {str(e)}",
                'improvement_suggestions': []
            }
    
    def _create_analysis_prompt(self, resume_text: str, jd_text: str,
                              hard_match_results: Dict[str, Any],
                              soft_match_results: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt"""
        hard_score = hard_match_results.get('overall_score', 0)
        soft_score = soft_match_results.get('combined_semantic_score', 0)
        
        prompt = f"""
        Analyze this resume against the job description and provide a comprehensive assessment.
        
        QUANTITATIVE ANALYSIS RESULTS:
        - Hard matching score: {hard_score}/100 (keyword and skill matching)
        - Semantic similarity score: {soft_score}/100 (contextual understanding)
        
        JOB DESCRIPTION:
        {jd_text[:1500]}...
        
        RESUME:
        {resume_text[:1500]}...
        
        Please provide:
        
        1. OVERALL VERDICT: [HIGH/MEDIUM/LOW] fit for this position
        2. OVERALL SCORE: [0-100] considering all factors
        3. KEY STRENGTHS: 3-5 main advantages of this candidate
        4. KEY WEAKNESSES: 3-5 areas where candidate falls short
        5. CRITICAL GAPS: Missing skills/experience that are deal-breakers
        6. IMPROVEMENT RECOMMENDATIONS: Specific actionable advice
        7. MATCH EXPLANATION: Why this score was given
        
        Be specific and actionable in your feedback.
        """
        
        return prompt
    
    def _parse_basic_response(self, response: str) -> Dict[str, Any]:
        """Parse basic LLM response into structured format"""
        results = {
            'llm_verdict': 'medium',
            'llm_score': 50,
            'gap_analysis': {},
            'personalized_feedback': response,
            'improvement_suggestions': [],
            'match_explanation': ''
        }
        
        try:
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract verdict
                if 'verdict:' in line.lower() or 'overall verdict:' in line.lower():
                    verdict_text = line.split(':', 1)[1].strip().upper()
                    if 'HIGH' in verdict_text:
                        results['llm_verdict'] = 'high'
                    elif 'LOW' in verdict_text:
                        results['llm_verdict'] = 'low'
                    else:
                        results['llm_verdict'] = 'medium'
                
                # Extract score
                elif 'score:' in line.lower() or 'overall score:' in line.lower():
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        score = int(''.join(filter(str.isdigit, score_text)))
                        results['llm_score'] = min(100, max(0, score))
                    except:
                        pass
                
                # Extract recommendations
                elif 'recommendation' in line.lower():
                    current_section = 'recommendations'
                elif 'improvement' in line.lower():
                    current_section = 'recommendations'
                elif 'explanation' in line.lower():
                    current_section = 'explanation'
                elif line.startswith('-') or line.startswith('•'):
                    if current_section == 'recommendations':
                        results['improvement_suggestions'].append(line[1:].strip())
                    elif current_section == 'explanation':
                        results['match_explanation'] += line[1:].strip() + ' '
        
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {str(e)}")
        
        return results
    
    def _extract_position_title(self, jd_text: str) -> str:
        """Extract position title from job description"""
        lines = jd_text.split('\n')[:5]  # Check first few lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100:
                # Likely a title if it's short and at the beginning
                if any(word in line.lower() for word in ['engineer', 'developer', 'analyst', 'manager', 'specialist']):
                    return line
        
        return "Software Engineer"  # Default fallback
    
    def _create_match_explanation(self, analysis: Dict[str, Any]) -> str:
        """Create match explanation from analysis results"""
        verdict = analysis.get('verdict', 'medium')
        score = analysis.get('score', 50)
        strengths = analysis.get('strengths', [])
        weaknesses = analysis.get('weaknesses', [])
        
        explanation = f"This candidate receives a {verdict.upper()} fit rating with a score of {score}/100. "
        
        if strengths:
            explanation += f"Key strengths include: {', '.join(strengths[:3])}. "
        
        if weaknesses:
            explanation += f"Areas for improvement: {', '.join(weaknesses[:3])}."
        
        return explanation
    
    def _generate_insights(self, hard_results: Dict[str, Any], 
                          soft_results: Dict[str, Any],
                          llm_results: Dict[str, Any]) -> List[str]:
        """Generate insights by combining all analysis results"""
        insights = []
        
        hard_score = hard_results.get('overall_score', 0)
        soft_score = soft_results.get('combined_semantic_score', 0)
        llm_score = llm_results.get('llm_score', 50)
        
        # Score consistency insights
        scores = [hard_score, soft_score, llm_score]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        if score_variance < 100:  # Low variance
            insights.append(f"Consistent performance across all evaluation methods (avg: {avg_score:.1f})")
        else:
            insights.append("Inconsistent performance across evaluation methods - deeper review recommended")
        
        # Specific insights based on score patterns
        if hard_score > 70 and soft_score < 40:
            insights.append("Strong keyword match but weak semantic alignment - may lack contextual fit")
        elif soft_score > 70 and hard_score < 40:
            insights.append("Good conceptual fit but missing specific required skills")
        elif all(s > 70 for s in scores):
            insights.append("Excellent candidate across all evaluation criteria")
        elif all(s < 30 for s in scores):
            insights.append("Poor fit across all evaluation criteria")
        
        # LLM-specific insights
        verdict = llm_results.get('llm_verdict', 'medium')
        if verdict == 'high' and avg_score < 60:
            insights.append("LLM identifies strong potential despite lower quantitative scores")
        elif verdict == 'low' and avg_score > 70:
            insights.append("Quantitative scores high but LLM identifies qualitative concerns")
        
        return insights[:5]  # Limit to top 5 insights
    
    def generate_hiring_recommendation(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final hiring recommendation based on all analysis results
        
        Args:
            all_results: Complete analysis results from all modules
        
        Returns:
            Final hiring recommendation
        """
        recommendation = {
            'decision': 'MAYBE',
            'confidence': 'medium',
            'reasoning': '',
            'next_steps': [],
            'risk_factors': [],
            'success_probability': 50
        }
        
        try:
            # Extract scores
            hard_score = all_results.get('hard_matching', {}).get('overall_score', 0)
            soft_score = all_results.get('soft_matching', {}).get('combined_semantic_score', 0)
            llm_score = all_results.get('llm_analysis', {}).get('llm_score', 50)
            
            # Calculate weighted final score
            final_score = (hard_score * 0.4 + soft_score * 0.3 + llm_score * 0.3)
            
            # Make decision based on final score
            if final_score >= 75:
                recommendation['decision'] = 'HIRE'
                recommendation['confidence'] = 'high'
                recommendation['success_probability'] = min(95, final_score + 10)
            elif final_score >= 60:
                recommendation['decision'] = 'INTERVIEW'
                recommendation['confidence'] = 'medium'
                recommendation['success_probability'] = final_score
            elif final_score >= 40:
                recommendation['decision'] = 'MAYBE'
                recommendation['confidence'] = 'low'
                recommendation['success_probability'] = final_score - 10
            else:
                recommendation['decision'] = 'REJECT'
                recommendation['confidence'] = 'high'
                recommendation['success_probability'] = max(10, final_score - 10)
            
            # Generate reasoning
            recommendation['reasoning'] = self._create_decision_reasoning(final_score, all_results)
            
            # Generate next steps
            recommendation['next_steps'] = self._generate_next_steps(recommendation['decision'], all_results)
            
            # Identify risk factors
            recommendation['risk_factors'] = self._identify_risk_factors(all_results)
            
        except Exception as e:
            logger.error(f"Failed to generate hiring recommendation: {str(e)}")
            recommendation['error'] = str(e)
        
        return recommendation
    
    def _create_decision_reasoning(self, final_score: float, all_results: Dict[str, Any]) -> str:
        """Create reasoning for hiring decision"""
        reasoning = f"Final evaluation score: {final_score:.1f}/100. "
        
        # Add component score details
        hard_score = all_results.get('hard_matching', {}).get('overall_score', 0)
        soft_score = all_results.get('soft_matching', {}).get('combined_semantic_score', 0)
        llm_score = all_results.get('llm_analysis', {}).get('llm_score', 50)
        
        reasoning += f"Component scores - Hard matching: {hard_score}, Semantic fit: {soft_score}, LLM analysis: {llm_score}. "
        
        # Add qualitative factors
        llm_verdict = all_results.get('llm_analysis', {}).get('llm_verdict', 'medium')
        reasoning += f"LLM assessment indicates {llm_verdict} overall fit."
        
        return reasoning
    
    def _generate_next_steps(self, decision: str, all_results: Dict[str, Any]) -> List[str]:
        """Generate next steps based on decision"""
        next_steps = []
        
        if decision == 'HIRE':
            next_steps = [
                "Proceed with offer preparation",
                "Verify references and background",
                "Prepare onboarding plan",
                "Schedule final stakeholder approval"
            ]
        elif decision == 'INTERVIEW':
            next_steps = [
                "Schedule technical interview",
                "Focus interview on identified weak areas",
                "Have candidate demonstrate key skills",
                "Get team feedback on cultural fit"
            ]
        elif decision == 'MAYBE':
            next_steps = [
                "Request additional information or portfolio samples",
                "Consider phone screening first",
                "Evaluate against other candidates",
                "Reassess if no better candidates available"
            ]
        else:  # REJECT
            next_steps = [
                "Send polite rejection email",
                "Provide constructive feedback if requested",
                "Keep resume on file for future opportunities",
                "Continue search for qualified candidates"
            ]
        
        return next_steps
    
    def _identify_risk_factors(self, all_results: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        # Check for significant gaps
        missing_skills = all_results.get('llm_analysis', {}).get('missing_skills', [])
        if len(missing_skills) > 3:
            risk_factors.append(f"Multiple missing critical skills: {', '.join(missing_skills[:3])}")
        
        # Check for low semantic fit
        soft_score = all_results.get('soft_matching', {}).get('combined_semantic_score', 0)
        if soft_score < 40:
            risk_factors.append("Low semantic alignment with job requirements")
        
        # Check for keyword gaps
        hard_score = all_results.get('hard_matching', {}).get('overall_score', 0)
        if hard_score < 30:
            risk_factors.append("Significant gaps in required technical skills")
        
        # Check for inconsistent performance
        scores = [
            all_results.get('hard_matching', {}).get('overall_score', 0),
            all_results.get('soft_matching', {}).get('combined_semantic_score', 0),
            all_results.get('llm_analysis', {}).get('llm_score', 50)
        ]
        
        if max(scores) - min(scores) > 40:
            risk_factors.append("Inconsistent performance across evaluation criteria")
        
        return risk_factors[:5]  # Limit to top 5 risk factors