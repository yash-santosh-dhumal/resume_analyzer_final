"""
Advanced LLM Integration with LangGraph and LangSmith
Stateful pipelines and observability for enterprise LLM workflows
"""

from typing import Dict, List, Any, Optional, Callable, Type
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid

# LangGraph and LangSmith imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from langchain.tools import BaseTool
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import traceable
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Mock classes for when LangGraph is not available
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, start, end): pass
        def add_conditional_edges(self, start, condition, mapping): pass
        def set_entry_point(self, name): pass
        def compile(self): return MockCompiledGraph()
    
    class MockCompiledGraph:
        def invoke(self, input_data): return {"output": "Mock response"}
    
    def traceable(func): return func

logger = logging.getLogger(__name__)

class AnalysisStage(Enum):
    """Stages in the resume analysis pipeline"""
    PREPROCESSING = "preprocessing"
    EXTRACTION = "extraction"
    MATCHING = "matching"
    SCORING = "scoring"
    REASONING = "reasoning"
    VALIDATION = "validation"
    REPORTING = "reporting"

@dataclass
class AnalysisState:
    """State object for LangGraph workflow"""
    # Input data
    resume_text: str = ""
    job_description_text: str = ""
    location: str = ""
    
    # Extracted information
    resume_data: Dict[str, Any] = field(default_factory=dict)
    job_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    hard_match_scores: Dict[str, float] = field(default_factory=dict)
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Final output
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    
    # Workflow metadata
    current_stage: AnalysisStage = AnalysisStage.PREPROCESSING
    errors: List[str] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    processing_time: Dict[str, float] = field(default_factory=dict)

class LangGraphPipeline:
    """
    Advanced LLM pipeline using LangGraph for stateful resume analysis
    """
    
    def __init__(self, config: Dict[str, Any], analyzer):
        """Initialize LangGraph pipeline"""
        self.config = config
        self.analyzer = analyzer
        self.langsmith_client = None
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, using simplified pipeline")
            return
        
        # Initialize LangSmith for observability
        self._initialize_langsmith()
        
        # Build the analysis graph
        self.graph = self._build_analysis_graph()
        
        logger.info("LangGraph pipeline initialized successfully")
    
    def _initialize_langsmith(self):
        """Initialize LangSmith client for observability"""
        try:
            langsmith_config = self.config.get('langsmith', {})
            if langsmith_config.get('enabled', False):
                self.langsmith_client = LangSmithClient(
                    api_url=langsmith_config.get('api_url', 'https://api.langsmith.com'),
                    api_key=langsmith_config.get('api_key', '')
                )
                logger.info("LangSmith observability enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LangSmith: {e}")
    
    def _build_analysis_graph(self) -> StateGraph:
        """Build the LangGraph workflow for resume analysis"""
        # Define the state schema
        workflow = StateGraph(AnalysisState)
        
        # Add nodes for each stage
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("extract_resume", self._extract_resume_node)
        workflow.add_node("extract_job", self._extract_job_node)
        workflow.add_node("hard_matching", self._hard_matching_node)
        workflow.add_node("semantic_matching", self._semantic_matching_node)
        workflow.add_node("llm_reasoning", self._llm_reasoning_node)
        workflow.add_node("score_calculation", self._score_calculation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("report_generation", self._report_generation_node)
        
        # Define the workflow edges
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "extract_resume")
        workflow.add_edge("extract_resume", "extract_job")
        workflow.add_edge("extract_job", "hard_matching")
        workflow.add_edge("hard_matching", "semantic_matching")
        workflow.add_edge("semantic_matching", "llm_reasoning")
        workflow.add_edge("llm_reasoning", "score_calculation")
        workflow.add_edge("score_calculation", "validation")
        
        # Conditional edge for validation
        workflow.add_conditional_edges(
            "validation",
            self._should_regenerate,
            {
                "regenerate": "llm_reasoning",
                "complete": "report_generation"
            }
        )
        
        workflow.add_edge("report_generation", END)
        
        return workflow.compile()
    
    @traceable
    def _preprocess_node(self, state: AnalysisState) -> AnalysisState:
        """Preprocessing stage - clean and prepare input data"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.PREPROCESSING
        
        try:
            # Clean resume text
            if state.resume_text:
                state.resume_text = self._clean_text(state.resume_text)
            
            # Clean job description
            if state.job_description_text:
                state.job_description_text = self._clean_text(state.job_description_text)
            
            state.stage_results["preprocessing"] = {
                "resume_length": len(state.resume_text),
                "job_description_length": len(state.job_description_text),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["preprocessing"] = {"status": "failed", "error": error_msg}
        
        # Record processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["preprocessing"] = processing_time
        
        return state
    
    @traceable
    def _extract_resume_node(self, state: AnalysisState) -> AnalysisState:
        """Extract structured data from resume"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.EXTRACTION
        
        try:
            # Extract resume information using the analyzer
            resume_info = {
                "skills": list(self.analyzer._extract_skills(state.resume_text)),
                "experience_years": self.analyzer._extract_years_experience(state.resume_text),
                "keywords": dict(self.analyzer._extract_keywords(state.resume_text)),
                "candidate_name": self.analyzer._extract_candidate_name(state.resume_text),
                "email": self.analyzer._extract_email(state.resume_text),
                "phone": self.analyzer._extract_phone(state.resume_text)
            }
            
            state.resume_data = resume_info
            state.stage_results["extract_resume"] = {
                "skills_count": len(resume_info["skills"]),
                "keywords_count": len(resume_info["keywords"]),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Resume extraction failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["extract_resume"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["extract_resume"] = processing_time
        
        return state
    
    @traceable
    def _extract_job_node(self, state: AnalysisState) -> AnalysisState:
        """Extract structured data from job description"""
        start_time = datetime.now()
        
        try:
            # Extract job requirements
            job_info = {
                "required_skills": list(self.analyzer._extract_skills(state.job_description_text)),
                "keywords": dict(self.analyzer._extract_keywords(state.job_description_text)),
                "experience_requirement": self.analyzer._extract_years_experience(state.job_description_text),
                "job_title": self.analyzer._extract_job_title(state.job_description_text),
                "company": self.analyzer._extract_company_name(state.job_description_text)
            }
            
            state.job_requirements = job_info
            state.stage_results["extract_job"] = {
                "required_skills_count": len(job_info["required_skills"]),
                "keywords_count": len(job_info["keywords"]),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Job extraction failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["extract_job"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["extract_job"] = processing_time
        
        return state
    
    @traceable
    def _hard_matching_node(self, state: AnalysisState) -> AnalysisState:
        """Perform hard matching analysis"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.MATCHING
        
        try:
            # Calculate hard matching scores using existing analyzer methods
            keyword_score = self.analyzer._calculate_keyword_score(
                state.resume_data.get("keywords", {}),
                state.job_requirements.get("keywords", {})
            )
            
            skill_score = self.analyzer._calculate_skill_score(
                set(state.resume_data.get("skills", [])),
                set(state.job_requirements.get("required_skills", []))
            )
            
            experience_score = self.analyzer._calculate_experience_score(
                state.resume_text, state.job_description_text
            )
            
            state.hard_match_scores = {
                "keyword_score": keyword_score,
                "skill_score": skill_score,
                "experience_score": experience_score
            }
            
            state.stage_results["hard_matching"] = {
                "average_score": sum(state.hard_match_scores.values()) / len(state.hard_match_scores),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Hard matching failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["hard_matching"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["hard_matching"] = processing_time
        
        return state
    
    @traceable
    def _semantic_matching_node(self, state: AnalysisState) -> AnalysisState:
        """Perform semantic matching analysis"""
        start_time = datetime.now()
        
        try:
            # Calculate semantic similarity
            context_score = self.analyzer._calculate_context_score(
                state.resume_text, state.job_description_text
            )
            
            # In a full implementation, this would use embeddings
            # For now, use the context score as semantic score
            state.semantic_scores = {
                "context_similarity": context_score,
                "semantic_alignment": context_score * 0.9,  # Slightly adjusted
                "domain_relevance": context_score * 1.1    # Slightly boosted
            }
            
            state.stage_results["semantic_matching"] = {
                "average_score": sum(state.semantic_scores.values()) / len(state.semantic_scores),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Semantic matching failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["semantic_matching"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["semantic_matching"] = processing_time
        
        return state
    
    @traceable
    def _llm_reasoning_node(self, state: AnalysisState) -> AnalysisState:
        """LLM-powered reasoning and analysis"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.REASONING
        
        try:
            # Generate LLM analysis prompt
            prompt = self._create_analysis_prompt(state)
            
            # In a full implementation, this would call an actual LLM
            # For now, generate analysis based on scores
            reasoning = self._generate_reasoning(state)
            recommendations = self._generate_recommendations(state)
            
            state.llm_analysis = {
                "reasoning": reasoning,
                "recommendations": recommendations,
                "confidence_factors": self._analyze_confidence_factors(state),
                "risk_assessment": self._assess_risks(state)
            }
            
            state.stage_results["llm_reasoning"] = {
                "reasoning_length": len(reasoning),
                "recommendations_count": len(recommendations),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"LLM reasoning failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["llm_reasoning"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["llm_reasoning"] = processing_time
        
        return state
    
    @traceable
    def _score_calculation_node(self, state: AnalysisState) -> AnalysisState:
        """Calculate final scores"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.SCORING
        
        try:
            # Combine scores with weights
            weights = {
                "hard_match": 0.4,
                "semantic_match": 0.3,
                "llm_analysis": 0.3
            }
            
            # Calculate weighted average
            hard_avg = sum(state.hard_match_scores.values()) / max(len(state.hard_match_scores), 1)
            semantic_avg = sum(state.semantic_scores.values()) / max(len(state.semantic_scores), 1)
            
            # LLM score based on reasoning quality and confidence
            llm_score = state.llm_analysis.get("confidence_factors", {}).get("overall_confidence", 50.0)
            
            state.overall_score = (
                hard_avg * weights["hard_match"] +
                semantic_avg * weights["semantic_match"] +
                llm_score * weights["llm_analysis"]
            )
            
            # Calculate confidence based on consistency across methods
            score_variance = self._calculate_score_variance([hard_avg, semantic_avg, llm_score])
            state.confidence = max(0, 100 - (score_variance * 10))  # Lower variance = higher confidence
            
            state.stage_results["score_calculation"] = {
                "overall_score": state.overall_score,
                "confidence": state.confidence,
                "score_variance": score_variance,
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Score calculation failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["score_calculation"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["score_calculation"] = processing_time
        
        return state
    
    @traceable
    def _validation_node(self, state: AnalysisState) -> AnalysisState:
        """Validate analysis results"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.VALIDATION
        
        try:
            validation_issues = []
            
            # Check score consistency
            if state.overall_score < 0 or state.overall_score > 100:
                validation_issues.append("Overall score out of valid range")
            
            # Check confidence level
            if state.confidence < 30:
                validation_issues.append("Low confidence in analysis results")
            
            # Check for missing data
            if not state.resume_data.get("skills"):
                validation_issues.append("No skills extracted from resume")
            
            if not state.job_requirements.get("required_skills"):
                validation_issues.append("No required skills extracted from job description")
            
            state.stage_results["validation"] = {
                "validation_issues": validation_issues,
                "is_valid": len(validation_issues) == 0,
                "status": "success"
            }
            
            if validation_issues:
                state.errors.extend(validation_issues)
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["validation"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["validation"] = processing_time
        
        return state
    
    @traceable
    def _report_generation_node(self, state: AnalysisState) -> AnalysisState:
        """Generate final analysis report"""
        start_time = datetime.now()
        state.current_stage = AnalysisStage.REPORTING
        
        try:
            # Compile final reasoning
            state.reasoning = self._compile_final_reasoning(state)
            
            # Compile final recommendations
            state.recommendations = list(set(
                state.llm_analysis.get("recommendations", []) +
                self.analyzer._generate_recommendations(
                    state.hard_match_scores.get("keyword_score", 0),
                    state.hard_match_scores.get("skill_score", 0),
                    state.semantic_scores.get("context_similarity", 0),
                    state.hard_match_scores.get("experience_score", 0),
                    set(state.job_requirements.get("required_skills", [])),
                    set(state.resume_data.get("skills", []))
                )
            ))
            
            state.stage_results["report_generation"] = {
                "final_reasoning_length": len(state.reasoning),
                "final_recommendations_count": len(state.recommendations),
                "total_processing_time": sum(state.processing_time.values()),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            state.errors.append(error_msg)
            state.stage_results["report_generation"] = {"status": "failed", "error": error_msg}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        state.processing_time["report_generation"] = processing_time
        
        return state
    
    def _should_regenerate(self, state: AnalysisState) -> str:
        """Determine if analysis should be regenerated"""
        validation_result = state.stage_results.get("validation", {})
        
        # Regenerate if validation failed and we haven't exceeded retry limit
        if not validation_result.get("is_valid", True):
            # Check if we've already tried regenerating (simple check)
            if "regeneration_attempted" not in state.stage_results:
                state.stage_results["regeneration_attempted"] = True
                return "regenerate"
        
        return "complete"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        return self.analyzer._clean_text(text)
    
    def _create_analysis_prompt(self, state: AnalysisState) -> str:
        """Create LLM analysis prompt"""
        return f"""
        Analyze the following resume against the job requirements:
        
        Resume Skills: {state.resume_data.get('skills', [])}
        Job Required Skills: {state.job_requirements.get('required_skills', [])}
        
        Hard Match Scores: {state.hard_match_scores}
        Semantic Scores: {state.semantic_scores}
        
        Provide detailed reasoning for the match quality.
        """
    
    def _generate_reasoning(self, state: AnalysisState) -> str:
        """Generate reasoning based on analysis"""
        hard_avg = sum(state.hard_match_scores.values()) / max(len(state.hard_match_scores), 1)
        semantic_avg = sum(state.semantic_scores.values()) / max(len(state.semantic_scores), 1)
        
        reasoning = f"Analysis shows hard match score of {hard_avg:.1f}% and semantic match score of {semantic_avg:.1f}%. "
        
        if hard_avg >= 70:
            reasoning += "Strong technical alignment with required skills. "
        elif hard_avg >= 50:
            reasoning += "Moderate technical alignment with some skill gaps. "
        else:
            reasoning += "Limited technical alignment, significant upskilling needed. "
        
        if semantic_avg >= 70:
            reasoning += "Excellent contextual fit for the role."
        elif semantic_avg >= 50:
            reasoning += "Good contextual understanding of role requirements."
        else:
            reasoning += "Limited contextual alignment with role expectations."
        
        return reasoning
    
    def _generate_recommendations(self, state: AnalysisState) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        hard_avg = sum(state.hard_match_scores.values()) / max(len(state.hard_match_scores), 1)
        
        if hard_avg < 60:
            recommendations.append("Focus on developing technical skills aligned with job requirements")
        
        missing_skills = set(state.job_requirements.get("required_skills", [])) - set(state.resume_data.get("skills", []))
        if missing_skills:
            recommendations.append(f"Acquire skills in: {', '.join(list(missing_skills)[:3])}")
        
        if state.hard_match_scores.get("experience_score", 0) < 50:
            recommendations.append("Gain more relevant work experience or highlight transferable skills")
        
        return recommendations
    
    def _analyze_confidence_factors(self, state: AnalysisState) -> Dict[str, float]:
        """Analyze factors affecting confidence"""
        factors = {}
        
        # Data quality factors
        factors["resume_data_quality"] = min(100, len(state.resume_data.get("skills", [])) * 10)
        factors["job_data_quality"] = min(100, len(state.job_requirements.get("required_skills", [])) * 10)
        
        # Score consistency
        scores = list(state.hard_match_scores.values()) + list(state.semantic_scores.values())
        if scores:
            variance = self._calculate_score_variance(scores)
            factors["score_consistency"] = max(0, 100 - variance * 5)
        else:
            factors["score_consistency"] = 0
        
        # Overall confidence
        factors["overall_confidence"] = sum(factors.values()) / len(factors)
        
        return factors
    
    def _assess_risks(self, state: AnalysisState) -> List[str]:
        """Assess risks in the analysis"""
        risks = []
        
        if len(state.errors) > 0:
            risks.append("Analysis errors encountered")
        
        if state.confidence < 50:
            risks.append("Low confidence in results")
        
        if not state.resume_data.get("skills"):
            risks.append("Limited skill extraction from resume")
        
        return risks
    
    def _calculate_score_variance(self, scores: List[float]) -> float:
        """Calculate variance in scores"""
        if not scores:
            return 0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance ** 0.5  # Return standard deviation
    
    def _compile_final_reasoning(self, state: AnalysisState) -> str:
        """Compile final reasoning from all stages"""
        reasoning = state.llm_analysis.get("reasoning", "")
        
        # Add processing summary
        total_time = sum(state.processing_time.values())
        reasoning += f" Analysis completed in {total_time:.2f} seconds across {len(state.processing_time)} stages."
        
        if state.errors:
            reasoning += f" Note: {len(state.errors)} issues encountered during processing."
        
        return reasoning
    
    def analyze_resume_advanced(self, resume_text: str, job_description_text: str,
                               location: str = "") -> Dict[str, Any]:
        """
        Run advanced resume analysis using LangGraph pipeline
        """
        if not LANGGRAPH_AVAILABLE:
            # Fallback to simple analysis
            logger.warning("LangGraph not available, using simple analysis")
            return self.analyzer.analyze_resume(resume_text, job_description_text)
        
        try:
            # Create initial state
            initial_state = AnalysisState(
                resume_text=resume_text,
                job_description_text=job_description_text,
                location=location
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Convert to standard format
            return {
                "overall_score": final_state.overall_score,
                "match_level": self._determine_match_level(final_state.overall_score),
                "confidence": final_state.confidence,
                "explanation": final_state.reasoning,
                "recommendations": final_state.recommendations,
                "component_scores": {
                    **final_state.hard_match_scores,
                    **final_state.semantic_scores
                },
                "advanced_analysis": {
                    "processing_stages": list(final_state.stage_results.keys()),
                    "total_processing_time": sum(final_state.processing_time.values()),
                    "errors": final_state.errors,
                    "confidence_factors": final_state.llm_analysis.get("confidence_factors", {}),
                    "risk_assessment": final_state.llm_analysis.get("risk_assessment", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            # Fallback to simple analysis
            return self.analyzer.analyze_resume(resume_text, job_description_text)
    
    def _determine_match_level(self, score: float) -> str:
        """Determine match level from score"""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 45:
            return "fair"
        else:
            return "poor"
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "langsmith_enabled": self.langsmith_client is not None,
            "pipeline_stages": len(AnalysisStage),
            "graph_nodes": 9,  # Number of nodes in the graph
            "supports_advanced_analysis": LANGGRAPH_AVAILABLE
        }