"""
LangGraph Pipeline
Structured stateful pipelines for resume-JD analysis workflow
"""

from typing import Dict, Any, List, Optional, TypedDict
import logging
from enum import Enum

try:
    from langgraph.graph import Graph, StateGraph
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

class AnalysisState(TypedDict):
    """State for resume analysis pipeline"""
    resume_text: str
    jd_text: str
    hard_match_results: Dict[str, Any]
    soft_match_results: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    final_score: float
    recommendations: List[str]
    errors: List[str]
    current_step: str

class AnalysisStep(Enum):
    """Analysis pipeline steps"""
    START = "start"
    HARD_MATCHING = "hard_matching"
    SOFT_MATCHING = "soft_matching"
    LLM_ANALYSIS = "llm_analysis"
    SCORING = "scoring"
    RECOMMENDATIONS = "recommendations"
    END = "end"

class ResumeAnalysisPipeline:
    """
    LangGraph-based structured pipeline for resume analysis
    """
    
    def __init__(self, hard_matcher, soft_matcher, llm_engine, scoring_engine):
        """
        Initialize analysis pipeline
        
        Args:
            hard_matcher: Hard matching component
            soft_matcher: Soft matching component
            llm_engine: LLM reasoning engine
            scoring_engine: Scoring engine
        """
        self.hard_matcher = hard_matcher
        self.soft_matcher = soft_matcher
        self.llm_engine = llm_engine
        self.scoring_engine = scoring_engine
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            logger.warning("LangGraph not available. Using fallback pipeline.")
    
    def _build_graph(self):
        """Build the analysis pipeline graph"""
        try:
            # Create state graph
            workflow = StateGraph(AnalysisState)
            
            # Add nodes
            workflow.add_node("hard_matching", self._hard_matching_node)
            workflow.add_node("soft_matching", self._soft_matching_node)
            workflow.add_node("llm_analysis", self._llm_analysis_node)
            workflow.add_node("scoring", self._scoring_node)
            workflow.add_node("recommendations", self._recommendations_node)
            
            # Define edges
            workflow.add_edge("hard_matching", "soft_matching")
            workflow.add_edge("soft_matching", "llm_analysis")
            workflow.add_edge("llm_analysis", "scoring")
            workflow.add_edge("scoring", "recommendations")
            
            # Set entry point
            workflow.set_entry_point("hard_matching")
            workflow.set_finish_point("recommendations")
            
            # Compile the graph
            self.graph = workflow.compile()
            
            logger.info("LangGraph analysis pipeline built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build LangGraph pipeline: {str(e)}")
    
    def _hard_matching_node(self, state: AnalysisState) -> AnalysisState:
        """Hard matching analysis node"""
        try:
            logger.info("Executing hard matching analysis")
            
            # Extract resume and JD data
            resume_data = {"processed_text": state["resume_text"]}
            jd_data = {"processed_text": state["jd_text"]}
            
            # Perform hard matching
            hard_results = self.hard_matcher.analyze_match(resume_data, jd_data)
            
            # Update state
            state["hard_match_results"] = hard_results
            state["current_step"] = AnalysisStep.HARD_MATCHING.value
            
            logger.info(f"Hard matching completed. Score: {hard_results.get('overall_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Hard matching node failed: {str(e)}")
            state["errors"].append(f"Hard matching error: {str(e)}")
        
        return state
    
    def _soft_matching_node(self, state: AnalysisState) -> AnalysisState:
        """Soft matching analysis node"""
        try:
            logger.info("Executing soft matching analysis")
            
            # Extract resume and JD data
            resume_data = {"processed_text": state["resume_text"]}
            jd_data = {"processed_text": state["jd_text"]}
            
            # Perform soft matching
            soft_results = self.soft_matcher.analyze_semantic_similarity(resume_data, jd_data)
            
            # Update state
            state["soft_match_results"] = soft_results
            state["current_step"] = AnalysisStep.SOFT_MATCHING.value
            
            logger.info(f"Soft matching completed. Score: {soft_results.get('combined_semantic_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Soft matching node failed: {str(e)}")
            state["errors"].append(f"Soft matching error: {str(e)}")
        
        return state
    
    def _llm_analysis_node(self, state: AnalysisState) -> AnalysisState:
        """LLM analysis node"""
        try:
            logger.info("Executing LLM analysis")
            
            # Perform LLM analysis
            llm_results = self.llm_engine.comprehensive_analysis(
                resume_text=state["resume_text"],
                jd_text=state["jd_text"],
                hard_match_results=state["hard_match_results"],
                soft_match_results=state["soft_match_results"]
            )
            
            # Update state
            state["llm_analysis"] = llm_results
            state["current_step"] = AnalysisStep.LLM_ANALYSIS.value
            
            logger.info(f"LLM analysis completed. Verdict: {llm_results.get('llm_verdict', 'N/A')}")
            
        except Exception as e:
            logger.error(f"LLM analysis node failed: {str(e)}")
            state["errors"].append(f"LLM analysis error: {str(e)}")
        
        return state
    
    def _scoring_node(self, state: AnalysisState) -> AnalysisState:
        """Scoring calculation node"""
        try:
            logger.info("Executing scoring calculation")
            
            # Calculate final score
            relevance_score = self.scoring_engine.calculate_score(
                hard_results=state["hard_match_results"],
                soft_results=state["soft_match_results"],
                llm_results=state["llm_analysis"]
            )
            
            # Update state
            state["final_score"] = relevance_score.overall_score
            state["current_step"] = AnalysisStep.SCORING.value
            
            logger.info(f"Scoring completed. Final score: {relevance_score.overall_score}")
            
        except Exception as e:
            logger.error(f"Scoring node failed: {str(e)}")
            state["errors"].append(f"Scoring error: {str(e)}")
        
        return state
    
    def _recommendations_node(self, state: AnalysisState) -> AnalysisState:
        """Recommendations generation node"""
        try:
            logger.info("Generating recommendations")
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # From hard matching
            if state["hard_match_results"]:
                hard_recs = state["hard_match_results"].get("recommendations", [])
                recommendations.extend(hard_recs)
            
            # From LLM analysis
            if state["llm_analysis"]:
                llm_recs = state["llm_analysis"].get("improvement_suggestions", [])
                recommendations.extend(llm_recs)
            
            # Update state
            state["recommendations"] = recommendations
            state["current_step"] = AnalysisStep.RECOMMENDATIONS.value
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Recommendations node failed: {str(e)}")
            state["errors"].append(f"Recommendations error: {str(e)}")
        
        return state
    
    def analyze(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
            
        Returns:
            Complete analysis results
        """
        # Initialize state
        initial_state: AnalysisState = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "hard_match_results": {},
            "soft_match_results": {},
            "llm_analysis": {},
            "final_score": 0.0,
            "recommendations": [],
            "errors": [],
            "current_step": AnalysisStep.START.value
        }
        
        try:
            if self.graph and LANGGRAPH_AVAILABLE:
                # Use LangGraph pipeline
                logger.info("Running LangGraph structured pipeline")
                final_state = self.graph.invoke(initial_state)
            else:
                # Fallback to sequential execution
                logger.info("Running fallback sequential pipeline")
                final_state = self._fallback_pipeline(initial_state)
            
            # Compile results
            return {
                "overall_score": final_state["final_score"],
                "hard_match_results": final_state["hard_match_results"],
                "soft_match_results": final_state["soft_match_results"],
                "llm_analysis": final_state["llm_analysis"],
                "recommendations": final_state["recommendations"],
                "errors": final_state["errors"],
                "pipeline_status": "completed",
                "final_step": final_state["current_step"]
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                "overall_score": 0.0,
                "errors": [f"Pipeline error: {str(e)}"],
                "pipeline_status": "failed",
                "final_step": initial_state["current_step"]
            }
    
    def _fallback_pipeline(self, state: AnalysisState) -> AnalysisState:
        """Fallback sequential pipeline when LangGraph is not available"""
        try:
            # Execute nodes sequentially
            state = self._hard_matching_node(state)
            state = self._soft_matching_node(state)
            state = self._llm_analysis_node(state)
            state = self._scoring_node(state)
            state = self._recommendations_node(state)
            
            state["current_step"] = AnalysisStep.END.value
            
        except Exception as e:
            logger.error(f"Fallback pipeline failed: {str(e)}")
            state["errors"].append(f"Fallback pipeline error: {str(e)}")
        
        return state
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline configuration status"""
        return {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "graph_compiled": self.graph is not None,
            "components_initialized": all([
                self.hard_matcher is not None,
                self.soft_matcher is not None,
                self.llm_engine is not None,
                self.scoring_engine is not None
            ])
        }