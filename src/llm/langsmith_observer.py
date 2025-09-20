"""
LangSmith Observer
Provides observability, testing, and debugging of LLM chains
"""

from typing import Dict, Any, Optional
import logging
import os

try:
    from langsmith import Client
    from langchain.callbacks import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)

class LangSmithObserver:
    """
    LangSmith integration for observability and debugging
    """
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "resume-analyzer"):
        """
        Initialize LangSmith observer
        
        Args:
            api_key: LangSmith API key
            project_name: Project name for tracking
        """
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.project_name = project_name
        self.client = None
        self.tracer = None
        
        if LANGSMITH_AVAILABLE and self.api_key:
            self._initialize_client()
        else:
            logger.warning("LangSmith not available or API key not provided")
    
    def _initialize_client(self):
        """Initialize LangSmith client and tracer"""
        try:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            # Initialize client
            self.client = Client(api_key=self.api_key)
            
            # Initialize tracer
            self.tracer = LangChainTracer(
                project_name=self.project_name,
                client=self.client
            )
            
            logger.info(f"LangSmith observer initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {str(e)}")
    
    def get_tracer(self):
        """Get LangChain tracer for chain execution"""
        return self.tracer
    
    def log_resume_analysis(self, resume_text: str, jd_text: str, 
                           results: Dict[str, Any]) -> Optional[str]:
        """
        Log resume analysis session
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
            results: Analysis results
            
        Returns:
            Session ID if successful
        """
        if not self.client:
            return None
        
        try:
            # Create a run for this analysis
            run = self.client.create_run(
                name="resume_analysis",
                run_type="llm",
                inputs={
                    "resume_length": len(resume_text),
                    "jd_length": len(jd_text),
                    "analysis_type": "full_pipeline"
                },
                outputs={
                    "overall_score": results.get('overall_score'),
                    "match_level": results.get('match_level'),
                    "confidence": results.get('confidence'),
                    "llm_verdict": results.get('llm_verdict')
                },
                project_name=self.project_name
            )
            
            logger.info(f"Logged resume analysis session: {run.id}")
            return str(run.id)
            
        except Exception as e:
            logger.error(f"Failed to log analysis session: {str(e)}")
            return None
    
    def create_dataset(self, name: str, description: str) -> Optional[str]:
        """
        Create a dataset for testing and evaluation
        
        Args:
            name: Dataset name
            description: Dataset description
            
        Returns:
            Dataset ID if successful
        """
        if not self.client:
            return None
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description
            )
            
            logger.info(f"Created dataset: {name} (ID: {dataset.id})")
            return str(dataset.id)
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return None
    
    def add_example_to_dataset(self, dataset_id: str, resume_text: str, 
                              jd_text: str, expected_score: float):
        """
        Add an example to a dataset for testing
        
        Args:
            dataset_id: Dataset ID
            resume_text: Resume content
            jd_text: Job description content
            expected_score: Expected relevance score
        """
        if not self.client:
            return
        
        try:
            self.client.create_example(
                dataset_id=dataset_id,
                inputs={
                    "resume": resume_text[:1000],  # Truncate for storage
                    "job_description": jd_text[:1000]
                },
                outputs={
                    "expected_score": expected_score
                }
            )
            
            logger.info(f"Added example to dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"Failed to add example to dataset: {str(e)}")
    
    def get_project_statistics(self) -> Dict[str, Any]:
        """
        Get project statistics and metrics
        
        Returns:
            Project statistics
        """
        if not self.client:
            return {}
        
        try:
            # Get runs for the project
            runs = list(self.client.list_runs(project_name=self.project_name, limit=100))
            
            if not runs:
                return {"message": "No runs found for project"}
            
            # Calculate basic statistics
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if run.status == "success")
            avg_score = sum(run.outputs.get('overall_score', 0) for run in runs if run.outputs) / total_runs
            
            return {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "success_rate": successful_runs / total_runs * 100,
                "average_score": avg_score,
                "project_name": self.project_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get project statistics: {str(e)}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if LangSmith observer is available"""
        return LANGSMITH_AVAILABLE and self.client is not None