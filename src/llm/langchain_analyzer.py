"""
LangChain Integration Module
Provides LangChain chains and prompts for resume analysis
"""

from typing import Dict, Any, List, Optional
import logging
import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langsmith_observer import LangSmithObserver
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResumeAnalysisOutputParser(BaseOutputParser):
    """Custom output parser for resume analysis results"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured format"""
        try:
            # Simple parsing logic - in production, use more robust parsing
            lines = text.strip().split('\n')
            result = {
                'verdict': 'medium',
                'score': 50,
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'missing_skills': []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if 'verdict:' in line.lower():
                    verdict = line.split(':', 1)[1].strip().lower()
                    result['verdict'] = verdict
                elif 'score:' in line.lower():
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        score = int(''.join(filter(str.isdigit, score_text)))
                        result['score'] = min(100, max(0, score))
                    except:
                        pass
                elif 'strengths:' in line.lower():
                    current_section = 'strengths'
                elif 'weaknesses:' in line.lower():
                    current_section = 'weaknesses'
                elif 'recommendations:' in line.lower():
                    current_section = 'recommendations'
                elif 'missing skills:' in line.lower():
                    current_section = 'missing_skills'
                elif line.startswith('-') or line.startswith('•'):
                    # Bullet point
                    item = line[1:].strip()
                    if current_section and item:
                        result[current_section].append(item)
                elif current_section and line:
                    # Continuation of current section
                    result[current_section].append(line)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {str(e)}")
            return {
                'verdict': 'medium',
                'score': 50,
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'missing_skills': [],
                'parse_error': str(e)
            }

class LangChainAnalyzer:
    """
    LangChain-based analyzer for resume-JD matching
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LangChain analyzer
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key
        self.model = model
        self.llm = None
        self.chains = {}
        self.output_parser = ResumeAnalysisOutputParser()
        
        # Initialize LangSmith observer
        self.langsmith_observer = None
        if LANGSMITH_AVAILABLE:
            self.langsmith_observer = LangSmithObserver(
                api_key=None,  # Will use environment variable
                project_name="innomatics-resume-analyzer"
            )
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_llm()
            self._create_chains()
        else:
            logger.error("LangChain not available. Install with: pip install langchain")
    
    def _initialize_llm(self):
        """Initialize LangChain LLM"""
        try:
            if self.api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=self.api_key,
                    model_name=self.model,
                    temperature=0.3,
                    max_tokens=1000
                )
                logger.info(f"LangChain LLM initialized: {self.model}")
            else:
                logger.warning("No API key provided for LangChain LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {str(e)}")
    
    def _create_chains(self):
        """Create LangChain chains for different analysis tasks"""
        if not self.llm:
            return
        
        # Resume Analysis Chain
        resume_analysis_template = """
        You are an expert HR recruiter analyzing how well a resume matches a job description.
        
        Job Description:
        {job_description}
        
        Resume:
        {resume}
        
        Please provide a comprehensive analysis with the following structure:
        
        Verdict: [high/medium/low] - Overall fit for the position
        Score: [0-100] - Numerical score representing match quality
        
        Strengths:
        - List key strengths and relevant qualifications
        - Highlight experiences that align well with requirements
        - Note impressive achievements or skills
        
        Weaknesses:
        - Identify gaps in experience or skills
        - Note areas where candidate falls short
        - Highlight missing qualifications
        
        Recommendations:
        - Specific suggestions for improving the resume
        - Skills or experiences to highlight better
        - Areas for professional development
        
        Missing Skills:
        - Critical skills mentioned in JD but absent in resume
        - Technologies or tools candidate should learn
        - Certifications that would strengthen candidacy
        
        Provide specific, actionable feedback that helps both the candidate and recruiter.
        """
        
        resume_prompt = ChatPromptTemplate.from_template(resume_analysis_template)
        self.chains['resume_analysis'] = LLMChain(
            llm=self.llm,
            prompt=resume_prompt,
            output_parser=self.output_parser
        )
        
        # Gap Analysis Chain
        gap_analysis_template = """
        Analyze the gaps between this resume and job requirements:
        
        Job Requirements:
        {job_requirements}
        
        Resume Content:
        {resume_content}
        
        Identify specific gaps in:
        1. Technical skills
        2. Experience level
        3. Education/certifications
        4. Industry knowledge
        5. Soft skills
        
        For each gap, provide:
        - Severity (Critical/Important/Nice-to-have)
        - Suggested actions for the candidate
        - Timeline for improvement
        
        Format your response as structured bullet points.
        """
        
        gap_prompt = ChatPromptTemplate.from_template(gap_analysis_template)
        self.chains['gap_analysis'] = LLMChain(
            llm=self.llm,
            prompt=gap_prompt
        )
        
        # Feedback Generation Chain
        feedback_template = """
        Generate personalized feedback for a job candidate based on their resume analysis:
        
        Candidate Resume: {resume}
        Job Position: {position}
        Analysis Results: {analysis_results}
        
        Create encouraging yet honest feedback that:
        1. Acknowledges their strengths
        2. Explains areas for improvement
        3. Provides specific, actionable recommendations
        4. Suggests learning resources or next steps
        5. Maintains a positive, growth-oriented tone
        
        The feedback should be professional but supportive, helping the candidate understand 
        how to better position themselves for this type of role.
        """
        
        feedback_prompt = ChatPromptTemplate.from_template(feedback_template)
        self.chains['feedback_generation'] = LLMChain(
            llm=self.llm,
            prompt=feedback_prompt
        )
        
        logger.info("LangChain analysis chains created")
    
    def analyze_resume_match(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Analyze resume-JD match using LangChain
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
        
        Returns:
            Comprehensive analysis results
        """
        if not LANGCHAIN_AVAILABLE or not self.llm:
            return {
                'error': 'LangChain not available or LLM not initialized',
                'verdict': 'medium',
                'score': 50
            }
        
        try:
            with get_openai_callback() as cb:
                # Run resume analysis chain
                analysis_result = self.chains['resume_analysis'].run(
                    job_description=jd_text,
                    resume=resume_text
                )
                
                # Add cost information
                analysis_result['llm_usage'] = {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                }
                
                logger.info(f"LangChain analysis completed. Tokens used: {cb.total_tokens}")
                
                return analysis_result
                
        except Exception as e:
            logger.error(f"LangChain resume analysis failed: {str(e)}")
            return {
                'error': str(e),
                'verdict': 'medium',
                'score': 50
            }
    
    def generate_gap_analysis(self, resume_text: str, job_requirements: str) -> str:
        """
        Generate detailed gap analysis
        
        Args:
            resume_text: Resume content
            job_requirements: Job requirements text
        
        Returns:
            Gap analysis text
        """
        if not LANGCHAIN_AVAILABLE or 'gap_analysis' not in self.chains:
            return "Gap analysis not available"
        
        try:
            return self.chains['gap_analysis'].run(
                job_requirements=job_requirements,
                resume_content=resume_text
            )
        except Exception as e:
            logger.error(f"Gap analysis failed: {str(e)}")
            return f"Gap analysis failed: {str(e)}"
    
    def generate_personalized_feedback(self, resume_text: str, position: str, 
                                     analysis_results: Dict[str, Any]) -> str:
        """
        Generate personalized feedback for candidate
        
        Args:
            resume_text: Resume content
            position: Job position title
            analysis_results: Previous analysis results
        
        Returns:
            Personalized feedback text
        """
        if not LANGCHAIN_AVAILABLE or 'feedback_generation' not in self.chains:
            return "Personalized feedback not available"
        
        try:
            return self.chains['feedback_generation'].run(
                resume=resume_text,
                position=position,
                analysis_results=str(analysis_results)
            )
        except Exception as e:
            logger.error(f"Feedback generation failed: {str(e)}")
            return f"Feedback generation failed: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if LangChain analyzer is available"""
        return LANGCHAIN_AVAILABLE and self.llm is not None