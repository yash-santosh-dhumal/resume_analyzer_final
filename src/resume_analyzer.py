"""
Main Resume Analyzer Application
Orchestrates all analysis components and provides unified API
"""

import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import sys

# Add src directory to path for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from parsers import DocumentParser, TextNormalizer
from matching import HardMatcher, SoftMatcher, EmbeddingGenerator
from llm import LLMReasoningEngine
from scoring import ScoringEngine, RelevanceScore
from database import DatabaseManager, ExportManager
from config.settings import load_config

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """
    Main application class that orchestrates resume analysis workflow
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the resume analyzer
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.document_parser = None
        self.text_normalizer = None
        self.hard_matcher = None
        self.soft_matcher = None
        self.embedding_generator = None
        self.llm_engine = None
        self.scoring_engine = None
        self.db_manager = None
        self.export_manager = None
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("Resume Analyzer initialized successfully")
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        try:
            # Core processing components only
            self.document_parser = DocumentParser()
            self.text_normalizer = TextNormalizer()
            
            # All heavy components will be loaded on-demand
            self.hard_matcher = None
            self.embedding_generator = None
            self.soft_matcher = None
            self.llm_engine = None
            self.scoring_engine = None
            self.db_manager = None
            self.export_manager = None
            
            logger.info("Basic components initialized successfully (all others on-demand)")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def _ensure_hard_matcher(self):
        """Initialize hard matcher on-demand"""
        if self.hard_matcher is None:
            try:
                self.hard_matcher = HardMatcher()
                logger.info("Hard matcher initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize hard matcher: {e}")
                raise
    
    def _ensure_scoring_engine(self):
        """Initialize scoring engine on-demand"""
        if self.scoring_engine is None:
            try:
                self.scoring_engine = ScoringEngine(self.config)
                logger.info("Scoring engine initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize scoring engine: {e}")
                # Don't raise - let the calling code handle the None case
                self.scoring_engine = None
    
    def _ensure_database_manager(self):
        """Initialize database manager on-demand"""
        if self.db_manager is None:
            try:
                self.db_manager = DatabaseManager(self.config)
                if self.export_manager is None:
                    self.export_manager = ExportManager(self.db_manager)
                logger.info("Database manager initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize database manager: {e}")
                raise

    def _ensure_embedding_generator(self):
        """Initialize embedding generator on-demand"""
        if self.embedding_generator is None:
            try:
                self.embedding_generator = EmbeddingGenerator(self.config)
                logger.info("Embedding generator initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize embedding generator: {e}")
                raise
    
    def _ensure_soft_matcher(self):
        """Initialize soft matcher on-demand"""
        if self.soft_matcher is None:
            try:
                self._ensure_embedding_generator()  # Soft matcher needs embedding generator
                self.soft_matcher = SoftMatcher(self.config)
                logger.info("Soft matcher initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize soft matcher: {e}")
                raise
    
    def _ensure_llm_engine(self):
        """Initialize LLM engine on-demand"""
        if self.llm_engine is None:
            try:
                self.llm_engine = LLMReasoningEngine(self.config)
                logger.info("LLM engine initialized on-demand")
            except Exception as e:
                logger.error(f"Failed to initialize LLM engine: {e}")
                raise
    
    def analyze_resume_for_job(self, resume_file_path: str, job_description_file_path: str,
                              save_to_db: bool = True) -> Dict[str, Any]:
        """
        Complete analysis workflow for a resume against a job description
        
        Args:
            resume_file_path: Path to resume file (PDF/DOCX)
            job_description_file_path: Path to job description file
            save_to_db: Whether to save results to database
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting analysis: {resume_file_path} vs {job_description_file_path}")
            
            # Step 1: Parse documents
            resume_data = self._parse_resume(resume_file_path)
            jd_data = self._parse_job_description(job_description_file_path)
            
            # Step 2: Perform matching analysis
            hard_results = self._perform_hard_matching(resume_data, jd_data)
            soft_results = self._perform_soft_matching(resume_data, jd_data)
            
            # Step 3: LLM analysis
            llm_results = self._perform_llm_analysis(resume_data, jd_data, hard_results, soft_results)
            
            # Step 4: Ensure scoring engine is initialized
            self._ensure_scoring_engine()
            
            # Step 5: Calculate final score
            relevance_score = self._calculate_relevance_score(hard_results, soft_results, llm_results)
            
            # Step 6: Generate hiring recommendation
            hiring_recommendation = self._generate_hiring_recommendation({
                'hard_matching': hard_results,
                'soft_matching': soft_results,
                'llm_analysis': llm_results,
                'relevance_score': relevance_score
            })
            
            # Step 7: Compile complete results
            processing_time = time.time() - start_time
            
            complete_results = {
                'metadata': {
                    'analysis_id': None,  # Will be set after DB save
                    'resume_filename': Path(resume_file_path).name,
                    'job_description_filename': Path(job_description_file_path).name,
                    'processing_time': processing_time,
                    'timestamp': time.time(),
                    'success': True
                },
                'resume_data': {
                    'candidate_name': resume_data.get('candidate_name'),
                    'email': resume_data.get('email'),
                    'phone': resume_data.get('phone'),
                    'skills': resume_data.get('skills', []),
                    'experience_years': resume_data.get('experience_years'),
                    'filename': resume_data.get('filename')
                },
                'job_data': {
                    'title': jd_data.get('title'),
                    'company': jd_data.get('company'),
                    'required_skills': jd_data.get('required_skills', []),
                    'filename': jd_data.get('filename')
                },
                'analysis_results': {
                    'overall_score': relevance_score.overall_score,
                    'match_level': relevance_score.match_level.value,
                    'confidence': relevance_score.confidence,
                    'explanation': relevance_score.explanation,
                    'recommendations': relevance_score.recommendations,
                    'risk_factors': relevance_score.risk_factors
                },
                'detailed_results': {
                    'hard_matching': hard_results,
                    'soft_matching': soft_results,
                    'llm_analysis': llm_results,
                    'scoring_details': {
                        'component_scores': relevance_score.component_scores,
                        'weighted_scores': relevance_score.weighted_scores
                    }
                },
                'hiring_recommendation': hiring_recommendation
            }
            
            # Step 7: Save to database if requested
            if save_to_db:
                analysis_id = self._save_analysis_to_db(complete_results, resume_data, jd_data)
                complete_results['metadata']['analysis_id'] = analysis_id
            
            logger.info(f"Analysis completed successfully in {processing_time:.2f} seconds")
            return complete_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'metadata': {
                    'resume_filename': Path(resume_file_path).name,
                    'job_description_filename': Path(job_description_file_path).name,
                    'processing_time': processing_time,
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                },
                'analysis_results': {
                    'overall_score': 0.0,
                    'match_level': 'poor',
                    'confidence': 10.0,
                    'explanation': f"Analysis failed: {str(e)}",
                    'recommendations': ["Manual review required due to processing error"],
                    'risk_factors': ["Automated analysis failed"]
                }
            }
    
    def _parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse resume file and extract information"""
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Parse document
            parsed_data = self.document_parser.parse_resume(file_path)
            
            # Normalize text
            if parsed_data.get('raw_text'):
                normalized_data = self.text_normalizer.process_document(parsed_data['raw_text'])
                parsed_data.update(normalized_data)
            
            # Add metadata
            parsed_data['filename'] = Path(file_path).name
            parsed_data['file_type'] = Path(file_path).suffix.lower().replace('.', '')
            parsed_data['file_content'] = file_content  # For hash calculation
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Resume parsing failed: {str(e)}")
            raise
    
    def _parse_job_description(self, file_path: str) -> Dict[str, Any]:
        """Parse job description file and extract information"""
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Parse document
            parsed_data = self.document_parser.parse_job_description(file_path)
            
            # Normalize text
            if parsed_data.get('raw_text'):
                normalized_data = self.text_normalizer.process_document(parsed_data['raw_text'])
                parsed_data.update(normalized_data)
            
            # Add metadata
            parsed_data['filename'] = Path(file_path).name
            parsed_data['file_content'] = file_content  # For hash calculation
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Job description parsing failed: {str(e)}")
            raise
    
    def _perform_hard_matching(self, resume_data: Dict[str, Any], 
                              jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hard matching analysis"""
        try:
            self._ensure_hard_matcher()  # Initialize on-demand
            return self.hard_matcher.analyze_match(
                resume_text=resume_data.get('processed_text', ''),
                job_description=jd_data.get('processed_text', ''),
                resume_skills=resume_data.get('skills', []),
                required_skills=jd_data.get('required_skills', [])
            )
        except Exception as e:
            logger.error(f"Hard matching failed: {str(e)}")
            return {
                'overall_score': 0,
                'keyword_score': 0,
                'skills_score': 0,
                'error': str(e)
            }
    
    def _perform_soft_matching(self, resume_data: Dict[str, Any], 
                              jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform soft matching analysis"""
        try:
            self._ensure_soft_matcher()  # Initialize on-demand
            return self.soft_matcher.analyze_semantic_similarity(
                resume_text=resume_data.get('processed_text', ''),
                job_description=jd_data.get('processed_text', '')
            )
        except Exception as e:
            logger.error(f"Soft matching failed: {str(e)}")
            return {
                'combined_semantic_score': 0,
                'semantic_score': 0,
                'embedding_score': 0,
                'error': str(e)
            }
    
    def _perform_llm_analysis(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any],
                             hard_results: Dict[str, Any], soft_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform LLM-based analysis"""
        try:
            self._ensure_llm_engine()  # Initialize on-demand
            return self.llm_engine.comprehensive_analysis(
                resume_text=resume_data.get('processed_text', ''),
                jd_text=jd_data.get('processed_text', ''),
                hard_match_results=hard_results,
                soft_match_results=soft_results
            )
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {
                'llm_verdict': 'medium',
                'llm_score': 50,
                'gap_analysis': {'error': str(e)},
                'personalized_feedback': f"LLM analysis failed: {str(e)}",
                'improvement_suggestions': [],
                'success': False
            }
    
    def _calculate_relevance_score(self, hard_results: Dict[str, Any], 
                                 soft_results: Dict[str, Any],
                                 llm_results: Dict[str, Any]) -> RelevanceScore:
        """Calculate final relevance score"""
        try:
            if self.scoring_engine is None:
                # Create a basic fallback score
                from scoring import MatchLevel
                return RelevanceScore(
                    overall_score=0.0,
                    match_level=MatchLevel.POOR,
                    confidence=0.0,
                    component_scores={},
                    weighted_scores={},
                    explanation="Scoring engine not available",
                    recommendations=["Please check system configuration"],
                    risk_factors=["Analysis system unavailable"]
                )
            return self.scoring_engine.calculate_relevance_score(
                hard_results, soft_results, llm_results
            )
        except Exception as e:
            logger.error(f"Scoring calculation failed: {str(e)}")
            if self.scoring_engine is not None:
                return self.scoring_engine._create_error_score(str(e))
            else:
                # Fallback when scoring engine is None
                from scoring import MatchLevel
                return RelevanceScore(
                    overall_score=0.0,
                    match_level=MatchLevel.POOR,
                    confidence=0.0,
                    component_scores={},
                    weighted_scores={},
                    explanation=f"Scoring failed: {str(e)}",
                    recommendations=["Please check system configuration"],
                    risk_factors=["Analysis system error"]
                )
    
    def _generate_hiring_recommendation(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hiring recommendation"""
        try:
            return self.llm_engine.generate_hiring_recommendation(all_results)
        except Exception as e:
            logger.error(f"Hiring recommendation failed: {str(e)}")
            return {
                'decision': 'MAYBE',
                'confidence': 'low',
                'reasoning': f"Recommendation generation failed: {str(e)}",
                'next_steps': ["Manual review required"],
                'risk_factors': ["Automated recommendation failed"],
                'success_probability': 50
            }
    
    def _save_analysis_to_db(self, results: Dict[str, Any], resume_data: Dict[str, Any], 
                           jd_data: Dict[str, Any]) -> Optional[int]:
        """Save analysis results to database"""
        try:
            # Save resume if not exists
            resume_obj = self.db_manager.save_resume(
                resume_data, resume_data.get('file_content', b'')
            )
            
            # Save job description if not exists
            jd_obj = self.db_manager.save_job_description(
                jd_data, jd_data.get('file_content', b'')
            )
            
            # Prepare analysis data
            analysis_data = {
                'resume_id': resume_obj.id,
                'job_description_id': jd_obj.id,
                'overall_score': results['analysis_results']['overall_score'],
                'match_level': results['analysis_results']['match_level'],
                'confidence': results['analysis_results']['confidence'],
                'hard_matching_score': results['detailed_results']['hard_matching'].get('overall_score', 0),
                'soft_matching_score': results['detailed_results']['soft_matching'].get('combined_semantic_score', 0),
                'llm_analysis_score': results['detailed_results']['llm_analysis'].get('llm_score', 0),
                'hard_matching_details': results['detailed_results']['hard_matching'],
                'soft_matching_details': results['detailed_results']['soft_matching'],
                'llm_analysis_details': results['detailed_results']['llm_analysis'],
                'scoring_details': results['detailed_results']['scoring_details'],
                'gap_analysis': results['detailed_results']['llm_analysis'].get('gap_analysis', ''),
                'personalized_feedback': results['detailed_results']['llm_analysis'].get('personalized_feedback', ''),
                'recommendations': results['analysis_results']['recommendations'],
                'risk_factors': results['analysis_results']['risk_factors'],
                'hiring_decision': results['hiring_recommendation']['decision'],
                'hiring_confidence': results['hiring_recommendation']['confidence'],
                'success_probability': results['hiring_recommendation']['success_probability'],
                'processing_time': results['metadata']['processing_time']
            }
            
            # Save analysis
            analysis_obj = self.db_manager.save_analysis(analysis_data)
            
            # Log operation
            self.db_manager.log_operation(
                'analyze_resume',
                resume_id=resume_obj.id,
                job_description_id=jd_obj.id,
                analysis_id=analysis_obj.id,
                processing_time=results['metadata']['processing_time'],
                status='success'
            )
            
            return analysis_obj.id
            
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            # Log failed operation
            try:
                self.db_manager.log_operation(
                    'analyze_resume',
                    status='error',
                    error_message=str(e)
                )
            except:
                pass
            return None
    
    # Batch processing methods
    def analyze_multiple_resumes(self, resume_files: List[str], job_description_file: str,
                               save_to_db: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes against a single job description
        
        Args:
            resume_files: List of resume file paths
            job_description_file: Job description file path
            save_to_db: Whether to save results to database
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, resume_file in enumerate(resume_files):
            try:
                logger.info(f"Processing resume {i+1}/{len(resume_files)}: {resume_file}")
                result = self.analyze_resume_for_job(resume_file, job_description_file, save_to_db)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {resume_file}: {str(e)}")
                results.append({
                    'metadata': {
                        'resume_filename': Path(resume_file).name,
                        'success': False,
                        'error': str(e)
                    }
                })
        
        return results
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get analysis results by ID from database"""
        try:
            analysis = self.db_manager.get_analysis(analysis_id)
            if not analysis:
                return None
            
            return analysis.to_dict()
        except Exception as e:
            logger.error(f"Failed to get analysis {analysis_id}: {str(e)}")
            return None
    
    def get_top_candidates_for_job(self, job_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top candidates for a job from database"""
        try:
            return self.db_manager.get_top_candidates(job_id, limit)
        except Exception as e:
            logger.error(f"Failed to get top candidates for job {job_id}: {str(e)}")
            return []
    
    def export_results(self, format_type: str, output_path: str, **kwargs) -> str:
        """
        Export analysis results
        
        Args:
            format_type: Export format ('csv', 'excel', 'json')
            output_path: Output file path
            **kwargs: Additional filter parameters
            
        Returns:
            Path to exported file
        """
        try:
            if format_type.lower() == 'csv':
                return self.export_manager.export_to_csv(output_path, **kwargs)
            elif format_type.lower() == 'excel':
                return self.export_manager.export_to_excel(output_path, **kwargs)
            elif format_type.lower() == 'json':
                return self.export_manager.export_to_json(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise
    
    def get_system_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get system usage statistics"""
        try:
            return self.db_manager.get_analysis_statistics(days)
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    def create_candidate_report(self, analysis_id: int) -> Dict[str, Any]:
        """Create detailed candidate report"""
        try:
            return self.export_manager.create_candidate_report(analysis_id)
        except Exception as e:
            logger.error(f"Failed to create candidate report: {str(e)}")
            raise
    
    def create_job_summary_report(self, job_id: int) -> Dict[str, Any]:
        """Create job summary report"""
        try:
            return self.export_manager.create_job_summary_report(job_id)
        except Exception as e:
            logger.error(f"Failed to create job summary report: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health status"""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        # Check each component
        components_to_check = [
            ('document_parser', self.document_parser),
            ('text_normalizer', self.text_normalizer),
            ('hard_matcher', self.hard_matcher),
            ('soft_matcher', self.soft_matcher),
            ('llm_engine', self.llm_engine),
            ('scoring_engine', self.scoring_engine),
            ('db_manager', self.db_manager)
        ]
        
        for name, component in components_to_check:
            try:
                if component is not None:
                    health_status['components'][name] = 'healthy'
                else:
                    health_status['components'][name] = 'not_initialized'
                    health_status['status'] = 'degraded'
            except Exception as e:
                health_status['components'][name] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
        
        # Check database connectivity
        try:
            session = self.db_manager.get_session()
            session.close()
            health_status['components']['database'] = 'healthy'
        except Exception as e:
            health_status['components']['database'] = f'error: {str(e)}'
            health_status['status'] = 'unhealthy'
        
        return health_status