"""
Database Manager
Handles database connections, operations, and data management
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from sqlalchemy import create_engine, and_, or_, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models import Base, Resume, JobDescription, ResumeAnalysis, AnalysisAuditLog, SystemConfiguration

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Main database manager for resume analysis system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database manager
        
        Args:
            config: Configuration dictionary with database settings
        """
        self.config = config
        self.engine = None
        self.Session = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            # Get database URL from config
            db_config = self.config.get('database', {})
            db_url = self._construct_database_url(db_config)
            
            # Create engine
            self.engine = create_engine(
                db_url,
                echo=db_config.get('echo', False),
                pool_size=db_config.get('pool_size', 5),
                max_overflow=db_config.get('max_overflow', 10)
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _construct_database_url(self, db_config: Dict[str, Any]) -> str:
        """Construct database URL from configuration"""
        db_type = db_config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            db_path = db_config.get('path', 'data/resume_analyzer.db')
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
        
        elif db_type == 'postgresql':
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            database = db_config.get('database', 'resume_analyzer')
            username = db_config.get('username', 'postgres')
            password = db_config.get('password', '')
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def get_session(self) -> Session:
        """Get new database session"""
        return self.Session()
    
    # Resume operations
    def save_resume(self, resume_data: Dict[str, Any], file_content: bytes) -> Resume:
        """
        Save resume to database
        
        Args:
            resume_data: Extracted resume data
            file_content: Original file content for hash calculation
            
        Returns:
            Saved Resume object
        """
        session = self.get_session()
        try:
            # Calculate file hash
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check for existing resume with same hash
            existing = session.query(Resume).filter(Resume.file_hash == file_hash).first()
            if existing:
                logger.info(f"Resume already exists: {existing.filename}")
                return existing
            
            # Create new resume
            resume = Resume(
                candidate_name=resume_data.get('candidate_name'),
                email=resume_data.get('email'),
                phone=resume_data.get('phone'),
                raw_text=resume_data.get('raw_text', ''),
                processed_text=resume_data.get('processed_text', ''),
                skills=resume_data.get('skills', []),
                experience_years=resume_data.get('experience_years'),
                education=resume_data.get('education', []),
                certifications=resume_data.get('certifications', []),
                work_experience=resume_data.get('work_experience', []),
                keywords=resume_data.get('keywords', []),
                filename=resume_data.get('filename'),
                file_hash=file_hash,
                file_type=resume_data.get('file_type', 'unknown')
            )
            
            session.add(resume)
            session.commit()
            
            logger.info(f"Resume saved successfully: {resume.filename}")
            return resume
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save resume: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_resume(self, resume_id: int) -> Optional[Resume]:
        """Get resume by ID"""
        session = self.get_session()
        try:
            return session.query(Resume).filter(Resume.id == resume_id).first()
        finally:
            session.close()
    
    def get_resumes(self, limit: int = 100, offset: int = 0, 
                   active_only: bool = True) -> List[Resume]:
        """Get list of resumes"""
        session = self.get_session()
        try:
            query = session.query(Resume)
            if active_only:
                query = query.filter(Resume.is_active == True)
            
            return query.order_by(desc(Resume.uploaded_at)).offset(offset).limit(limit).all()
        finally:
            session.close()
    
    def search_resumes(self, search_term: str, skills: List[str] = None) -> List[Resume]:
        """Search resumes by name, skills, or keywords"""
        session = self.get_session()
        try:
            query = session.query(Resume).filter(Resume.is_active == True)
            
            # Text search
            if search_term:
                query = query.filter(
                    or_(
                        Resume.candidate_name.ilike(f'%{search_term}%'),
                        Resume.processed_text.ilike(f'%{search_term}%')
                    )
                )
            
            # Skills search (PostgreSQL JSON operations)
            if skills and self.config.get('database', {}).get('type') == 'postgresql':
                for skill in skills:
                    query = query.filter(Resume.skills.op('?')(skill))
            
            return query.order_by(desc(Resume.uploaded_at)).limit(50).all()
        finally:
            session.close()
    
    # Job Description operations
    def save_job_description(self, jd_data: Dict[str, Any], file_content: bytes) -> JobDescription:
        """
        Save job description to database
        
        Args:
            jd_data: Extracted job description data
            file_content: Original file content for hash calculation
            
        Returns:
            Saved JobDescription object
        """
        session = self.get_session()
        try:
            # Calculate file hash
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check for existing JD with same hash
            existing = session.query(JobDescription).filter(JobDescription.file_hash == file_hash).first()
            if existing:
                logger.info(f"Job description already exists: {existing.filename}")
                return existing
            
            # Create new job description
            jd = JobDescription(
                title=jd_data.get('title', ''),
                company=jd_data.get('company', ''),
                department=jd_data.get('department', ''),
                location=jd_data.get('location', ''),
                raw_text=jd_data.get('raw_text', ''),
                processed_text=jd_data.get('processed_text', ''),
                required_skills=jd_data.get('required_skills', []),
                preferred_skills=jd_data.get('preferred_skills', []),
                keywords=jd_data.get('keywords', []),
                requirements=jd_data.get('requirements', ''),
                responsibilities=jd_data.get('responsibilities', ''),
                filename=jd_data.get('filename'),
                file_hash=file_hash
            )
            
            session.add(jd)
            session.commit()
            
            logger.info(f"Job description saved successfully: {jd.filename}")
            return jd
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save job description: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_job_description(self, jd_id: int) -> Optional[JobDescription]:
        """Get job description by ID"""
        session = self.get_session()
        try:
            return session.query(JobDescription).filter(JobDescription.id == jd_id).first()
        finally:
            session.close()
    
    def get_job_descriptions(self, limit: int = 100, offset: int = 0,
                           active_only: bool = True) -> List[JobDescription]:
        """Get list of job descriptions"""
        session = self.get_session()
        try:
            query = session.query(JobDescription)
            if active_only:
                query = query.filter(JobDescription.is_active == True)
            
            return query.order_by(desc(JobDescription.created_at)).offset(offset).limit(limit).all()
        finally:
            session.close()
    
    # Analysis operations
    def save_analysis(self, analysis_data: Dict[str, Any]) -> ResumeAnalysis:
        """
        Save analysis results to database
        
        Args:
            analysis_data: Complete analysis results
            
        Returns:
            Saved ResumeAnalysis object
        """
        session = self.get_session()
        try:
            analysis = ResumeAnalysis(
                resume_id=analysis_data['resume_id'],
                job_description_id=analysis_data['job_description_id'],
                overall_score=analysis_data['overall_score'],
                match_level=analysis_data.get('match_level', 'fair'),
                confidence=analysis_data.get('confidence', 50.0),
                hard_matching_score=analysis_data.get('hard_matching_score', 0.0),
                soft_matching_score=analysis_data.get('soft_matching_score', 0.0),
                llm_analysis_score=analysis_data.get('llm_analysis_score', 0.0),
                hard_matching_details=analysis_data.get('hard_matching_details', {}),
                soft_matching_details=analysis_data.get('soft_matching_details', {}),
                llm_analysis_details=analysis_data.get('llm_analysis_details', {}),
                scoring_details=analysis_data.get('scoring_details', {}),
                gap_analysis=analysis_data.get('gap_analysis', ''),
                personalized_feedback=analysis_data.get('personalized_feedback', ''),
                recommendations=analysis_data.get('recommendations', []),
                risk_factors=analysis_data.get('risk_factors', []),
                hiring_decision=analysis_data.get('hiring_decision', 'MAYBE'),
                hiring_confidence=analysis_data.get('hiring_confidence', 'medium'),
                success_probability=analysis_data.get('success_probability', 50.0),
                processing_time=analysis_data.get('processing_time', 0.0)
            )
            
            session.add(analysis)
            session.commit()
            
            logger.info(f"Analysis saved successfully: ID {analysis.id}")
            return analysis
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save analysis: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_analysis(self, analysis_id: int) -> Optional[ResumeAnalysis]:
        """Get analysis by ID"""
        session = self.get_session()
        try:
            return session.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        finally:
            session.close()
    
    def get_analyses_for_job(self, job_id: int, limit: int = 100) -> List[ResumeAnalysis]:
        """Get all analyses for a specific job"""
        session = self.get_session()
        try:
            return session.query(ResumeAnalysis)\
                .filter(ResumeAnalysis.job_description_id == job_id)\
                .order_by(desc(ResumeAnalysis.overall_score))\
                .limit(limit).all()
        finally:
            session.close()
    
    def get_analyses_for_resume(self, resume_id: int) -> List[ResumeAnalysis]:
        """Get all analyses for a specific resume"""
        session = self.get_session()
        try:
            return session.query(ResumeAnalysis)\
                .filter(ResumeAnalysis.resume_id == resume_id)\
                .order_by(desc(ResumeAnalysis.created_at)).all()
        finally:
            session.close()
    
    def get_top_candidates(self, job_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top candidates for a job with resume details"""
        session = self.get_session()
        try:
            results = session.query(ResumeAnalysis, Resume)\
                .join(Resume, ResumeAnalysis.resume_id == Resume.id)\
                .filter(ResumeAnalysis.job_description_id == job_id)\
                .order_by(desc(ResumeAnalysis.overall_score))\
                .limit(limit).all()
            
            candidates = []
            for analysis, resume in results:
                candidates.append({
                    'analysis': analysis.to_dict(),
                    'resume': resume.to_dict()
                })
            
            return candidates
        finally:
            session.close()
    
    # Audit logging
    def log_operation(self, operation_type: str, **kwargs) -> AnalysisAuditLog:
        """Log an operation for audit trail"""
        session = self.get_session()
        try:
            log_entry = AnalysisAuditLog(
                operation_type=operation_type,
                user_id=kwargs.get('user_id'),
                session_id=kwargs.get('session_id'),
                resume_id=kwargs.get('resume_id'),
                job_description_id=kwargs.get('job_description_id'),
                analysis_id=kwargs.get('analysis_id'),
                operation_data=kwargs.get('operation_data', {}),
                status=kwargs.get('status', 'success'),
                error_message=kwargs.get('error_message'),
                processing_time=kwargs.get('processing_time')
            )
            
            session.add(log_entry)
            session.commit()
            
            return log_entry
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log operation: {str(e)}")
            raise
        finally:
            session.close()
    
    # Export functionality
    def export_analyses_to_dict(self, job_id: Optional[int] = None, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Export analysis data for CSV/Excel export
        
        Args:
            job_id: Optional job ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of analysis dictionaries with related data
        """
        session = self.get_session()
        try:
            query = session.query(ResumeAnalysis, Resume, JobDescription)\
                .join(Resume, ResumeAnalysis.resume_id == Resume.id)\
                .join(JobDescription, ResumeAnalysis.job_description_id == JobDescription.id)
            
            # Apply filters
            if job_id:
                query = query.filter(ResumeAnalysis.job_description_id == job_id)
            
            if start_date:
                query = query.filter(ResumeAnalysis.created_at >= start_date)
            
            if end_date:
                query = query.filter(ResumeAnalysis.created_at <= end_date)
            
            results = query.order_by(desc(ResumeAnalysis.overall_score)).all()
            
            export_data = []
            for analysis, resume, job_desc in results:
                export_data.append({
                    # Analysis data
                    'analysis_id': analysis.id,
                    'overall_score': analysis.overall_score,
                    'match_level': analysis.match_level,
                    'confidence': analysis.confidence,
                    'hard_matching_score': analysis.hard_matching_score,
                    'soft_matching_score': analysis.soft_matching_score,
                    'llm_analysis_score': analysis.llm_analysis_score,
                    'hiring_decision': analysis.hiring_decision,
                    'success_probability': analysis.success_probability,
                    'analysis_date': analysis.created_at.isoformat() if analysis.created_at else None,
                    
                    # Resume data
                    'candidate_name': resume.candidate_name,
                    'candidate_email': resume.email,
                    'candidate_phone': resume.phone,
                    'experience_years': resume.experience_years,
                    'resume_filename': resume.filename,
                    
                    # Job data
                    'job_title': job_desc.title,
                    'company': job_desc.company,
                    'department': job_desc.department,
                    'location': job_desc.location,
                    
                    # Analysis details
                    'gap_analysis': analysis.gap_analysis,
                    'personalized_feedback': analysis.personalized_feedback,
                    'recommendations': '; '.join(analysis.recommendations or []),
                    'risk_factors': '; '.join(analysis.risk_factors or [])
                })
            
            return export_data
            
        finally:
            session.close()
    
    # Statistics and reporting
    def get_analysis_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get analysis statistics for the specified period"""
        session = self.get_session()
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Basic counts
            total_analyses = session.query(ResumeAnalysis)\
                .filter(ResumeAnalysis.created_at >= since_date).count()
            
            total_resumes = session.query(Resume)\
                .filter(Resume.uploaded_at >= since_date).count()
            
            total_jobs = session.query(JobDescription)\
                .filter(JobDescription.created_at >= since_date).count()
            
            # Score statistics
            score_stats = session.query(
                func.avg(ResumeAnalysis.overall_score).label('avg_score'),
                func.min(ResumeAnalysis.overall_score).label('min_score'),
                func.max(ResumeAnalysis.overall_score).label('max_score')
            ).filter(ResumeAnalysis.created_at >= since_date).first()
            
            # Match level distribution
            match_distribution = session.query(
                ResumeAnalysis.match_level,
                func.count(ResumeAnalysis.id).label('count')
            ).filter(ResumeAnalysis.created_at >= since_date)\
            .group_by(ResumeAnalysis.match_level).all()
            
            # Hiring decision distribution
            hiring_distribution = session.query(
                ResumeAnalysis.hiring_decision,
                func.count(ResumeAnalysis.id).label('count')
            ).filter(ResumeAnalysis.created_at >= since_date)\
            .group_by(ResumeAnalysis.hiring_decision).all()
            
            return {
                'period_days': days,
                'total_analyses': total_analyses,
                'total_resumes': total_resumes,
                'total_jobs': total_jobs,
                'average_score': float(score_stats.avg_score) if score_stats.avg_score else 0,
                'min_score': float(score_stats.min_score) if score_stats.min_score else 0,
                'max_score': float(score_stats.max_score) if score_stats.max_score else 0,
                'match_level_distribution': {level: count for level, count in match_distribution},
                'hiring_decision_distribution': {decision: count for decision, count in hiring_distribution}
            }
            
        finally:
            session.close()
    
    # Configuration management
    def get_config_value(self, config_key: str, default_value: Any = None) -> Any:
        """Get configuration value from database"""
        session = self.get_session()
        try:
            config = session.query(SystemConfiguration)\
                .filter(SystemConfiguration.config_key == config_key).first()
            
            return config.config_value if config else default_value
        finally:
            session.close()
    
    def set_config_value(self, config_key: str, config_value: Any, 
                        config_type: str = 'general', description: str = '') -> SystemConfiguration:
        """Set configuration value in database"""
        session = self.get_session()
        try:
            config = session.query(SystemConfiguration)\
                .filter(SystemConfiguration.config_key == config_key).first()
            
            if config:
                config.config_value = config_value
                config.updated_at = datetime.utcnow()
            else:
                config = SystemConfiguration(
                    config_key=config_key,
                    config_value=config_value,
                    config_type=config_type,
                    description=description
                )
                session.add(config)
            
            session.commit()
            return config
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to set config value: {str(e)}")
            raise
        finally:
            session.close()
    
    def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old data beyond specified days"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old audit logs
            deleted_logs = session.query(AnalysisAuditLog)\
                .filter(AnalysisAuditLog.timestamp < cutoff_date).count()
            
            session.query(AnalysisAuditLog)\
                .filter(AnalysisAuditLog.timestamp < cutoff_date).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {deleted_logs} old audit log entries")
            
            return {
                'deleted_audit_logs': deleted_logs
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup old data: {str(e)}")
            raise
        finally:
            session.close()