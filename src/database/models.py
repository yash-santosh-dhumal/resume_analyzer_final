"""
Database Models
SQLAlchemy models for storing resume analysis data
"""

from datetime import datetime
from typing import Dict, Any, Optional
import json

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

class JobDescription(Base):
    """Job Description model"""
    __tablename__ = 'job_descriptions'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    company = Column(String(200))
    department = Column(String(100))
    location = Column(String(100))
    
    # Content
    raw_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    
    # Extracted information
    required_skills = Column(JSON)  # List of required skills
    preferred_skills = Column(JSON)  # List of preferred skills
    keywords = Column(JSON)  # Extracted keywords
    requirements = Column(Text)
    responsibilities = Column(Text)
    
    # Metadata
    filename = Column(String(255))
    file_hash = Column(String(64))  # SHA-256 hash for duplicate detection
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    analyses = relationship("ResumeAnalysis", back_populates="job_description")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'company': self.company,
            'department': self.department,
            'location': self.location,
            'required_skills': self.required_skills,
            'preferred_skills': self.preferred_skills,
            'keywords': self.keywords,
            'filename': self.filename,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active
        }

class Resume(Base):
    """Resume model"""
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True)
    
    # Personal information
    candidate_name = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    
    # Content
    raw_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    
    # Extracted information
    skills = Column(JSON)  # List of extracted skills
    experience_years = Column(Float)
    education = Column(JSON)  # Education details
    certifications = Column(JSON)  # Certifications
    work_experience = Column(JSON)  # Work history
    keywords = Column(JSON)  # Extracted keywords
    
    # Metadata
    filename = Column(String(255))
    file_hash = Column(String(64))  # SHA-256 hash for duplicate detection
    file_type = Column(String(10))  # pdf, docx, etc.
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    analyses = relationship("ResumeAnalysis", back_populates="resume")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'candidate_name': self.candidate_name,
            'email': self.email,
            'phone': self.phone,
            'skills': self.skills,
            'experience_years': self.experience_years,
            'education': self.education,
            'certifications': self.certifications,
            'filename': self.filename,
            'file_type': self.file_type,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'is_active': self.is_active
        }

class ResumeAnalysis(Base):
    """Resume analysis results model"""
    __tablename__ = 'resume_analyses'
    
    id = Column(Integer, primary_key=True)
    
    # Foreign keys
    resume_id = Column(Integer, ForeignKey('resumes.id'), nullable=False)
    job_description_id = Column(Integer, ForeignKey('job_descriptions.id'), nullable=False)
    
    # Overall scoring
    overall_score = Column(Float, nullable=False)
    match_level = Column(String(20))  # excellent, good, fair, poor
    confidence = Column(Float)
    
    # Component scores
    hard_matching_score = Column(Float)
    soft_matching_score = Column(Float)
    llm_analysis_score = Column(Float)
    
    # Detailed results (stored as JSON)
    hard_matching_details = Column(JSON)
    soft_matching_details = Column(JSON)
    llm_analysis_details = Column(JSON)
    scoring_details = Column(JSON)
    
    # Analysis outputs
    gap_analysis = Column(Text)
    personalized_feedback = Column(Text)
    recommendations = Column(JSON)  # List of recommendations
    risk_factors = Column(JSON)  # List of risk factors
    
    # Hiring recommendation
    hiring_decision = Column(String(20))  # HIRE, INTERVIEW, MAYBE, REJECT
    hiring_confidence = Column(String(20))  # high, medium, low
    success_probability = Column(Float)
    
    # Metadata
    analysis_version = Column(String(20), default="1.0")
    processing_time = Column(Float)  # Time in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    resume = relationship("Resume", back_populates="analyses")
    job_description = relationship("JobDescription", back_populates="analyses")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'resume_id': self.resume_id,
            'job_description_id': self.job_description_id,
            'overall_score': self.overall_score,
            'match_level': self.match_level,
            'confidence': self.confidence,
            'hard_matching_score': self.hard_matching_score,
            'soft_matching_score': self.soft_matching_score,
            'llm_analysis_score': self.llm_analysis_score,
            'gap_analysis': self.gap_analysis,
            'personalized_feedback': self.personalized_feedback,
            'recommendations': self.recommendations,
            'risk_factors': self.risk_factors,
            'hiring_decision': self.hiring_decision,
            'hiring_confidence': self.hiring_confidence,
            'success_probability': self.success_probability,
            'analysis_version': self.analysis_version,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class AnalysisAuditLog(Base):
    """Audit log for tracking analysis operations"""
    __tablename__ = 'analysis_audit_logs'
    
    id = Column(Integer, primary_key=True)
    
    # Operation details
    operation_type = Column(String(50), nullable=False)  # upload, analyze, export, etc.
    user_id = Column(String(100))  # Optional user identifier
    session_id = Column(String(100))  # Session identifier
    
    # Related entities
    resume_id = Column(Integer, ForeignKey('resumes.id'))
    job_description_id = Column(Integer, ForeignKey('job_descriptions.id'))
    analysis_id = Column(Integer, ForeignKey('resume_analyses.id'))
    
    # Operation details
    operation_data = Column(JSON)  # Additional operation-specific data
    status = Column(String(20))  # success, error, warning
    error_message = Column(Text)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'operation_type': self.operation_type,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resume_id': self.resume_id,
            'job_description_id': self.job_description_id,
            'analysis_id': self.analysis_id,
            'status': self.status,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'processing_time': self.processing_time
        }

class SystemConfiguration(Base):
    """System configuration settings"""
    __tablename__ = 'system_configurations'
    
    id = Column(Integer, primary_key=True)
    
    # Configuration details
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(JSON)
    config_type = Column(String(20))  # scoring, llm, matching, etc.
    description = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'config_key': self.config_key,
            'config_value': self.config_value,
            'config_type': self.config_type,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active
        }