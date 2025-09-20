"""
FastAPI Backend for Resume Analyzer
Enterprise-grade REST API for external integrations and scalable operations
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import sys
import os
from datetime import datetime, timedelta
import uuid
import tempfile
import shutil
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from simple_resume_analyzer import ResumeAnalyzer
from enterprise.location_manager import LocationManager, InnomaticsLocation
from enterprise.bulk_processor import BulkProcessingEngine, JobPriority
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Pydantic models for API
class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(..., description="Resume text content")
    job_description_text: str = Field(..., description="Job description text")
    location: Optional[str] = Field(None, description="Processing location")
    save_to_db: bool = Field(True, description="Whether to save results to database")

class BulkAnalysisRequest(BaseModel):
    job_description_text: str = Field(..., description="Job description text")
    location: str = Field(..., description="Processing location")
    priority: str = Field("normal", description="Job priority (low, normal, high, urgent)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class LocationFilter(BaseModel):
    location: Optional[str] = None
    region: Optional[str] = None
    skills: Optional[List[str]] = None
    industry: Optional[List[str]] = None

class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    role: str = Field(..., description="Role: student, placement_team, admin")
    location: str = Field(..., description="User location")
    full_name: str = Field(..., max_length=100)

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    total_resumes: int
    processed_resumes: int
    failed_resumes: int
    created_at: datetime
    estimated_completion: Optional[datetime]

# FastAPI app configuration
app = FastAPI(
    title="Resume Analyzer API",
    description="Enterprise Resume Analysis System for Innomatics Research Labs",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
analyzer = None
location_manager = None
bulk_processor = None
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global analyzer, location_manager, bulk_processor, db_manager
    
    try:
        # Initialize components
        analyzer = ResumeAnalyzer()
        location_manager = LocationManager()
        
        # Load configuration (you can modify this to load from file)
        config = {
            "bulk_processing": {
                "max_workers": 10,
                "max_concurrent_jobs": 5,
                "batch_size": 50,
                "timeout_seconds": 300
            },
            "database": {
                "type": "sqlite",
                "path": "data/enterprise_resume_analyzer.db"
            }
        }
        
        # Initialize database manager
        db_manager = DatabaseManager(config)
        
        # Initialize bulk processor
        bulk_processor = BulkProcessingEngine(config, analyzer)
        bulk_processor.start()
        
        logger.info("FastAPI services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global bulk_processor
    if bulk_processor:
        bulk_processor.stop()
    logger.info("FastAPI services shut down")

# Dependency for authentication (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    # In production, implement proper JWT token validation
    # For now, return a mock user
    return {
        "user_id": "demo_user",
        "username": "demo",
        "role": "placement_team",
        "location": "hyderabad"
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "analyzer": analyzer is not None,
            "location_manager": location_manager is not None,
            "bulk_processor": bulk_processor is not None and bulk_processor.running,
            "database": db_manager is not None
        }
    }

# Location management endpoints
@app.get("/api/locations")
async def get_locations(user: dict = Depends(get_current_user)):
    """Get all Innomatics office locations"""
    return {
        "locations": location_manager.get_all_locations(),
        "current_user_location": user.get("location")
    }

@app.get("/api/locations/{location_name}/stats")
async def get_location_stats(location_name: str, user: dict = Depends(get_current_user)):
    """Get statistics for a specific location"""
    location = location_manager.get_location_by_name(location_name)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    
    return location_manager.get_location_statistics(location)

# Resume analysis endpoints
@app.post("/api/analyze/single")
async def analyze_single_resume(
    request: ResumeAnalysisRequest,
    user: dict = Depends(get_current_user)
):
    """Analyze a single resume against job description"""
    try:
        # Set location if provided
        if request.location:
            location = location_manager.get_location_by_name(request.location)
            if location:
                location_manager.set_current_location(location)
        
        # Perform analysis
        result = analyzer.analyze_resume(request.resume_text, request.job_description_text)
        
        # Add location-specific metadata
        if request.location:
            result['location_metadata'] = {
                'processed_at_location': request.location,
                'location_specific_weights': location_manager.get_location_specific_scoring_weights(location)
            }
        
        return {
            "success": True,
            "result": result,
            "processed_by": user.get("username"),
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Single analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/upload")
async def analyze_uploaded_files(
    background_tasks: BackgroundTasks,
    resume_file: UploadFile = File(...),
    job_description_file: UploadFile = File(...),
    location: str = Form(...),
    user: dict = Depends(get_current_user)
):
    """Analyze uploaded resume and job description files"""
    try:
        # Validate file types
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        resume_ext = Path(resume_file.filename).suffix.lower()
        jd_ext = Path(job_description_file.filename).suffix.lower()
        
        if resume_ext not in allowed_extensions or jd_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded files temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = os.path.join(temp_dir, resume_file.filename)
            jd_path = os.path.join(temp_dir, job_description_file.filename)
            
            with open(resume_path, "wb") as f:
                shutil.copyfileobj(resume_file.file, f)
            
            with open(jd_path, "wb") as f:
                shutil.copyfileobj(job_description_file.file, f)
            
            # Perform analysis
            result = analyzer.analyze_resume_for_job(resume_path, jd_path, save_to_db=True)
            
            return {
                "success": True,
                "result": result,
                "processed_by": user.get("username"),
                "processed_at": datetime.now().isoformat(),
                "location": location
            }
            
    except Exception as e:
        logger.error(f"File upload analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Bulk processing endpoints
@app.post("/api/analyze/bulk")
async def submit_bulk_analysis(
    request: BulkAnalysisRequest,
    resume_files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user)
):
    """Submit bulk resume analysis job"""
    try:
        # Validate priority
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        priority = priority_map.get(request.priority.lower(), JobPriority.NORMAL)
        
        # Save uploaded files
        temp_dir = tempfile.mkdtemp(prefix="bulk_analysis_")
        resume_paths = []
        
        # Save job description
        jd_filename = f"job_description_{uuid.uuid4().hex[:8]}.txt"
        jd_path = os.path.join(temp_dir, jd_filename)
        with open(jd_path, "w", encoding="utf-8") as f:
            f.write(request.job_description_text)
        
        # Save resume files
        for resume_file in resume_files:
            resume_path = os.path.join(temp_dir, resume_file.filename)
            with open(resume_path, "wb") as f:
                shutil.copyfileobj(resume_file.file, f)
            resume_paths.append(resume_path)
        
        # Submit bulk job
        job_id = bulk_processor.submit_bulk_job(
            resume_files=resume_paths,
            job_description_file=jd_path,
            location=request.location,
            priority=priority,
            metadata={
                **(request.metadata or {}),
                "submitted_by": user.get("username"),
                "submitted_at": datetime.now().isoformat(),
                "temp_dir": temp_dir
            }
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": f"Bulk analysis job submitted with {len(resume_paths)} resumes",
            "estimated_completion": datetime.now() + timedelta(minutes=len(resume_paths) * 0.5)
        }
        
    except Exception as e:
        logger.error(f"Bulk analysis submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    """Get status of a bulk analysis job"""
    job = bulk_processor.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        total_resumes=job.total_resumes,
        processed_resumes=job.processed_resumes,
        failed_resumes=job.failed_resumes,
        created_at=job.created_at,
        estimated_completion=job.estimated_completion
    )

@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str, user: dict = Depends(get_current_user)):
    """Get results of a completed bulk analysis job"""
    job = bulk_processor.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    summary = bulk_processor.get_job_results_summary(job_id)
    return {
        "job_summary": summary,
        "detailed_results": job.results if len(job.results) <= 100 else job.results[:100],
        "total_results": len(job.results),
        "note": "Detailed results limited to 100 entries. Use export endpoint for full results."
    }

@app.get("/api/jobs/{job_id}/export/{format}")
async def export_job_results(job_id: str, format: str, user: dict = Depends(get_current_user)):
    """Export job results in specified format"""
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Supported formats: json, csv")
    
    exported_data = bulk_processor.export_job_results(job_id, format)
    if not exported_data:
        raise HTTPException(status_code=404, detail="Job not found or no results")
    
    # Create temporary file for download
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix=f'.{format}', 
        delete=False,
        encoding='utf-8'
    )
    temp_file.write(exported_data)
    temp_file.close()
    
    return FileResponse(
        temp_file.name,
        filename=f"job_{job_id}_results.{format}",
        media_type="application/octet-stream"
    )

@app.get("/api/jobs")
async def list_jobs(
    status: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 50,
    user: dict = Depends(get_current_user)
):
    """List bulk analysis jobs with optional filtering"""
    # Filter by status if provided
    status_filter = None
    if status:
        from enterprise.bulk_processor import JobStatus
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    
    jobs = bulk_processor.get_all_jobs(status_filter)
    
    # Filter by location if provided
    if location:
        jobs = [job for job in jobs if job.location.lower() == location.lower()]
    
    # Limit results
    jobs = jobs[:limit]
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "location": job.location,
                "total_resumes": job.total_resumes,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in jobs
        ],
        "total_jobs": len(jobs),
        "applied_filters": {
            "status": status,
            "location": location,
            "limit": limit
        }
    }

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str, user: dict = Depends(get_current_user)):
    """Cancel a pending or running bulk analysis job"""
    success = bulk_processor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"success": True, "message": f"Job {job_id} cancelled successfully"}

# Analytics and metrics endpoints
@app.get("/api/metrics/processing")
async def get_processing_metrics(user: dict = Depends(get_current_user)):
    """Get current processing metrics"""
    metrics = bulk_processor.get_processing_metrics()
    return {
        "processing_metrics": {
            "total_jobs": metrics.total_jobs,
            "completed_jobs": metrics.completed_jobs,
            "failed_jobs": metrics.failed_jobs,
            "success_rate": (metrics.completed_jobs / max(metrics.total_jobs, 1)) * 100,
            "average_processing_time": metrics.average_processing_time,
            "total_resumes_processed": metrics.total_resumes_processed,
            "resumes_per_minute": metrics.resumes_per_minute,
            "queue_length": metrics.queue_length,
            "active_workers": metrics.active_workers
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/analytics/location/{location_name}")
async def get_location_analytics(
    location_name: str,
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get analytics for a specific location"""
    location = location_manager.get_location_by_name(location_name)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Get recent jobs for this location
    jobs = bulk_processor.get_all_jobs()
    location_jobs = [
        job for job in jobs 
        if job.location.lower() == location_name.lower() and
        job.created_at >= datetime.now() - timedelta(days=days)
    ]
    
    # Calculate analytics
    total_resumes = sum(job.total_resumes for job in location_jobs)
    processed_resumes = sum(job.processed_resumes for job in location_jobs)
    failed_resumes = sum(job.failed_resumes for job in location_jobs)
    
    return {
        "location": location_name,
        "period_days": days,
        "analytics": {
            "total_jobs": len(location_jobs),
            "total_resumes": total_resumes,
            "processed_resumes": processed_resumes,
            "failed_resumes": failed_resumes,
            "success_rate": (processed_resumes / max(total_resumes, 1)) * 100,
            "average_resumes_per_job": total_resumes / max(len(location_jobs), 1),
            "job_completion_rate": len([j for j in location_jobs if j.status.value == "completed"]) / max(len(location_jobs), 1) * 100
        },
        "location_config": location_manager.get_location_statistics(location)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)