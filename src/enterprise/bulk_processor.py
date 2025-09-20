"""
Bulk Processing Engine
Handles large-scale resume analysis for thousands of weekly applications
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, PriorityQueue
import time

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Bulk processing job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Job priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0

@dataclass
class BulkAnalysisJob:
    """Represents a bulk analysis job"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = "bulk_resume_analysis"
    resume_files: List[str] = field(default_factory=list)
    job_description_file: str = ""
    location: str = ""
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    error_message: Optional[str] = None
    total_resumes: int = 0
    processed_resumes: int = 0
    failed_resumes: int = 0
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    average_processing_time: float = 0.0
    total_resumes_processed: int = 0
    resumes_per_minute: float = 0.0
    queue_length: int = 0
    active_workers: int = 0

class BulkProcessingEngine:
    """
    High-performance bulk processing engine for resume analysis
    """
    
    def __init__(self, config: Dict[str, Any], resume_analyzer):
        """
        Initialize bulk processing engine
        
        Args:
            config: Configuration dictionary
            resume_analyzer: Resume analyzer instance
        """
        self.config = config
        self.resume_analyzer = resume_analyzer
        
        # Processing configuration
        self.max_workers = config.get('bulk_processing', {}).get('max_workers', 10)
        self.max_concurrent_jobs = config.get('bulk_processing', {}).get('max_concurrent_jobs', 5)
        self.batch_size = config.get('bulk_processing', {}).get('batch_size', 50)
        self.processing_timeout = config.get('bulk_processing', {}).get('timeout_seconds', 300)
        
        # Job management
        self.job_queue = PriorityQueue()
        self.active_jobs: Dict[str, BulkAnalysisJob] = {}
        self.completed_jobs: Dict[str, BulkAnalysisJob] = {}
        self.job_history: List[BulkAnalysisJob] = []
        
        # Threading and processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        self.worker_threads = []
        self.metrics = ProcessingMetrics()
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        logger.info(f"Bulk processing engine initialized with {self.max_workers} workers")
    
    def start(self):
        """Start the bulk processing engine"""
        if self.running:
            logger.warning("Bulk processing engine is already running")
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.max_concurrent_jobs):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"BulkProcessor-Worker-{i+1}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
            
        logger.info(f"Bulk processing engine started with {len(self.worker_threads)} workers")
    
    def stop(self):
        """Stop the bulk processing engine"""
        self.running = False
        
        # Cancel all pending jobs
        while not self.job_queue.empty():
            try:
                priority, job_id = self.job_queue.get_nowait()
                if job_id in self.active_jobs:
                    self.active_jobs[job_id].status = JobStatus.CANCELLED
            except:
                break
                
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Bulk processing engine stopped")
    
    def submit_bulk_job(self, resume_files: List[str], job_description_file: str,
                       location: str = "", priority: JobPriority = JobPriority.NORMAL,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a new bulk analysis job
        
        Args:
            resume_files: List of resume file paths
            job_description_file: Job description file path
            location: Location for processing
            priority: Job priority
            metadata: Additional metadata
            
        Returns:
            Job ID for tracking
        """
        job = BulkAnalysisJob(
            resume_files=resume_files,
            job_description_file=job_description_file,
            location=location,
            priority=priority,
            total_resumes=len(resume_files),
            metadata=metadata or {}
        )
        
        # Add to active jobs
        self.active_jobs[job.job_id] = job
        
        # Add to queue with priority
        self.job_queue.put((priority.value, job.job_id))
        
        # Update metrics
        self.metrics.total_jobs += 1
        self.metrics.queue_length = self.job_queue.qsize()
        
        logger.info(f"Submitted bulk job {job.job_id} with {len(resume_files)} resumes")
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[BulkAnalysisJob]:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    def get_all_jobs(self, status_filter: Optional[JobStatus] = None) -> List[BulkAnalysisJob]:
        """Get all jobs, optionally filtered by status"""
        all_jobs = list(self.active_jobs.values()) + list(self.completed_jobs.values())
        
        if status_filter:
            return [job for job in all_jobs if job.status == status_filter]
        
        return sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                logger.info(f"Cancelled job {job_id}")
                return True
        return False
    
    def get_processing_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        # Update real-time metrics
        self.metrics.queue_length = self.job_queue.qsize()
        self.metrics.active_workers = len([
            job for job in self.active_jobs.values() 
            if job.status == JobStatus.RUNNING
        ])
        
        return self.metrics
    
    def _worker_loop(self):
        """Main worker loop for processing jobs"""
        while self.running:
            try:
                # Get next job from queue (blocking with timeout)
                try:
                    priority, job_id = self.job_queue.get(timeout=1.0)
                except:
                    continue
                
                if job_id not in self.active_jobs:
                    continue
                    
                job = self.active_jobs[job_id]
                
                # Check if job was cancelled
                if job.status == JobStatus.CANCELLED:
                    self._move_to_completed(job_id)
                    continue
                
                # Process the job
                self._process_bulk_job(job)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def _process_bulk_job(self, job: BulkAnalysisJob):
        """Process a single bulk job"""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Estimate completion time
            avg_time_per_resume = 2.0  # seconds
            job.estimated_completion = datetime.now() + timedelta(
                seconds=job.total_resumes * avg_time_per_resume
            )
            
            logger.info(f"Started processing job {job.job_id} with {job.total_resumes} resumes")
            
            # Process resumes in batches
            batch_results = []
            
            for i in range(0, len(job.resume_files), self.batch_size):
                batch_files = job.resume_files[i:i + self.batch_size]
                
                # Process batch in parallel
                batch_futures = []
                for resume_file in batch_files:
                    future = self.executor.submit(
                        self._process_single_resume,
                        resume_file,
                        job.job_description_file,
                        job.job_id
                    )
                    batch_futures.append((resume_file, future))
                
                # Collect results
                for resume_file, future in batch_futures:
                    try:
                        result = future.result(timeout=self.processing_timeout)
                        batch_results.append(result)
                        job.processed_resumes += 1
                    except Exception as e:
                        logger.error(f"Failed to process {resume_file}: {e}")
                        batch_results.append({
                            'resume_file': resume_file,
                            'error': str(e),
                            'status': 'failed'
                        })
                        job.failed_resumes += 1
                    
                    # Update progress
                    job.progress = (job.processed_resumes + job.failed_resumes) / job.total_resumes * 100
                    
                    # Call progress callbacks
                    for callback in self.progress_callbacks:
                        try:
                            callback(job)
                        except Exception as e:
                            logger.error(f"Progress callback error: {e}")
                
                # Check if job was cancelled
                if job.status == JobStatus.CANCELLED:
                    break
            
            # Store results
            job.results = batch_results
            job.completed_at = datetime.now()
            
            if job.status != JobStatus.CANCELLED:
                job.status = JobStatus.COMPLETED
                logger.info(f"Completed job {job.job_id}: {job.processed_resumes} successful, {job.failed_resumes} failed")
            
            # Update metrics
            self.metrics.completed_jobs += 1
            self.metrics.total_resumes_processed += job.processed_resumes
            
            # Calculate processing time metrics
            if job.started_at and job.completed_at:
                processing_time = (job.completed_at - job.started_at).total_seconds()
                if processing_time > 0:
                    resumes_per_minute = (job.processed_resumes / processing_time) * 60
                    self.metrics.resumes_per_minute = (
                        (self.metrics.resumes_per_minute + resumes_per_minute) / 2
                        if self.metrics.resumes_per_minute > 0 else resumes_per_minute
                    )
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Completion callback error: {e}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self.metrics.failed_jobs += 1
            logger.error(f"Job {job.job_id} failed: {e}")
        
        finally:
            # Move job to completed
            self._move_to_completed(job.job_id)
    
    def _process_single_resume(self, resume_file: str, job_description_file: str, job_id: str) -> Dict[str, Any]:
        """Process a single resume"""
        try:
            # Use the analyzer to process the resume
            result = self.resume_analyzer.analyze_resume_for_job(
                resume_file, 
                job_description_file, 
                save_to_db=True
            )
            
            # Add processing metadata
            result['processing_metadata'] = {
                'job_id': job_id,
                'processed_at': datetime.now().isoformat(),
                'resume_file': resume_file
            }
            
            return result
            
        except Exception as e:
            return {
                'resume_file': resume_file,
                'error': str(e),
                'status': 'failed',
                'processing_metadata': {
                    'job_id': job_id,
                    'processed_at': datetime.now().isoformat(),
                    'resume_file': resume_file
                }
            }
    
    def _move_to_completed(self, job_id: str):
        """Move job from active to completed"""
        if job_id in self.active_jobs:
            job = self.active_jobs.pop(job_id)
            self.completed_jobs[job_id] = job
            self.job_history.append(job)
            
            # Keep only last 1000 completed jobs in memory
            if len(self.completed_jobs) > 1000:
                oldest_job_id = min(self.completed_jobs.keys(), 
                                  key=lambda x: self.completed_jobs[x].completed_at)
                del self.completed_jobs[oldest_job_id]
    
    def add_progress_callback(self, callback: Callable[[BulkAnalysisJob], None]):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[BulkAnalysisJob], None]):
        """Add a completion callback function"""
        self.completion_callbacks.append(callback)
    
    def get_job_results_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of job results"""
        job = self.get_job_status(job_id)
        if not job:
            return None
        
        summary = {
            'job_id': job_id,
            'status': job.status.value,
            'total_resumes': job.total_resumes,
            'processed_resumes': job.processed_resumes,
            'failed_resumes': job.failed_resumes,
            'progress_percentage': job.progress,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'estimated_completion': job.estimated_completion.isoformat() if job.estimated_completion else None,
            'processing_time_seconds': (
                (job.completed_at - job.started_at).total_seconds() 
                if job.started_at and job.completed_at else None
            ),
            'location': job.location,
            'priority': job.priority.value
        }
        
        if job.status == JobStatus.COMPLETED and job.results:
            # Calculate result statistics
            successful_results = [r for r in job.results if 'error' not in r]
            if successful_results:
                scores = [
                    r.get('analysis_results', {}).get('overall_score', 0) 
                    for r in successful_results
                ]
                summary['result_statistics'] = {
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'max_score': max(scores) if scores else 0,
                    'min_score': min(scores) if scores else 0,
                    'hire_recommendations': len([
                        r for r in successful_results 
                        if r.get('hiring_recommendation', {}).get('decision') == 'HIRE'
                    ]),
                    'interview_recommendations': len([
                        r for r in successful_results 
                        if r.get('hiring_recommendation', {}).get('decision') == 'INTERVIEW'
                    ]),
                    'reject_recommendations': len([
                        r for r in successful_results 
                        if r.get('hiring_recommendation', {}).get('decision') == 'REJECT'
                    ])
                }
        
        return summary
    
    def export_job_results(self, job_id: str, format: str = 'json') -> Optional[str]:
        """Export job results in specified format"""
        job = self.get_job_status(job_id)
        if not job or not job.results:
            return None
        
        if format == 'json':
            return json.dumps(job.results, indent=2, default=str)
        elif format == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if job.results:
                # Get all keys from first successful result
                successful_results = [r for r in job.results if 'error' not in r]
                if successful_results:
                    fieldnames = ['resume_file', 'overall_score', 'match_level', 'hiring_decision', 'confidence']
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in successful_results:
                        analysis = result.get('analysis_results', {})
                        hiring = result.get('hiring_recommendation', {})
                        writer.writerow({
                            'resume_file': result.get('resume_data', {}).get('filename', ''),
                            'overall_score': analysis.get('overall_score', 0),
                            'match_level': analysis.get('match_level', ''),
                            'hiring_decision': hiring.get('decision', ''),
                            'confidence': hiring.get('confidence', '')
                        })
            
            return output.getvalue()
        
        return None