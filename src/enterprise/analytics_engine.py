"""
Enterprise Analytics Engine
Advanced analytics dashboard for placement team metrics, historical trends, and cross-location insights
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics available"""
    VOLUME = "volume"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    TRENDS = "trends"
    COMPARISONS = "comparisons"

class TimeRange(Enum):
    """Time range options for analytics"""
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_6_MONTHS = "last_6_months"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"

@dataclass
class AnalyticsMetric:
    """Single analytics metric"""
    name: str
    value: float
    unit: str
    description: str
    trend: Optional[float] = None  # Percentage change
    benchmark: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LocationMetrics:
    """Metrics for a specific location"""
    location: str
    total_analyses: int = 0
    successful_analyses: int = 0
    average_score: float = 0.0
    hire_rate: float = 0.0
    interview_rate: float = 0.0
    processing_time_avg: float = 0.0
    top_skills: List[str] = field(default_factory=list)
    industry_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_volume: Dict[str, int] = field(default_factory=dict)

@dataclass
class PlacementTeamMetrics:
    """Metrics for placement team performance"""
    team_member: str
    location: str
    analyses_conducted: int = 0
    average_score_given: float = 0.0
    accuracy_rate: float = 0.0  # Compared to peer consensus
    processing_speed: float = 0.0  # Analyses per hour
    specialization_areas: List[str] = field(default_factory=list)
    feedback_quality_score: float = 0.0

class AnalyticsEngine:
    """
    Enterprise analytics engine for resume analysis system
    """
    
    def __init__(self, database_manager, location_manager, bulk_processor):
        """Initialize analytics engine"""
        self.db = database_manager
        self.location_manager = location_manager
        self.bulk_processor = bulk_processor
        
        # Cache for frequently accessed metrics
        self.metrics_cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        logger.info("Analytics engine initialized")
    
    def get_system_overview(self, time_range: TimeRange = TimeRange.LAST_30_DAYS) -> Dict[str, Any]:
        """Get high-level system overview metrics"""
        cache_key = f"system_overview_{time_range.value}"
        
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        start_date, end_date = self._get_date_range(time_range)
        
        # Get processing metrics from bulk processor
        processing_metrics = self.bulk_processor.get_processing_metrics()
        
        # Calculate system-wide metrics
        overview = {
            "summary": {
                "total_analyses": processing_metrics.total_jobs,
                "successful_analyses": processing_metrics.completed_jobs,
                "failed_analyses": processing_metrics.failed_jobs,
                "success_rate": (processing_metrics.completed_jobs / max(processing_metrics.total_jobs, 1)) * 100,
                "average_processing_time": processing_metrics.average_processing_time,
                "current_queue_length": processing_metrics.queue_length,
                "active_workers": processing_metrics.active_workers
            },
            "performance": {
                "resumes_per_minute": processing_metrics.resumes_per_minute,
                "total_resumes_processed": processing_metrics.total_resumes_processed,
                "system_utilization": (processing_metrics.active_workers / 10) * 100,  # Assuming 10 max workers
                "throughput_trend": self._calculate_throughput_trend()
            },
            "locations": self._get_location_summary(),
            "trends": self._calculate_system_trends(start_date, end_date),
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache the results
        self.metrics_cache[cache_key] = overview
        return overview
    
    def get_location_analytics(self, location: str, 
                              time_range: TimeRange = TimeRange.LAST_30_DAYS) -> LocationMetrics:
        """Get detailed analytics for a specific location"""
        cache_key = f"location_{location}_{time_range.value}"
        
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        start_date, end_date = self._get_date_range(time_range)
        
        # Get jobs for this location
        all_jobs = self.bulk_processor.get_all_jobs()
        location_jobs = [
            job for job in all_jobs 
            if job.location.lower() == location.lower() and
            start_date <= job.created_at <= end_date
        ]
        
        # Calculate location metrics
        metrics = LocationMetrics(location=location)
        
        if location_jobs:
            metrics.total_analyses = len(location_jobs)
            metrics.successful_analyses = len([j for j in location_jobs if j.status.value == "completed"])
            
            # Calculate average score from completed jobs
            completed_jobs = [j for j in location_jobs if j.status.value == "completed" and j.results]
            if completed_jobs:
                all_scores = []
                hire_count = 0
                interview_count = 0
                
                for job in completed_jobs:
                    for result in job.results:
                        if 'error' not in result:
                            score = result.get('analysis_results', {}).get('overall_score', 0)
                            all_scores.append(score)
                            
                            decision = result.get('hiring_recommendation', {}).get('decision', '')
                            if decision == 'HIRE':
                                hire_count += 1
                            elif decision in ['INTERVIEW', 'MAYBE']:
                                interview_count += 1
                
                if all_scores:
                    metrics.average_score = sum(all_scores) / len(all_scores)
                    metrics.hire_rate = (hire_count / len(all_scores)) * 100
                    metrics.interview_rate = (interview_count / len(all_scores)) * 100
                
                # Calculate processing time
                processing_times = []
                for job in completed_jobs:
                    if job.started_at and job.completed_at:
                        processing_time = (job.completed_at - job.started_at).total_seconds()
                        processing_times.append(processing_time)
                
                if processing_times:
                    metrics.processing_time_avg = sum(processing_times) / len(processing_times)
            
            # Extract top skills and industry distribution
            metrics.top_skills = self._extract_top_skills_for_location(location_jobs)
            metrics.industry_distribution = self._extract_industry_distribution(location_jobs)
            metrics.monthly_volume = self._calculate_monthly_volume(location_jobs)
        
        # Cache the results
        self.metrics_cache[cache_key] = metrics
        return metrics
    
    def get_placement_team_analytics(self, time_range: TimeRange = TimeRange.LAST_30_DAYS) -> List[PlacementTeamMetrics]:
        """Get analytics for placement team performance"""
        # In a full implementation, this would query user activity logs
        # For now, return mock data based on locations
        
        team_metrics = []
        for location_enum, config in self.location_manager.locations.items():
            # Mock team member data for each location
            team_member = PlacementTeamMetrics(
                team_member=f"Team Lead - {config.display_name}",
                location=location_enum.value,
                analyses_conducted=50,  # Mock data
                average_score_given=72.5,
                accuracy_rate=88.5,
                processing_speed=12.0,  # Analyses per hour
                specialization_areas=config.preferred_skills[:3],
                feedback_quality_score=85.0
            )
            team_metrics.append(team_member)
        
        return team_metrics
    
    def get_comparative_analytics(self) -> Dict[str, Any]:
        """Get comparative analytics across locations"""
        all_locations = [loc.value for loc in self.location_manager.locations.keys()]
        location_comparisons = {}
        
        for location in all_locations:
            metrics = self.get_location_analytics(location)
            location_comparisons[location] = {
                "total_analyses": metrics.total_analyses,
                "success_rate": (metrics.successful_analyses / max(metrics.total_analyses, 1)) * 100,
                "average_score": metrics.average_score,
                "hire_rate": metrics.hire_rate,
                "processing_efficiency": 1 / max(metrics.processing_time_avg, 1) * 100
            }
        
        # Calculate rankings
        rankings = self._calculate_location_rankings(location_comparisons)
        
        return {
            "location_comparisons": location_comparisons,
            "rankings": rankings,
            "top_performer": self._identify_top_performer(location_comparisons),
            "improvement_opportunities": self._identify_improvement_opportunities(location_comparisons)
        }
    
    def get_trend_analysis(self, metric: str, 
                          time_range: TimeRange = TimeRange.LAST_90_DAYS) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        start_date, end_date = self._get_date_range(time_range)
        
        # Generate time series data
        time_series = self._generate_time_series(metric, start_date, end_date)
        
        # Calculate trend statistics
        if len(time_series) > 1:
            values = [point['value'] for point in time_series]
            trend_slope = self._calculate_trend_slope(values)
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            
            # Seasonal analysis
            seasonal_patterns = self._detect_seasonal_patterns(time_series)
            
            # Forecast next period
            forecast = self._simple_forecast(values, periods=7)
        else:
            trend_slope = 0
            trend_direction = "stable"
            seasonal_patterns = {}
            forecast = []
        
        return {
            "metric": metric,
            "time_range": time_range.value,
            "time_series": time_series,
            "trend_analysis": {
                "slope": trend_slope,
                "direction": trend_direction,
                "strength": abs(trend_slope)
            },
            "seasonal_patterns": seasonal_patterns,
            "forecast": forecast,
            "statistics": {
                "mean": statistics.mean([p['value'] for p in time_series]) if time_series else 0,
                "median": statistics.median([p['value'] for p in time_series]) if time_series else 0,
                "std_dev": statistics.stdev([p['value'] for p in time_series]) if len(time_series) > 1 else 0
            }
        }
    
    def get_skills_analytics(self, location: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics on skill trends and demands"""
        all_jobs = self.bulk_processor.get_all_jobs()
        
        if location:
            jobs = [job for job in all_jobs if job.location.lower() == location.lower()]
        else:
            jobs = all_jobs
        
        # Extract skills from job results
        all_skills = Counter()
        successful_placements = Counter()
        
        for job in jobs:
            if job.results:
                for result in job.results:
                    if 'error' not in result:
                        # Extract skills from resume data
                        skills = result.get('resume_data', {}).get('skills', [])
                        for skill in skills:
                            all_skills[skill.lower()] += 1
                            
                            # Count successful placements
                            decision = result.get('hiring_recommendation', {}).get('decision', '')
                            if decision in ['HIRE', 'INTERVIEW']:
                                successful_placements[skill.lower()] += 1
        
        # Calculate skill success rates
        skill_success_rates = {}
        for skill, count in all_skills.items():
            success_count = successful_placements.get(skill, 0)
            skill_success_rates[skill] = (success_count / count) * 100 if count > 0 else 0
        
        # Top skills by frequency
        top_skills_by_frequency = dict(all_skills.most_common(20))
        
        # Top skills by success rate (minimum 5 occurrences)
        top_skills_by_success = {
            skill: rate for skill, rate in 
            sorted(skill_success_rates.items(), key=lambda x: x[1], reverse=True)
            if all_skills[skill] >= 5
        }
        
        return {
            "location": location or "all_locations",
            "total_unique_skills": len(all_skills),
            "total_skill_instances": sum(all_skills.values()),
            "top_skills_by_frequency": top_skills_by_frequency,
            "top_skills_by_success_rate": dict(list(top_skills_by_success.items())[:20]),
            "emerging_skills": self._identify_emerging_skills(all_skills),
            "skill_gap_analysis": self._analyze_skill_gaps(jobs)
        }
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance dashboard data"""
        current_metrics = self.bulk_processor.get_processing_metrics()
        
        # System health indicators
        health_score = self._calculate_system_health_score(current_metrics)
        
        # Resource utilization
        resource_usage = {
            "queue_utilization": min(100, (current_metrics.queue_length / 100) * 100),
            "worker_utilization": (current_metrics.active_workers / 10) * 100,
            "processing_efficiency": min(100, current_metrics.resumes_per_minute * 2)
        }
        
        # Recent activity summary
        recent_jobs = self.bulk_processor.get_all_jobs()[-10:]  # Last 10 jobs
        recent_activity = [
            {
                "job_id": job.job_id[:8],
                "status": job.status.value,
                "location": job.location,
                "resumes": job.total_resumes,
                "progress": job.progress,
                "created_at": job.created_at.isoformat()
            }
            for job in recent_jobs
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "score": health_score,
                "status": "healthy" if health_score > 80 else "warning" if health_score > 60 else "critical"
            },
            "current_metrics": {
                "active_jobs": len([j for j in recent_jobs if j.status.value == "running"]),
                "queue_length": current_metrics.queue_length,
                "processing_rate": current_metrics.resumes_per_minute,
                "success_rate": (current_metrics.completed_jobs / max(current_metrics.total_jobs, 1)) * 100
            },
            "resource_utilization": resource_usage,
            "recent_activity": recent_activity,
            "alerts": self._generate_system_alerts(current_metrics)
        }
    
    def export_analytics_report(self, format: str = "json", 
                               time_range: TimeRange = TimeRange.LAST_30_DAYS) -> str:
        """Export comprehensive analytics report"""
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_range": time_range.value,
                "format": format
            },
            "system_overview": self.get_system_overview(time_range),
            "location_analytics": {
                loc.value: self.get_location_analytics(loc.value, time_range)
                for loc in self.location_manager.locations.keys()
            },
            "comparative_analytics": self.get_comparative_analytics(),
            "skills_analytics": self.get_skills_analytics(),
            "placement_team_metrics": self.get_placement_team_analytics(time_range)
        }
        
        if format == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format == "csv":
            # Convert to CSV format (simplified)
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write summary data
            writer.writerow(["Metric", "Value"])
            summary = report_data["system_overview"]["summary"]
            for key, value in summary.items():
                writer.writerow([key, value])
            
            return output.getvalue()
        
        return json.dumps(report_data, indent=2, default=str)
    
    # Helper methods
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.metrics_cache:
            return False
        
        # For simplicity, assuming cache entries have a timestamp
        # In production, implement proper cache TTL
        return True  # Simplified for demo
    
    def _get_date_range(self, time_range: TimeRange) -> Tuple[datetime, datetime]:
        """Get start and end dates for time range"""
        end_date = datetime.now()
        
        if time_range == TimeRange.LAST_7_DAYS:
            start_date = end_date - timedelta(days=7)
        elif time_range == TimeRange.LAST_30_DAYS:
            start_date = end_date - timedelta(days=30)
        elif time_range == TimeRange.LAST_90_DAYS:
            start_date = end_date - timedelta(days=90)
        elif time_range == TimeRange.LAST_6_MONTHS:
            start_date = end_date - timedelta(days=180)
        elif time_range == TimeRange.LAST_YEAR:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)  # Default
        
        return start_date, end_date
    
    def _get_location_summary(self) -> Dict[str, Any]:
        """Get summary metrics for all locations"""
        summary = {}
        for location_enum in self.location_manager.locations.keys():
            location = location_enum.value
            metrics = self.get_location_analytics(location)
            summary[location] = {
                "total_analyses": metrics.total_analyses,
                "average_score": metrics.average_score,
                "hire_rate": metrics.hire_rate
            }
        return summary
    
    def _calculate_throughput_trend(self) -> float:
        """Calculate throughput trend (simplified)"""
        # In production, this would analyze historical data
        return 5.2  # Mock 5.2% increase
    
    def _calculate_system_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate various system trends"""
        return {
            "volume_trend": 12.5,  # Mock data
            "quality_trend": 3.2,
            "efficiency_trend": 8.1
        }
    
    def _extract_top_skills_for_location(self, jobs: List) -> List[str]:
        """Extract top skills for location from job results"""
        skills_counter = Counter()
        
        for job in jobs:
            if job.results:
                for result in job.results:
                    if 'error' not in result:
                        skills = result.get('resume_data', {}).get('skills', [])
                        skills_counter.update(skills)
        
        return [skill for skill, _ in skills_counter.most_common(10)]
    
    def _extract_industry_distribution(self, jobs: List) -> Dict[str, int]:
        """Extract industry distribution from jobs"""
        # Mock industry distribution based on location
        return {
            "Technology": 45,
            "Healthcare": 20,
            "Finance": 15,
            "Education": 10,
            "Other": 10
        }
    
    def _calculate_monthly_volume(self, jobs: List) -> Dict[str, int]:
        """Calculate monthly volume of analyses"""
        monthly_volume = defaultdict(int)
        
        for job in jobs:
            month_key = job.created_at.strftime("%Y-%m")
            monthly_volume[month_key] += job.total_resumes
        
        return dict(monthly_volume)
    
    def _calculate_location_rankings(self, comparisons: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Calculate rankings for different metrics"""
        rankings = {}
        
        metrics = ["total_analyses", "success_rate", "average_score", "hire_rate", "processing_efficiency"]
        
        for metric in metrics:
            sorted_locations = sorted(
                comparisons.keys(),
                key=lambda x: comparisons[x].get(metric, 0),
                reverse=True
            )
            rankings[metric] = sorted_locations
        
        return rankings
    
    def _identify_top_performer(self, comparisons: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Identify top performing location"""
        # Simple scoring: average of normalized metrics
        scores = {}
        
        for location in comparisons.keys():
            metrics = comparisons[location]
            # Normalize and average (simplified)
            score = (
                metrics.get("success_rate", 0) * 0.3 +
                metrics.get("average_score", 0) * 0.3 +
                metrics.get("hire_rate", 0) * 0.2 +
                metrics.get("processing_efficiency", 0) * 0.2
            )
            scores[location] = score
        
        top_location = max(scores.keys(), key=lambda x: scores[x])
        
        return {
            "location": top_location,
            "score": scores[top_location],
            "strengths": ["High success rate", "Excellent processing efficiency"]
        }
    
    def _identify_improvement_opportunities(self, comparisons: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        opportunities = []
        
        for location, metrics in comparisons.items():
            if metrics.get("success_rate", 0) < 80:
                opportunities.append({
                    "location": location,
                    "area": "success_rate",
                    "current_value": metrics.get("success_rate", 0),
                    "recommendation": "Improve quality control processes"
                })
            
            if metrics.get("processing_efficiency", 0) < 50:
                opportunities.append({
                    "location": location,
                    "area": "processing_efficiency",
                    "current_value": metrics.get("processing_efficiency", 0),
                    "recommendation": "Optimize processing workflows"
                })
        
        return opportunities
    
    def _generate_time_series(self, metric: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Generate time series data for metric"""
        # Mock time series data
        import random
        
        time_series = []
        current_date = start_date
        base_value = 75  # Base metric value
        
        while current_date <= end_date:
            # Add some random variation
            value = base_value + random.uniform(-10, 10)
            time_series.append({
                "date": current_date.isoformat(),
                "value": value
            })
            current_date += timedelta(days=1)
        
        return time_series
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        y = values
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _detect_seasonal_patterns(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect seasonal patterns in time series"""
        # Simplified seasonal analysis
        return {
            "weekly_pattern": "Higher activity on weekdays",
            "monthly_pattern": "Peak activity mid-month",
            "seasonal_strength": 0.3
        }
    
    def _simple_forecast(self, values: List[float], periods: int = 7) -> List[Dict[str, Any]]:
        """Simple forecasting using moving average"""
        if len(values) < 3:
            return []
        
        # Use last 3 values for simple moving average
        recent_avg = sum(values[-3:]) / 3
        forecast = []
        
        for i in range(periods):
            future_date = datetime.now() + timedelta(days=i+1)
            forecast.append({
                "date": future_date.isoformat(),
                "predicted_value": recent_avg,
                "confidence_interval": [recent_avg - 5, recent_avg + 5]
            })
        
        return forecast
    
    def _identify_emerging_skills(self, skills_counter: Counter) -> List[str]:
        """Identify emerging skills based on recent trends"""
        # Mock emerging skills analysis
        return ["kubernetes", "machine learning", "cloud computing", "devops", "react native"]
    
    def _analyze_skill_gaps(self, jobs: List) -> Dict[str, Any]:
        """Analyze skill gaps between job requirements and candidate skills"""
        # Mock skill gap analysis
        return {
            "most_requested_missing_skills": ["python", "aws", "kubernetes"],
            "average_skill_gap_percentage": 35.2,
            "locations_with_highest_gaps": ["pune", "delhi_ncr"]
        }
    
    def _calculate_system_health_score(self, metrics) -> float:
        """Calculate overall system health score"""
        # Factors: success rate, queue length, processing speed
        success_rate = (metrics.completed_jobs / max(metrics.total_jobs, 1)) * 100
        queue_health = max(0, 100 - (metrics.queue_length / 50) * 100)  # Lower queue = better
        processing_health = min(100, metrics.resumes_per_minute * 5)  # Higher rate = better
        
        return (success_rate * 0.4 + queue_health * 0.3 + processing_health * 0.3)
    
    def _generate_system_alerts(self, metrics) -> List[Dict[str, Any]]:
        """Generate system alerts based on current metrics"""
        alerts = []
        
        if metrics.queue_length > 50:
            alerts.append({
                "level": "warning",
                "message": "High queue length detected",
                "metric": "queue_length",
                "value": metrics.queue_length
            })
        
        if metrics.failed_jobs / max(metrics.total_jobs, 1) > 0.1:
            alerts.append({
                "level": "critical",
                "message": "High failure rate detected",
                "metric": "failure_rate",
                "value": (metrics.failed_jobs / max(metrics.total_jobs, 1)) * 100
            })
        
        return alerts