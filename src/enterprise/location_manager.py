"""
Location Manager
Handles multi-location operations for Innomatics Research Labs offices
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class InnomaticsLocation(Enum):
    """Enum for Innomatics Research Labs office locations"""
    HYDERABAD = "hyderabad"
    BANGALORE = "bangalore" 
    PUNE = "pune"
    DELHI_NCR = "delhi_ncr"

@dataclass
class LocationConfig:
    """Configuration for each location"""
    name: str
    display_name: str
    time_zone: str
    region: str
    contact_email: str
    max_concurrent_analyses: int
    preferred_skills: List[str]
    industry_focus: List[str]

class LocationManager:
    """
    Manages multi-location operations and location-specific processing
    """
    
    def __init__(self):
        """Initialize location manager with office configurations"""
        self.locations = self._initialize_locations()
        self.current_location = None
        
    def _initialize_locations(self) -> Dict[InnomaticsLocation, LocationConfig]:
        """Initialize all Innomatics office locations with their configurations"""
        return {
            InnomaticsLocation.HYDERABAD: LocationConfig(
                name="hyderabad",
                display_name="Hyderabad Office",
                time_zone="Asia/Kolkata",
                region="South India",
                contact_email="placement.hyderabad@innomatics.in",
                max_concurrent_analyses=50,
                preferred_skills=["python", "machine learning", "data science", "ai", "deep learning"],
                industry_focus=["technology", "fintech", "healthcare", "e-commerce"]
            ),
            InnomaticsLocation.BANGALORE: LocationConfig(
                name="bangalore",
                display_name="Bangalore Office", 
                time_zone="Asia/Kolkata",
                region="South India",
                contact_email="placement.bangalore@innomatics.in",
                max_concurrent_analyses=75,
                preferred_skills=["java", "spring", "microservices", "kubernetes", "aws", "react"],
                industry_focus=["technology", "startup", "product", "enterprise"]
            ),
            InnomaticsLocation.PUNE: LocationConfig(
                name="pune",
                display_name="Pune Office",
                time_zone="Asia/Kolkata", 
                region="West India",
                contact_email="placement.pune@innomatics.in",
                max_concurrent_analyses=40,
                preferred_skills=["java", "angular", "devops", "cloud", "automation"],
                industry_focus=["manufacturing", "automotive", "technology", "consulting"]
            ),
            InnomaticsLocation.DELHI_NCR: LocationConfig(
                name="delhi_ncr",
                display_name="Delhi NCR Office",
                time_zone="Asia/Kolkata",
                region="North India", 
                contact_email="placement.delhi@innomatics.in",
                max_concurrent_analyses=60,
                preferred_skills=["python", "django", "node.js", "government tech", "fintech"],
                industry_focus=["government", "fintech", "consulting", "enterprise"]
            )
        }
    
    def get_location_by_name(self, location_name: str) -> Optional[InnomaticsLocation]:
        """Get location enum by string name"""
        for location in InnomaticsLocation:
            if location.value.lower() == location_name.lower():
                return location
        return None
    
    def get_location_config(self, location: InnomaticsLocation) -> LocationConfig:
        """Get configuration for a specific location"""
        return self.locations.get(location)
    
    def get_all_locations(self) -> List[Dict[str, Any]]:
        """Get all locations with their configurations"""
        return [
            {
                "id": loc.value,
                "name": config.name,
                "display_name": config.display_name,
                "region": config.region,
                "max_concurrent": config.max_concurrent_analyses,
                "preferred_skills": config.preferred_skills,
                "industry_focus": config.industry_focus
            }
            for loc, config in self.locations.items()
        ]
    
    def set_current_location(self, location: InnomaticsLocation):
        """Set the current active location for processing"""
        self.current_location = location
        logger.info(f"Set current location to {location.value}")
    
    def get_location_specific_scoring_weights(self, location: InnomaticsLocation) -> Dict[str, float]:
        """Get location-specific scoring weights based on regional preferences"""
        config = self.get_location_config(location)
        
        # Base weights
        weights = {
            'technical_skills': 0.30,
            'experience': 0.25,
            'education': 0.15,
            'location_preference': 0.10,
            'industry_alignment': 0.10,
            'soft_skills': 0.10
        }
        
        # Adjust weights based on location focus
        if location == InnomaticsLocation.BANGALORE:
            # Bangalore focuses more on technical skills for startups
            weights['technical_skills'] = 0.35
            weights['industry_alignment'] = 0.15
        elif location == InnomaticsLocation.HYDERABAD:
            # Hyderabad emphasizes AI/ML skills
            weights['technical_skills'] = 0.35
            weights['education'] = 0.20
        elif location == InnomaticsLocation.PUNE:
            # Pune values experience more for manufacturing/automotive
            weights['experience'] = 0.30
            weights['industry_alignment'] = 0.15
        elif location == InnomaticsLocation.DELHI_NCR:
            # Delhi focuses on government/enterprise experience
            weights['experience'] = 0.30
            weights['soft_skills'] = 0.15
            
        return weights
    
    def filter_jobs_by_location(self, jobs: List[Dict[str, Any]], 
                               location: Optional[InnomaticsLocation] = None) -> List[Dict[str, Any]]:
        """Filter job opportunities by location"""
        if not location:
            location = self.current_location
            
        if not location:
            return jobs
            
        config = self.get_location_config(location)
        filtered_jobs = []
        
        for job in jobs:
            # Check if job matches location criteria
            job_location = job.get('location', '').lower()
            job_industry = job.get('industry', '').lower()
            job_skills = [skill.lower() for skill in job.get('required_skills', [])]
            
            # Location match
            location_match = (
                location.value in job_location or
                config.region.lower() in job_location or
                config.name in job_location
            )
            
            # Industry focus match
            industry_match = any(
                focus.lower() in job_industry 
                for focus in config.industry_focus
            )
            
            # Preferred skills match
            skill_match = any(
                pref_skill.lower() in job_skills 
                for pref_skill in config.preferred_skills
            )
            
            # Include job if it matches any criteria
            if location_match or industry_match or skill_match:
                job['location_relevance_score'] = self._calculate_location_relevance(
                    job, config
                )
                filtered_jobs.append(job)
        
        # Sort by location relevance
        return sorted(filtered_jobs, 
                     key=lambda x: x.get('location_relevance_score', 0), 
                     reverse=True)
    
    def _calculate_location_relevance(self, job: Dict[str, Any], 
                                    config: LocationConfig) -> float:
        """Calculate how relevant a job is to a specific location"""
        score = 0.0
        
        job_location = job.get('location', '').lower()
        job_industry = job.get('industry', '').lower() 
        job_skills = [skill.lower() for skill in job.get('required_skills', [])]
        
        # Location proximity score (40%)
        if config.name in job_location:
            score += 40.0
        elif config.region.lower() in job_location:
            score += 25.0
        elif 'india' in job_location:
            score += 10.0
            
        # Industry alignment score (35%)
        industry_matches = sum(
            1 for focus in config.industry_focus 
            if focus.lower() in job_industry
        )
        score += (industry_matches / len(config.industry_focus)) * 35.0
        
        # Skills alignment score (25%)
        skill_matches = sum(
            1 for pref_skill in config.preferred_skills
            if pref_skill.lower() in job_skills
        )
        if config.preferred_skills:
            score += (skill_matches / len(config.preferred_skills)) * 25.0
            
        return min(score, 100.0)
    
    def get_location_statistics(self, location: InnomaticsLocation) -> Dict[str, Any]:
        """Get statistics and metrics for a specific location"""
        config = self.get_location_config(location)
        
        return {
            'location': location.value,
            'display_name': config.display_name,
            'region': config.region,
            'max_concurrent_analyses': config.max_concurrent_analyses,
            'preferred_skills_count': len(config.preferred_skills),
            'industry_focus_count': len(config.industry_focus),
            'contact_email': config.contact_email,
            'time_zone': config.time_zone
        }
    
    def optimize_analysis_distribution(self, total_analyses: int) -> Dict[InnomaticsLocation, int]:
        """Distribute analysis load across locations based on capacity"""
        total_capacity = sum(
            config.max_concurrent_analyses 
            for config in self.locations.values()
        )
        
        distribution = {}
        for location, config in self.locations.items():
            # Distribute based on capacity percentage
            capacity_ratio = config.max_concurrent_analyses / total_capacity
            allocated = int(total_analyses * capacity_ratio)
            distribution[location] = allocated
            
        # Handle remainder
        remainder = total_analyses - sum(distribution.values())
        if remainder > 0:
            # Give remainder to location with highest capacity
            max_capacity_location = max(
                self.locations.keys(),
                key=lambda x: self.locations[x].max_concurrent_analyses
            )
            distribution[max_capacity_location] += remainder
            
        return distribution
    
    def validate_location_access(self, user_location: str, 
                                target_location: InnomaticsLocation) -> bool:
        """Validate if user from one location can access another location's data"""
        user_loc = self.get_location_by_name(user_location)
        
        if not user_loc:
            return False
            
        # Same location access is always allowed
        if user_loc == target_location:
            return True
            
        # Regional access within India is allowed
        user_config = self.get_location_config(user_loc)
        target_config = self.get_location_config(target_location)
        
        return user_config.region == target_config.region