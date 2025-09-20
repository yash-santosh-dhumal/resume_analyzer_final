"""
Enterprise Resume Analyzer Module
Complete enterprise-grade features for multi-location operations
"""

from .location_manager import InnomaticsLocation, LocationManager
from .bulk_processor import BulkProcessor, JobStatus, BulkJob
from .user_manager import UserRole, Permission, UserManager, User
from .llm_pipeline import LLMPipeline, AnalysisStage, PipelineState
from .analytics_engine import AnalyticsEngine, MetricType, ReportType
from .notification_service import NotificationService, NotificationType, NotificationChannel
from .production_manager import ProductionManager, PerformanceMonitor, LoadBalancer, PerformanceOptimizer
from .config_manager import get_config_manager, get_config, update_config, EnterpriseConfig

__version__ = "1.0.0"

__all__ = [
    # Location Management
    "InnomaticsLocation",
    "LocationManager",
    
    # Bulk Processing
    "BulkProcessor", 
    "JobStatus",
    "BulkJob",
    
    # User Management & Security
    "UserRole",
    "Permission", 
    "UserManager",
    "User",
    
    # Advanced LLM Pipeline
    "LLMPipeline",
    "AnalysisStage", 
    "PipelineState",
    
    # Analytics & Reporting
    "AnalyticsEngine",
    "MetricType",
    "ReportType",
    
    # Notifications
    "NotificationService",
    "NotificationType", 
    "NotificationChannel",
    
    # Production Management
    "ProductionManager",
    "PerformanceMonitor",
    "LoadBalancer", 
    "PerformanceOptimizer",
    
    # Configuration
    "get_config_manager",
    "get_config",
    "update_config", 
    "EnterpriseConfig"
]

# Enterprise feature availability
ENTERPRISE_FEATURES = {
    "multi_location": True,
    "bulk_processing": True,
    "rbac_security": True, 
    "advanced_llm": True,
    "analytics_dashboard": True,
    "notification_system": True,
    "production_monitoring": True,
    "load_balancing": True,
    "performance_optimization": True
}

def get_enterprise_status():
    """Get enterprise features status"""
    return {
        "version": __version__,
        "features": ENTERPRISE_FEATURES,
        "total_features": len(ENTERPRISE_FEATURES),
        "enabled_features": sum(ENTERPRISE_FEATURES.values()),
        "completion_percentage": (sum(ENTERPRISE_FEATURES.values()) / len(ENTERPRISE_FEATURES)) * 100
    }