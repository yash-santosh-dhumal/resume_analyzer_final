"""
Enterprise Configuration Management
Centralized configuration for all enterprise features
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "resume_analyzer"
    username: str = "admin"
    password: str = "secure_password"
    pool_size: int = 20
    max_overflow: int = 10

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50

@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = "notifications@innomatics.in"
    password: str = "email_password"
    use_tls: bool = True
    sender_name: str = "Innomatics Resume Analyzer"

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = "your-super-secret-jwt-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    interval_seconds: int = 30
    metrics_retention_hours: int = 168  # 1 week
    alert_email_recipients: list = None
    health_check_timeout: int = 10

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    strategy: str = "weighted"  # round_robin, least_connections, weighted
    health_check_interval: int = 60
    health_check_timeout: int = 5
    max_retries: int = 3

@dataclass
class BulkProcessingConfig:
    """Bulk processing configuration"""
    max_workers: int = 8
    batch_size: int = 50
    queue_size: int = 1000
    timeout_seconds: int = 300
    retry_attempts: int = 3

@dataclass
class LLMConfig:
    """LLM configuration"""
    model_name: str = "gpt-4"
    api_key: str = "your-openai-api-key"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60

@dataclass
class EnterpriseConfig:
    """Main enterprise configuration"""
    # Environment
    environment: str = "production"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Components
    database: DatabaseConfig = None
    redis: RedisConfig = None
    email: EmailConfig = None
    security: SecurityConfig = None
    monitoring: MonitoringConfig = None
    load_balancer: LoadBalancerConfig = None
    bulk_processing: BulkProcessingConfig = None
    llm: LLMConfig = None
    
    # Locations
    default_location: str = "hyderabad"
    enabled_locations: list = None
    
    # Features
    enable_notifications: bool = True
    enable_analytics: bool = True
    enable_bulk_processing: bool = True
    enable_monitoring: bool = True
    enable_load_balancing: bool = True
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.redis is None:
            self.redis = RedisConfig()
        if self.email is None:
            self.email = EmailConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.load_balancer is None:
            self.load_balancer = LoadBalancerConfig()
        if self.bulk_processing is None:
            self.bulk_processing = BulkProcessingConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.enabled_locations is None:
            self.enabled_locations = ["hyderabad", "bangalore", "pune", "delhi_ncr"]
        if self.monitoring.alert_email_recipients is None:
            self.monitoring.alert_email_recipients = ["admin@innomatics.in", "devops@innomatics.in"]

class ConfigManager:
    """
    Configuration management system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or self._get_default_config_path()
        self.config = None
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Check environment variable first
        config_path = os.getenv("RESUME_ANALYZER_CONFIG")
        if config_path:
            return config_path
        
        # Check current directory
        for filename in ["config.yaml", "config.yml", "config.json"]:
            if os.path.exists(filename):
                return filename
        
        # Default to config directory
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "enterprise.yaml")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.json'):
                        config_data = json.load(f)
                    else:
                        config_data = yaml.safe_load(f)
                
                self.config = self._dict_to_config(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Create default configuration
                self.config = EnterpriseConfig()
                self.save_config()
                logger.info(f"Default configuration created at {self.config_path}")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = EnterpriseConfig()
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> EnterpriseConfig:
        """Convert dictionary to configuration object"""
        # Extract component configurations
        components = {}
        
        if "database" in config_data:
            components["database"] = DatabaseConfig(**config_data["database"])
        
        if "redis" in config_data:
            components["redis"] = RedisConfig(**config_data["redis"])
        
        if "email" in config_data:
            components["email"] = EmailConfig(**config_data["email"])
        
        if "security" in config_data:
            components["security"] = SecurityConfig(**config_data["security"])
        
        if "monitoring" in config_data:
            components["monitoring"] = MonitoringConfig(**config_data["monitoring"])
        
        if "load_balancer" in config_data:
            components["load_balancer"] = LoadBalancerConfig(**config_data["load_balancer"])
        
        if "bulk_processing" in config_data:
            components["bulk_processing"] = BulkProcessingConfig(**config_data["bulk_processing"])
        
        if "llm" in config_data:
            components["llm"] = LLMConfig(**config_data["llm"])
        
        # Create main config
        main_config = {k: v for k, v in config_data.items() 
                      if k not in ["database", "redis", "email", "security", "monitoring", 
                                  "load_balancer", "bulk_processing", "llm"]}
        main_config.update(components)
        
        return EnterpriseConfig(**main_config)
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_config(self) -> EnterpriseConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            # Convert current config to dict
            config_dict = asdict(self.config)
            
            # Apply updates recursively
            self._deep_update(config_dict, updates)
            
            # Convert back to config object
            self.config = self._dict_to_config(config_dict)
            
            # Save updated configuration
            self.save_config()
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate database configuration
        if not self.config.database.host:
            validation_results["errors"].append("Database host is required")
            validation_results["valid"] = False
        
        # Validate security configuration
        if len(self.config.security.jwt_secret_key) < 32:
            validation_results["warnings"].append("JWT secret key should be at least 32 characters")
        
        if self.config.security.jwt_secret_key == "your-super-secret-jwt-key-change-in-production":
            validation_results["errors"].append("JWT secret key must be changed from default value")
            validation_results["valid"] = False
        
        # Validate LLM configuration
        if not self.config.llm.api_key or self.config.llm.api_key == "your-openai-api-key":
            validation_results["errors"].append("LLM API key must be configured")
            validation_results["valid"] = False
        
        # Validate email configuration
        if self.config.enable_notifications and not self.config.email.password:
            validation_results["warnings"].append("Email password not configured - notifications may not work")
        
        # Validate locations
        if not self.config.enabled_locations:
            validation_results["errors"].append("At least one location must be enabled")
            validation_results["valid"] = False
        
        if self.config.default_location not in self.config.enabled_locations:
            validation_results["errors"].append("Default location must be in enabled locations")
            validation_results["valid"] = False
        
        return validation_results
    
    def get_environment_specific_config(self, environment: str) -> EnterpriseConfig:
        """Get configuration for specific environment"""
        # Load environment-specific overrides
        env_config_path = self.config_path.replace('.yaml', f'.{environment}.yaml')
        
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, 'r') as f:
                    env_overrides = yaml.safe_load(f)
                
                # Create a copy of current config
                config_dict = asdict(self.config)
                
                # Apply environment overrides
                self._deep_update(config_dict, env_overrides)
                
                return self._dict_to_config(config_dict)
                
            except Exception as e:
                logger.error(f"Error loading environment config {env_config_path}: {e}")
        
        return self.config

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> EnterpriseConfig:
    """Get current enterprise configuration"""
    return get_config_manager().get_config()

def update_config(updates: Dict[str, Any]):
    """Update enterprise configuration"""
    get_config_manager().update_config(updates)

# Environment-specific configurations
DEVELOPMENT_CONFIG_OVERRIDES = {
    "environment": "development",
    "debug": True,
    "log_level": "DEBUG",
    "monitoring": {
        "interval_seconds": 10
    },
    "bulk_processing": {
        "max_workers": 2,
        "batch_size": 10
    }
}

STAGING_CONFIG_OVERRIDES = {
    "environment": "staging",
    "debug": False,
    "log_level": "INFO",
    "monitoring": {
        "interval_seconds": 30
    }
}

PRODUCTION_CONFIG_OVERRIDES = {
    "environment": "production",
    "debug": False,
    "log_level": "WARNING",
    "monitoring": {
        "interval_seconds": 30,
        "metrics_retention_hours": 720  # 30 days
    },
    "bulk_processing": {
        "max_workers": 16,
        "batch_size": 100
    },
    "security": {
        "jwt_expiration_hours": 8,  # Shorter in production
        "max_login_attempts": 3,
        "lockout_duration_minutes": 60
    }
}