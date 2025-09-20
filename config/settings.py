# Configuration settings for the application

import os
from typing import Dict, Any

# API Keys and External Services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_huggingface_api_key_here")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/resume_analyzer.db")
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite or postgresql

# File Upload Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"

# Matching Algorithm Weights
MATCHING_WEIGHTS = {
    "hard_match_weight": 0.4,
    "soft_match_weight": 0.6,
    "keyword_weight": 0.3,
    "skill_weight": 0.4,
    "education_weight": 0.2,
    "experience_weight": 0.1
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.3,
    "max_tokens": 1000,
    "backup_model": "sentence-transformers/all-MiniLM-L6-v2"
}

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    "type": "chroma",  # chroma, faiss, or pinecone
    "persist_directory": "./data/chroma_db",
    "collection_name": "resume_jd_embeddings"
}

# Scoring Thresholds
SCORING_THRESHOLDS = {
    "high_match": 75,
    "medium_match": 50,
    "low_match": 25
}

# Text Processing Configuration
TEXT_PROCESSING = {
    "spacy_model": "en_core_web_sm",
    "remove_stopwords": True,
    "lemmatize": True,
    "min_word_length": 2
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "file": "logs/app.log",
    "rotation": "1 week",
    "retention": "1 month"
}

# Web Application Configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "debug": False,
    "page_title": "Resume Relevance Analyzer - Innomatics Research Labs",
    "page_icon": "📄"
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "api_keys": {
            "openai": OPENAI_API_KEY,
            "huggingface": HUGGINGFACE_API_KEY
        },
        "database": {
            "url": DATABASE_URL,
            "type": DATABASE_TYPE
        },
        "file_upload": {
            "max_size": MAX_FILE_SIZE,
            "allowed_extensions": ALLOWED_EXTENSIONS,
            "upload_folder": UPLOAD_FOLDER,
            "export_folder": EXPORT_FOLDER
        },
        "matching_weights": MATCHING_WEIGHTS,
        "llm": LLM_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "scoring": SCORING_THRESHOLDS,
        "text_processing": TEXT_PROCESSING,
        "logging": LOGGING_CONFIG,
        "web": WEB_CONFIG
    }

# Alias for compatibility
def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration (alias for get_config)"""
    # Ignore config_path for now, use default configuration
    return get_config()