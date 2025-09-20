"""
Resume Analyzer System
Complete AI-powered resume relevance analysis system
"""

import os
import sys

# Add current directory to path for absolute imports
sys.path.append(os.path.dirname(__file__))

from resume_analyzer import ResumeAnalyzer
from config.settings import load_config

__version__ = "1.0.0"

__all__ = [
    'ResumeAnalyzer',
    'load_config'
]
# Main application entry point