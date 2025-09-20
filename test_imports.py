#!/usr/bin/env python3
"""
Test script to diagnose import issues
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=== Testing Resume Analyzer Imports ===")

try:
    print("1. Testing config.settings import...")
    from config.settings import load_config
    print("✅ config.settings imported successfully")
except Exception as e:
    print(f"❌ config.settings failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing parsers import...")
    from parsers import DocumentParser, TextNormalizer
    print("✅ parsers imported successfully")
except Exception as e:
    print(f"❌ parsers failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing matching import...")
    from matching import HardMatcher, SoftMatcher, EmbeddingGenerator
    print("✅ matching imported successfully")
except Exception as e:
    print(f"❌ matching failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4. Testing llm import...")
    from llm import LLMReasoningEngine
    print("✅ llm imported successfully")
except Exception as e:
    print(f"❌ llm failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n5. Testing scoring import...")
    from scoring import ScoringEngine, RelevanceScore
    print("✅ scoring imported successfully")
except Exception as e:
    print(f"❌ scoring failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n6. Testing database import...")
    from database import DatabaseManager, ExportManager
    print("✅ database imported successfully")
except Exception as e:
    print(f"❌ database failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n7. Testing resume_analyzer module import...")
    import resume_analyzer
    print("✅ resume_analyzer module imported successfully")
    print(f"Available attributes: {[attr for attr in dir(resume_analyzer) if not attr.startswith('_')]}")
    
    if hasattr(resume_analyzer, 'ResumeAnalyzer'):
        print("✅ ResumeAnalyzer class found in module")
    else:
        print("❌ ResumeAnalyzer class NOT found in module")
        
except Exception as e:
    print(f"❌ resume_analyzer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Import Test Complete ===")