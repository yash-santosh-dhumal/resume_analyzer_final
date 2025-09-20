#!/usr/bin/env python3
"""
Detailed import test for resume_analyzer.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=== Detailed Import Analysis ===")

# Test each import from resume_analyzer.py individually
imports_to_test = [
    ("parsers", "DocumentParser, TextNormalizer"),
    ("matching", "HardMatcher, SoftMatcher, EmbeddingGenerator"), 
    ("llm", "LLMReasoningEngine"),
    ("scoring", "ScoringEngine, RelevanceScore"),
    ("database", "DatabaseManager, ExportManager"),
    ("config.settings", "load_config")
]

all_passed = True

for module, items in imports_to_test:
    try:
        print(f"\n--- Testing: from {module} import {items} ---")
        exec(f"from {module} import {items}")
        print(f"✅ SUCCESS: {module}")
    except Exception as e:
        print(f"❌ FAILED: {module} - {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

print(f"\n=== Summary ===")
if all_passed:
    print("✅ All imports passed individually")
    print("Now testing complete resume_analyzer module...")
    try:
        import resume_analyzer
        print("✅ resume_analyzer module imported successfully")
        
        if hasattr(resume_analyzer, 'ResumeAnalyzer'):
            print("✅ ResumeAnalyzer class found")
        else:
            print("❌ ResumeAnalyzer class NOT found")
            print(f"Available attributes: {[attr for attr in dir(resume_analyzer) if not attr.startswith('_')]}")
            
    except Exception as e:
        print(f"❌ resume_analyzer module import failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Some imports failed - need to fix dependencies first")