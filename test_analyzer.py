#!/usr/bin/env python3
"""
Test ResumeAnalyzer class instantiation
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=== Testing ResumeAnalyzer Class ===")

try:
    print("1. Importing ResumeAnalyzer...")
    import resume_analyzer as ra_module
    ResumeAnalyzer = ra_module.ResumeAnalyzer
    print("✅ ResumeAnalyzer imported successfully")
    
    print("\n2. Creating ResumeAnalyzer instance...")
    analyzer = ResumeAnalyzer()
    print("✅ ResumeAnalyzer instantiated successfully")
    print(f"Instance type: {type(analyzer)}")
    print(f"Available methods: {[method for method in dir(analyzer) if not method.startswith('_')]}")
    
    print("\n3. Testing basic functionality...")
    # Test if the analyzer has the expected methods
    expected_methods = ['analyze_resume', 'generate_report']
    for method in expected_methods:
        if hasattr(analyzer, method):
            print(f"✅ Method '{method}' found")
        else:
            print(f"❌ Method '{method}' missing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== ResumeAnalyzer Test Complete ===")