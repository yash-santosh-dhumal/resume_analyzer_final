#!/usr/bin/env python3
"""
Simple test for ResumeAnalyzer
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing ResumeAnalyzer import...")
try:
    import resume_analyzer as ra_module
    print("✅ Module imported")
    
    ResumeAnalyzer = ra_module.ResumeAnalyzer
    print("✅ Class found")
    
    print("Creating instance...")
    analyzer = ResumeAnalyzer()
    print("✅ Instance created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()