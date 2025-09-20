"""
Test script to verify all navigation pages work correctly
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_analyzer_import():
    """Test if the analyzer can be imported and initialized"""
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        analyzer = ResumeAnalyzer()
        print("✅ ResumeAnalyzer import and initialization: SUCCESS")
        return analyzer
    except Exception as e:
        print(f"❌ ResumeAnalyzer import failed: {e}")
        return None

def test_analyzer_methods(analyzer):
    """Test if all required methods exist and work"""
    if analyzer is None:
        return False
    
    # Test get_system_statistics
    try:
        stats = analyzer.get_system_statistics(30)
        required_keys = ['total_analyses', 'average_score', 'total_resumes', 'total_jobs']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        print("✅ get_system_statistics: SUCCESS")
    except Exception as e:
        print(f"❌ get_system_statistics failed: {e}")
        return False
    
    # Test health_check
    try:
        health = analyzer.health_check()
        required_keys = ['status', 'components']
        for key in required_keys:
            assert key in health, f"Missing key: {key}"
        print("✅ health_check: SUCCESS")
    except Exception as e:
        print(f"❌ health_check failed: {e}")
        return False
    
    # Test analyze_resume_for_job (basic structure check)
    try:
        # Just check if the method exists
        assert hasattr(analyzer, 'analyze_resume_for_job'), "analyze_resume_for_job method missing"
        print("✅ analyze_resume_for_job method exists: SUCCESS")
    except Exception as e:
        print(f"❌ analyze_resume_for_job check failed: {e}")
        return False
    
    return True

def test_web_app_imports():
    """Test if web app can import all required modules"""
    try:
        # Test basic imports
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        import tempfile
        print("✅ Web app basic imports: SUCCESS")
        
        # Test analyzer import in web app context
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from simple_resume_analyzer import ResumeAnalyzer
        print("✅ Web app analyzer import: SUCCESS")
        
        return True
    except Exception as e:
        print(f"❌ Web app imports failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Resume Analyzer Navigation Components")
    print("=" * 50)
    
    # Test 1: Analyzer import and initialization
    analyzer = test_analyzer_import()
    
    # Test 2: Analyzer methods
    methods_ok = test_analyzer_methods(analyzer)
    
    # Test 3: Web app imports
    webapp_ok = test_web_app_imports()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    print(f"Analyzer Import: {'✅ PASS' if analyzer else '❌ FAIL'}")
    print(f"Analyzer Methods: {'✅ PASS' if methods_ok else '❌ FAIL'}")
    print(f"Web App Imports: {'✅ PASS' if webapp_ok else '❌ FAIL'}")
    
    if analyzer and methods_ok and webapp_ok:
        print("\n🎉 ALL TESTS PASSED - Navigation should work correctly!")
        print("\n📋 Available Navigation Pages:")
        print("1. 🏠 Dashboard - System overview and quick actions")
        print("2. 📝 Analyze Resume - Single resume analysis")
        print("3. 📊 Batch Analysis - Multiple resume analysis")
        print("4. 🔍 View Results - Browse previous results")
        print("5. 📈 Reports & Analytics - System reports")
        print("6. ⚙️ System Status - Health checks and diagnostics")
    else:
        print("\n❌ SOME TESTS FAILED - Check errors above")

if __name__ == "__main__":
    main()