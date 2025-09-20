#!/usr/bin/env python3
"""
Test script to reproduce the exact error from webapp with PDF files
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_with_actual_pdf_files():
    """Test with actual PDF files like the webapp"""
    print("🧪 Testing with PDF files like webapp...")
    try:
        from resume_analyzer import ResumeAnalyzer
        
        # Initialize analyzer like webapp does
        analyzer = ResumeAnalyzer()
        
        # Create test PDF files (simulate what webapp receives)
        # Since we can't create real PDFs easily, let's create files with PDF extension
        # that will trigger the PDF parsing path
        
        test_resume = "RESUME-2.pdf"  # Same name as in webapp
        test_job = "sample_jd_1.pdf"  # Same name as in webapp
        
        # Create dummy PDF content (this will fail PDF parsing, which might trigger the error)
        with open(test_resume, 'w') as f:
            f.write("Dummy PDF content that will trigger parsing error")
        
        with open(test_job, 'w') as f:
            f.write("Dummy PDF job description content")
        
        # This should trigger the exact same code path as the webapp
        result = analyzer.analyze_resume_for_job(test_resume, test_job, save_to_db=True)
        
        # Clean up
        os.remove(test_resume)
        os.remove(test_job)
        
        print(f"Analysis result:")
        print(f"  Success: {result['metadata']['success']}")
        if not result['metadata']['success']:
            print(f"  Error: {result['metadata'].get('error', 'No error message')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        # Clean up test files if they exist
        for f in ['RESUME-2.pdf', 'sample_jd_1.pdf']:
            if os.path.exists(f):
                os.remove(f)
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_actual_pdf_files()