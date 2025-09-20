#!/usr/bin/env python3
"""
Test script to verify the _create_error_score NoneType error fix
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_analyzer():
    """Test the simple analyzer"""
    print("🧪 Testing Simple Resume Analyzer...")
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Test with sample data
        resume_text = """
        John Doe
        Software Developer
        Experience: Python, JavaScript, React
        Education: Computer Science degree
        """
        
        job_text = "Looking for Python developer with web experience"
        
        result = analyzer.analyze_resume(resume_text, job_text)
        print(f"✅ Simple analyzer test passed - Score: {result['overall_score']}")
        return True
        
    except Exception as e:
        print(f"❌ Simple analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_full_analyzer():
    """Test the full analyzer"""
    print("\n🧪 Testing Full Resume Analyzer...")
    try:
        from resume_analyzer import ResumeAnalyzer
        
        # Try to create analyzer - this might fail due to heavy dependencies
        analyzer = ResumeAnalyzer()
        
        # Create test files
        test_resume = "test_resume.txt"
        test_job = "test_job.txt"
        
        with open(test_resume, 'w') as f:
            f.write("""
            Jane Smith
            Senior Software Engineer
            
            Experience:
            - 5 years Python development
            - Machine learning projects
            - Web applications with Django/Flask
            
            Skills:
            - Python, JavaScript, SQL
            - TensorFlow, PyTorch
            - AWS, Docker, Kubernetes
            
            Education:
            - M.S. Computer Science, XYZ University
            """)
        
        with open(test_job, 'w') as f:
            f.write("""
            Senior Python Developer Position
            
            Requirements:
            - 3+ years Python experience
            - Machine learning background
            - Cloud platforms (AWS/Azure)
            - Strong problem-solving skills
            """)
        
        # This should now handle the None scoring_engine gracefully
        result = analyzer.analyze_resume_for_job(test_resume, test_job, save_to_db=False)
        
        # Clean up test files
        os.remove(test_resume)
        os.remove(test_job)
        
        print(f"✅ Full analyzer test passed - Analysis completed successfully")
        print(f"   Status: {result.get('status', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Full analyzer test failed: {e}")
        # Clean up test files if they exist
        for f in ['test_resume.txt', 'test_job.txt']:
            if os.path.exists(f):
                os.remove(f)
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Resume Analyzer Error Fix Tests")
    print("=" * 50)
    
    results = []
    
    # Test simple analyzer
    results.append(test_simple_analyzer())
    
    # Test full analyzer
    results.append(test_full_analyzer())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! The _create_error_score NoneType error has been fixed!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    
    print(f"Simple Analyzer: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Full Analyzer: {'✅ PASS' if results[1] else '❌ FAIL'}")

if __name__ == "__main__":
    main()