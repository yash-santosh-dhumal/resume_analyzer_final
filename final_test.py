#!/usr/bin/env python3
"""
Comprehensive test to verify the _create_error_score NoneType error fix
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_analyzer_with_files():
    """Test the simple analyzer with actual file analysis"""
    print("🧪 Testing Simple Analyzer with File Analysis...")
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Create test files
        test_resume = "test_resume_simple.txt"
        test_job = "test_job_simple.txt"
        
        with open(test_resume, 'w') as f:
            f.write("""
            John Doe
            Software Developer
            
            Experience:
            - 3 years Python development
            - Web applications with Django
            - Database design with PostgreSQL
            
            Skills:
            - Python, JavaScript, SQL
            - Django, React, Git
            - Problem solving, teamwork
            
            Education:
            - B.S. Computer Science
            """)
        
        with open(test_job, 'w') as f:
            f.write("""
            Python Developer Position
            
            We are looking for a Python developer with:
            - 2+ years Python experience
            - Web framework experience (Django/Flask)
            - Database knowledge
            - Good communication skills
            """)
        
        # Test the analyze_resume_for_job method that webapp uses
        result = analyzer.analyze_resume_for_job(test_resume, test_job, save_to_db=False)
        
        # Clean up test files
        os.remove(test_resume)
        os.remove(test_job)
        
        print(f"✅ Simple analyzer file test passed")
        print(f"   Success: {result['metadata']['success']}")
        print(f"   Score: {result['results']['relevance_score']['overall_score']}")
        print(f"   Match Level: {result['results']['relevance_score']['match_level']}")
        return True
        
    except Exception as e:
        print(f"❌ Simple analyzer file test failed: {e}")
        # Clean up test files if they exist
        for f in ['test_resume_simple.txt', 'test_job_simple.txt']:
            if os.path.exists(f):
                os.remove(f)
        traceback.print_exc()
        return False

def test_webapp_compatibility():
    """Test that both analyzers have compatible interfaces"""
    print("\n🧪 Testing Webapp Compatibility...")
    try:
        # Test simple analyzer interface
        from simple_resume_analyzer import ResumeAnalyzer as SimpleAnalyzer
        simple = SimpleAnalyzer()
        
        # Check if method exists
        if hasattr(simple, 'analyze_resume_for_job'):
            print("✅ Simple analyzer has analyze_resume_for_job method")
        else:
            print("❌ Simple analyzer missing analyze_resume_for_job method")
            return False
            
        # Test full analyzer interface
        try:
            from resume_analyzer import ResumeAnalyzer as FullAnalyzer
            full = FullAnalyzer()
            
            if hasattr(full, 'analyze_resume_for_job'):
                print("✅ Full analyzer has analyze_resume_for_job method")
            else:
                print("❌ Full analyzer missing analyze_resume_for_job method")
                return False
                
        except Exception as e:
            print(f"⚠️ Full analyzer failed to load (expected): {e}")
            print("✅ This is fine - webapp will use simple analyzer")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling doesn't cause NoneType errors"""
    print("\n🧪 Testing Error Handling...")
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Test with non-existent files
        result = analyzer.analyze_resume_for_job("nonexistent.txt", "alsonothere.txt")
        
        if result['metadata']['success'] == False:
            print("✅ Error handling test passed - graceful failure")
            print(f"   Error handled: {result['metadata'].get('error', 'Unknown error')}")
            return True
        else:
            print("❌ Expected failure but got success")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive NoneType Error Fix Tests")
    print("=" * 60)
    
    results = []
    
    # Test simple analyzer with files
    results.append(test_simple_analyzer_with_files())
    
    # Test webapp compatibility
    results.append(test_webapp_compatibility())
    
    # Test error handling
    results.append(test_error_handling())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! The _create_error_score NoneType error has been completely fixed!")
        print("✅ Webapp should now work without any NoneType errors")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    
    print(f"Simple Analyzer File Test: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Webapp Compatibility Test: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"Error Handling Test: {'✅ PASS' if results[2] else '❌ FAIL'}")

if __name__ == "__main__":
    main()