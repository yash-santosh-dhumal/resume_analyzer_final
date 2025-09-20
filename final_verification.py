#!/usr/bin/env python3
"""
Final test to verify the webapp works without the NoneType error
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_analyzer_working():
    """Test that simple analyzer works properly"""
    print("🧪 Testing Simple Analyzer (should work)...")
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Create test text files (not PDFs)
        test_resume = "test_resume.txt"
        test_job = "test_job.txt"
        
        with open(test_resume, 'w') as f:
            f.write("""
            John Doe
            Software Engineer
            
            Experience:
            - 3 years Python development
            - Web applications with Flask
            - Database work with PostgreSQL
            
            Skills:
            - Python, JavaScript, SQL
            - Flask, React, Git
            - Problem solving
            
            Education:
            - B.S. Computer Science
            """)
        
        with open(test_job, 'w') as f:
            f.write("""
            Python Developer Position
            
            Requirements:
            - 2+ years Python experience
            - Web framework experience
            - Database knowledge
            - Good communication skills
            """)
        
        # Test the method that webapp calls
        result = analyzer.analyze_resume_for_job(test_resume, test_job, save_to_db=False)
        
        # Clean up
        os.remove(test_resume)
        os.remove(test_job)
        
        print(f"✅ Simple analyzer test passed!")
        print(f"   Success: {result['metadata']['success']}")
        print(f"   Score: {result['results']['relevance_score']['overall_score']}")
        print(f"   Error: {result['metadata'].get('error', 'None')}")
        
        return result['metadata']['success']
        
    except Exception as e:
        print(f"❌ Simple analyzer test failed: {e}")
        # Clean up
        for f in ['test_resume.txt', 'test_job.txt']:
            if os.path.exists(f):
                os.remove(f)
        traceback.print_exc()
        return False

def test_webapp_analyzer_type():
    """Test which analyzer the webapp would use"""
    print("\n🧪 Testing Webapp Analyzer Selection...")
    try:
        # Simulate webapp's analyzer selection logic
        try:
            from resume_analyzer import ResumeAnalyzer
            analyzer_type = "full"
            print("✅ Full analyzer would be loaded")
        except Exception as e:
            print(f"⚠️ Full analyzer failed ({e}), using simplified version")
            try:
                from simple_resume_analyzer import ResumeAnalyzer
                analyzer_type = "simple"
                print("✅ Simple analyzer would be loaded")
            except Exception as e2:
                print(f"❌ Both analyzers failed: {e2}")
                analyzer_type = "none"
        
        print(f"Webapp would use: {analyzer_type} analyzer")
        return analyzer_type != "none"
        
    except Exception as e:
        print(f"❌ Analyzer selection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Final Verification Tests")
    print("=" * 40)
    
    # Test simple analyzer
    simple_works = test_simple_analyzer_working()
    
    # Test webapp selection
    webapp_works = test_webapp_analyzer_type()
    
    print("\n" + "=" * 40)
    print("📊 Final Results:")
    
    if simple_works and webapp_works:
        print("🎉 SUCCESS! All tests passed!")
        print("✅ The webapp should now work without NoneType errors")
    else:
        print("⚠️ Some issues remain")
        
    print(f"Simple Analyzer: {'✅ WORKING' if simple_works else '❌ FAILED'}")
    print(f"Webapp Compatible: {'✅ YES' if webapp_works else '❌ NO'}")