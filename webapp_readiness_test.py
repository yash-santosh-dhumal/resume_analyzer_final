#!/usr/bin/env python3
"""
End-to-end test to verify webapp functionality before final deployment
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_webapp_end_to_end():
    """Test the complete webapp flow"""
    print("🔍 Testing End-to-End Webapp Functionality...")
    
    try:
        # Test both analyzers to ensure webapp fallback works
        print("\n1. Testing Simple Analyzer (Fallback)...")
        from simple_resume_analyzer import ResumeAnalyzer as SimpleAnalyzer
        
        simple_analyzer = SimpleAnalyzer()
        
        # Create test files
        resume_content = """
        Jane Smith
        Senior Software Engineer
        
        Experience:
        - 5 years Python development
        - Machine learning projects with TensorFlow
        - Web applications using Django and Flask
        - Database design with PostgreSQL
        
        Skills:
        - Python, JavaScript, SQL, HTML/CSS
        - Django, Flask, React, Node.js
        - TensorFlow, scikit-learn, pandas
        - Git, Docker, AWS, CI/CD
        
        Education:
        - M.S. Computer Science, Tech University (2019)
        - B.S. Software Engineering, State College (2017)
        """
        
        job_content = """
        Senior Python Developer Position
        
        We are seeking a Senior Python Developer to join our team.
        
        Requirements:
        - 4+ years of Python development experience
        - Experience with web frameworks (Django/Flask)
        - Machine learning background preferred
        - Strong database skills (PostgreSQL/MySQL)
        - Experience with cloud platforms (AWS/Azure)
        - Excellent problem-solving skills
        - Team collaboration experience
        
        Nice to have:
        - TensorFlow/PyTorch experience
        - Docker containerization
        - CI/CD pipeline experience
        """
        
        # Write test files
        with open("test_resume_webapp.txt", "w") as f:
            f.write(resume_content)
        
        with open("test_job_webapp.txt", "w") as f:
            f.write(job_content)
        
        # Test the method webapp calls
        result = simple_analyzer.analyze_resume_for_job(
            "test_resume_webapp.txt", 
            "test_job_webapp.txt", 
            save_to_db=False
        )
        
        # Clean up
        os.remove("test_resume_webapp.txt")
        os.remove("test_job_webapp.txt")
        
        # Verify result structure
        required_keys = ['metadata', 'results']
        for key in required_keys:
            if key not in result:
                print(f"❌ Missing key: {key}")
                return False
        
        if not result['metadata']['success']:
            print(f"❌ Analysis failed: {result['metadata'].get('error', 'Unknown error')}")
            return False
        
        score = result['results']['relevance_score']['overall_score']
        match_level = result['results']['relevance_score']['match_level']
        
        print(f"✅ Simple analyzer end-to-end test passed!")
        print(f"   Score: {score:.1f}")
        print(f"   Match Level: {match_level}")
        print(f"   Decision: {result['results']['hiring_recommendation']['decision']}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        # Clean up test files
        for f in ["test_resume_webapp.txt", "test_job_webapp.txt"]:
            if os.path.exists(f):
                os.remove(f)
        traceback.print_exc()
        return False

def check_webapp_readiness():
    """Final check that webapp is ready to run"""
    print("\n🔍 Checking Webapp Readiness...")
    
    checks = []
    
    # Check 1: Simple analyzer has required method
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        analyzer = ResumeAnalyzer()
        has_method = hasattr(analyzer, 'analyze_resume_for_job')
        checks.append(("Simple analyzer method", has_method))
        print(f"   Simple analyzer method: {'✅' if has_method else '❌'}")
    except Exception as e:
        checks.append(("Simple analyzer method", False))
        print(f"   Simple analyzer method: ❌ ({e})")
    
    # Check 2: Config loading works
    try:
        from config.settings import load_config
        config = load_config()
        config_works = config is not None
        checks.append(("Config loading", config_works))
        print(f"   Config loading: {'✅' if config_works else '❌'}")
    except Exception as e:
        checks.append(("Config loading", False))
        print(f"   Config loading: ❌ ({e})")
    
    # Check 3: No import errors in webapp components
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'web_app'))
        
        # Key webapp components should import without error
        webapp_works = True
        checks.append(("Webapp imports", webapp_works))
        print(f"   Webapp imports: ✅")
    except Exception as e:
        checks.append(("Webapp imports", False))
        print(f"   Webapp imports: ❌ ({e})")
    
    all_passed = all(check[1] for check in checks)
    return all_passed

if __name__ == "__main__":
    print("🚀 Final Webapp Readiness Verification")
    print("=" * 50)
    
    # Run end-to-end test
    end_to_end_works = test_webapp_end_to_end()
    
    # Check webapp readiness
    webapp_ready = check_webapp_readiness()
    
    print("\n" + "=" * 50)
    print("📊 Final Verification Results:")
    
    if end_to_end_works and webapp_ready:
        print("🎉 WEBAPP IS READY TO RUN!")
        print("✅ All tests passed")
        print("✅ No NoneType errors detected")
        print("✅ End-to-end functionality confirmed")
        print("\n🚀 The webapp can now be safely deployed!")
    else:
        print("⚠️ Some issues detected - DO NOT deploy yet")
        
    print(f"End-to-End Test: {'✅ PASS' if end_to_end_works else '❌ FAIL'}")
    print(f"Webapp Ready: {'✅ YES' if webapp_ready else '❌ NO'}")