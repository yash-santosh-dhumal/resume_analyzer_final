"""
Test and Debug Batch Analysis Functionality
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_files():
    """Create sample test files for batch analysis"""
    test_dir = Path("test_batch_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample job description
    jd_content = """
Software Developer Position
Company: Tech Solutions Inc.

We are looking for a skilled Software Developer with the following qualifications:
- 3+ years of Python development experience
- Experience with web frameworks (Django, Flask)
- Knowledge of databases (PostgreSQL, MySQL)
- Familiarity with version control (Git)
- Strong problem-solving skills
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Develop and maintain web applications
- Write clean, efficient code
- Collaborate with cross-functional teams
- Participate in code reviews
"""
    
    # Create sample resumes
    resume1_content = """
John Doe
Software Engineer
Email: john.doe@email.com
Phone: +1-555-0123

EXPERIENCE:
Software Engineer at ABC Corp (2020-2023)
- Developed web applications using Python and Django
- Worked with PostgreSQL databases
- Used Git for version control
- Collaborated with team of 5 developers

EDUCATION:
Bachelor of Science in Computer Science
University of Technology (2016-2020)

SKILLS:
Python, Django, PostgreSQL, Git, HTML, CSS, JavaScript
"""
    
    resume2_content = """
Jane Smith
Python Developer
Email: jane.smith@email.com
Phone: +1-555-0456

EXPERIENCE:
Python Developer at XYZ Inc (2019-2023)
- Built web applications using Flask
- Experience with MySQL databases
- Used Git for version control
- Strong problem-solving abilities

Junior Developer at StartupCo (2018-2019)
- Learned Python programming
- Basic web development

EDUCATION:
Bachelor of Computer Engineering
State University (2014-2018)

SKILLS:
Python, Flask, MySQL, Git, Problem Solving, Web Development
"""
    
    resume3_content = """
Mike Johnson
Marketing Specialist
Email: mike.johnson@email.com
Phone: +1-555-0789

EXPERIENCE:
Marketing Specialist at Brand Corp (2020-2023)
- Managed social media campaigns
- Analyzed marketing metrics
- Created content for various platforms

Marketing Assistant at MediaCo (2018-2020)
- Assisted with campaign development
- Conducted market research

EDUCATION:
Bachelor of Business Administration
Business College (2014-2018)

SKILLS:
Social Media Marketing, Content Creation, Analytics, Communication
"""
    
    # Write files
    with open(test_dir / "job_description.txt", "w") as f:
        f.write(jd_content)
    
    with open(test_dir / "resume_john_doe.txt", "w") as f:
        f.write(resume1_content)
    
    with open(test_dir / "resume_jane_smith.txt", "w") as f:
        f.write(resume2_content)
    
    with open(test_dir / "resume_mike_johnson.txt", "w") as f:
        f.write(resume3_content)
    
    return test_dir

def test_batch_analysis():
    """Test the batch analysis functionality"""
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        # Initialize analyzer
        analyzer = ResumeAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Create test files
        test_dir = create_test_files()
        print(f"✅ Test files created in {test_dir}")
        
        # Test files
        jd_file = test_dir / "job_description.txt"
        resume_files = [
            test_dir / "resume_john_doe.txt",
            test_dir / "resume_jane_smith.txt", 
            test_dir / "resume_mike_johnson.txt"
        ]
        
        print(f"📋 Testing batch analysis with {len(resume_files)} resumes...")
        
        # Run batch analysis
        results = []
        for i, resume_file in enumerate(resume_files):
            print(f"  Analyzing {i+1}/{len(resume_files)}: {resume_file.name}")
            
            try:
                result = analyzer.analyze_resume_for_job(
                    str(resume_file), 
                    str(jd_file), 
                    save_to_db=False
                )
                
                if result['metadata']['success']:
                    score = result['analysis_results']['overall_score']
                    match_level = result['analysis_results']['match_level']
                    candidate = result['resume_data']['candidate_name']
                    print(f"    ✅ {candidate}: Score {score:.1f} ({match_level})")
                    results.append(result)
                else:
                    print(f"    ❌ Analysis failed")
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        print(f"\n📊 Batch Analysis Summary:")
        print(f"  Total resumes: {len(resume_files)}")
        print(f"  Successful analyses: {len(results)}")
        print(f"  Failed analyses: {len(resume_files) - len(results)}")
        
        if results:
            avg_score = sum(r['analysis_results']['overall_score'] for r in results) / len(results)
            print(f"  Average score: {avg_score:.1f}")
            
            print(f"\n📋 Individual Results:")
            for result in results:
                candidate = result['resume_data']['candidate_name']
                score = result['analysis_results']['overall_score']
                match_level = result['analysis_results']['match_level']
                decision = result['hiring_recommendation']['decision']
                print(f"  • {candidate}: {score:.1f} ({match_level}) - {decision}")
        
        # Cleanup test files
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test files")
        
        return len(results) == len(resume_files)
        
    except Exception as e:
        print(f"❌ Batch analysis test failed: {str(e)}")
        return False

def main():
    """Run batch analysis test"""
    print("🚀 Testing Batch Analysis Functionality")
    print("=" * 50)
    
    success = test_batch_analysis()
    
    print("\n📊 Test Summary:")
    print("=" * 50)
    if success:
        print("✅ BATCH ANALYSIS TEST PASSED!")
        print("\n💡 Batch analysis should work correctly in the web app")
        print("   Make sure to:")
        print("   1. Upload a job description file")
        print("   2. Upload multiple resume files (use Ctrl+Click or Shift+Click)")
        print("   3. Click 'Start Batch Analysis' button")
    else:
        print("❌ BATCH ANALYSIS TEST FAILED!")
        print("   Check the errors above for issues")

if __name__ == "__main__":
    main()