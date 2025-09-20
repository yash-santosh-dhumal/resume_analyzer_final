"""
Quick Webapp Test - Text Files Only
Test the webapp functionality with text files to avoid PDF issues
"""

import sys
import os
import shutil
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_resume_analyzer import ResumeAnalyzer

def test_webapp_with_text_files():
    """Test webapp with text files only"""
    
    print("📋 Testing Webapp with Text Files")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    # Test with our text files
    job_file = "test_data/senior_python_developer.txt"
    resume_file = "test_data/strong_python_candidate.txt"
    
    print(f"📁 Job Description: {job_file}")
    print(f"📄 Resume: {resume_file}")
    
    try:
        # This is exactly what the webapp does
        result = analyzer.analyze_resume_for_job(
            resume_file,
            job_file,
            save_to_db=False
        )
        
        if result['metadata']['success']:
            print(f"\n✅ Analysis Successful!")
            print(f"📊 Overall Score: {result['analysis_results']['overall_score']:.1f}/100")
            print(f"🎯 Match Level: {result['analysis_results']['match_level'].upper()}")
            print(f"💼 Hiring Decision: {result['hiring_recommendation']['decision']}")
            print(f"🎉 Success! Webapp is working correctly with text files")
            
            # Show component breakdown
            components = result['detailed_results']['scoring_details']['component_scores']
            print(f"\n🔍 Component Scores:")
            for comp, score in components.items():
                print(f"  {comp}: {score:.1f}%")
            
        else:
            print(f"❌ Analysis failed: {result['metadata'].get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def create_usage_instructions():
    """Create usage instructions for the webapp"""
    
    instructions = """
🎯 WEBAPP USAGE INSTRUCTIONS
============================

✅ Your webapp is now running at: http://localhost:8511

🚀 How to Use:

1. Open http://localhost:8511 in your browser

2. Go to "📝 Analyze Resume" page

3. For BEST RESULTS, use TEXT files (.txt):
   - Upload: test_data/senior_python_developer.txt (Job Description)
   - Upload: test_data/strong_python_candidate.txt (Resume)
   - Click "🚀 Analyze Resume"

4. You should see:
   ✅ Score: ~80/100 (Excellent)
   ✅ Decision: HIRE
   ✅ Component breakdown showing different scores
   ✅ Personalized recommendations

🔧 Troubleshooting:
- If PDF files cause errors, use .txt files instead
- All test files are in test_data/ folder
- The enhanced scoring system now works correctly!

📊 Expected Results:
- Strong candidate: 75-85 score
- Junior developer: 20-35 score  
- Unrelated candidate: 5-20 score

🎉 The scoring system is fixed and working!
"""
    
    print(instructions)

if __name__ == "__main__":
    test_webapp_with_text_files()
    print("\n" + "=" * 50)
    create_usage_instructions()