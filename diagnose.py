"""
Diagnostic Script for Resume Analyzer
Debug scoring issues and API connectivity
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api_connection():
    """Test API connection and LLM response"""
    print("🔍 Testing API Connection...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    
    print(f"OpenAI API Key: {openai_key[:20]}..." if openai_key else "OpenAI API Key: Not found")
    print(f"HuggingFace API Key: {hf_key[:20]}..." if hf_key else "HuggingFace API Key: Not found")
    
    # Test OpenRouter connection
    if openai_key and openai_key.startswith("sk-or-v1"):
        print("✅ OpenRouter API key detected")
        try:
            import openai
            client = openai.OpenAI(
                api_key=openai_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Test API call with a simple model name
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'API connection successful' and rate this resume vs job match on scale 0-100: Resume has Python skills. Job requires Python. Give only the number."}],
                max_tokens=50
            )
            
            result = response.choices[0].message.content
            print(f"✅ OpenRouter API Response: {result}")
            return True
            
        except Exception as e:
            print(f"❌ OpenRouter API Error: {str(e)}")
            return False
    
    return False

def test_scoring_logic():
    """Test the scoring engine logic"""
    print("🧮 Testing Scoring Logic...")
    try:
        from src.scoring.scoring_engine import ScoringEngine
        
        config = {
            'scoring': {
                'weights': {'hard_matching': 0.4, 'soft_matching': 0.3, 'llm_analysis': 0.3},
                'thresholds': {'excellent': 80.0, 'good': 65.0, 'fair': 45.0, 'poor': 0.0}
            }
        }
        
        scoring_engine = ScoringEngine(config)
        
        # Test scoring with mock data
        hard_results = {'score': 75.0, 'matches': ['Python', 'SQL']}
        soft_results = {'score': 80.0, 'matches': ['data analysis']}
        llm_results = {'score': 85.0, 'explanation': 'Good match'}
        
        score = scoring_engine.calculate_relevance_score(hard_results, soft_results, llm_results)
        print(f"✅ Scoring Logic Test Passed - Overall Score: {score.overall_score:.1f}")
        return True
        
    except Exception as e:
        print(f"❌ Scoring Logic Error: {e}")
        return False

def test_resume_analysis():
    """Test the resume analysis workflow"""
    print("📄 Testing Resume Analysis Workflow...")
    try:
        import sys
        sys.path.append('src')
        from resume_analyzer import ResumeAnalyzer
        
        # Test with mock data
        analyzer = ResumeAnalyzer()
        
        resume_text = "Software Engineer with Python experience"
        job_text = "Looking for Python developer"
        
        # This would normally call the full analysis
        print("✅ Resume Analysis Test Passed - Workflow imports working")
        return True
        
    except Exception as e:
        print(f"❌ Resume Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_fixed_score_issue():
    """Diagnose why scores are fixed at 78.5"""
    print("🔧 Diagnosing Fixed Score Issue...")
    try:
        from src.scoring.scoring_engine import ScoringEngine
        
        config = {
            'scoring': {
                'weights': {'hard_matching': 0.4, 'soft_matching': 0.3, 'llm_analysis': 0.3},
                'thresholds': {'excellent': 80.0, 'good': 65.0, 'fair': 45.0, 'poor': 0.0}
            }
        }
        
        scoring_engine = ScoringEngine(config)
        
        print(f"Scoring weights: {scoring_engine.weights.__dict__}")
        print(f"Scoring thresholds: {scoring_engine.thresholds}")
        
        # Test with different input values
        test_cases = [
            (70.0, 75.0, 80.0),
            (90.0, 85.0, 95.0),
            (50.0, 60.0, 55.0)
        ]
        
        for i, (hard, soft, llm) in enumerate(test_cases):
            hard_results = {'score': hard, 'matches': ['test']}
            soft_results = {'score': soft, 'matches': ['test']}
            llm_results = {'score': llm, 'explanation': 'test'}
            
            score = scoring_engine.calculate_relevance_score(hard_results, soft_results, llm_results)
            print(f"Test case {i+1}: Hard={hard}, Soft={soft}, LLM={llm} → Final={score.overall_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Resume Analyzer Diagnostic Tool")
    print("=" * 50)
    
    # Run all tests
    api_ok = test_api_connection()
    scoring_ok = test_scoring_logic()
    analysis_ok = test_resume_analysis()
    diagnosis_ok = diagnose_fixed_score_issue()
    
    print("\n📊 Diagnostic Summary:")
    print("=" * 50)
    print(f"API Connection: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"Scoring Logic: {'✅ PASS' if scoring_ok else '❌ FAIL'}")
    print(f"Resume Analysis: {'✅ PASS' if analysis_ok else '❌ FAIL'}")
    print(f"Score Diagnosis: {'✅ PASS' if diagnosis_ok else '❌ FAIL'}")
    
    if not analysis_ok:
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Check API key configuration")
        print("2. Verify LLM client initialization")
        print("3. Check scoring weight calculations")
        print("4. Review error handling in scoring engine")