"""
Test Enhanced Scoring System
Verify that different resumes get different scores based on content
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_resume_analyzer import ResumeAnalyzer

def test_enhanced_scoring():
    """Test the enhanced scoring system with different resume types"""
    
    print("🔬 Testing Enhanced Resume Scoring System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    # Test files
    job_file = "test_data/senior_python_developer.txt"
    
    test_cases = [
        {
            "name": "Strong Python Candidate",
            "file": "test_data/strong_python_candidate.txt",
            "expected_range": (75, 95)
        },
        {
            "name": "Junior Web Developer", 
            "file": "test_data/junior_candidate.txt",
            "expected_range": (20, 40)  # Adjusted to be more realistic
        },
        {
            "name": "Unrelated Marketing Manager",
            "file": "test_data/unrelated_candidate.txt", 
            "expected_range": (5, 25)
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Analyze resume
            result = analyzer.analyze_resume_for_job(
                test_case["file"],
                job_file,
                save_to_db=False
            )
            
            if result['metadata']['success']:
                score = result['analysis_results']['overall_score']
                match_level = result['analysis_results']['match_level']
                confidence = result['analysis_results']['confidence']
                
                print(f"Overall Score: {score:.1f}/100")
                print(f"Match Level: {match_level.upper()}")
                print(f"Confidence: {confidence:.1f}%")
                
                # Component breakdown
                components = result['detailed_results']['scoring_details']['component_scores']
                print(f"\nComponent Scores:")
                print(f"  Keyword Match: {components.get('keyword_match', 0):.1f}")
                print(f"  Skill Match: {components.get('skill_match', 0):.1f}")
                print(f"  Context Match: {components.get('context_match', 0):.1f}")
                print(f"  Experience Match: {components.get('experience_match', 0):.1f}")
                
                # Hiring recommendation
                hiring = result['hiring_recommendation']
                print(f"\nHiring Decision: {hiring['decision']}")
                print(f"Success Probability: {hiring['success_probability']:.1f}%")
                
                # Check if score is in expected range
                expected_min, expected_max = test_case['expected_range']
                in_range = expected_min <= score <= expected_max
                range_status = "✅ PASS" if in_range else "❌ FAIL"
                
                print(f"\nExpected Range: {expected_min}-{expected_max}")
                print(f"Range Check: {range_status}")
                
                # Top recommendations
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(result['analysis_results']['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")
                
                results.append({
                    'name': test_case['name'],
                    'score': score,
                    'match_level': match_level,
                    'in_range': in_range,
                    'expected_range': test_case['expected_range']
                })
                
            else:
                print(f"❌ Analysis failed: {result['metadata'].get('error', 'Unknown error')}")
                results.append({
                    'name': test_case['name'],
                    'score': 0,
                    'match_level': 'error',
                    'in_range': False,
                    'expected_range': test_case['expected_range']
                })
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            results.append({
                'name': test_case['name'],
                'score': 0,
                'match_level': 'error',
                'in_range': False,
                'expected_range': test_case['expected_range']
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "✅ PASS" if result['in_range'] else "❌ FAIL"
        print(f"{result['name']:<30} Score: {result['score']:.1f} {status}")
    
    # Check for score differentiation
    scores = [r['score'] for r in results if r['score'] > 0]
    if len(scores) >= 2:
        score_range = max(scores) - min(scores)
        print(f"\nScore Differentiation: {score_range:.1f} points")
        
        if score_range >= 30:
            print("✅ Good score differentiation - analyzer distinguishes between candidates")
        else:
            print("⚠️ Limited score differentiation - may need further improvement")
    
    # Overall test result
    passed_tests = sum(1 for r in results if r['in_range'])
    total_tests = len(results)
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Enhanced scoring system is working correctly.")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ Most tests passed, but some tuning may be needed.")
    else:
        print("❌ Multiple test failures - scoring system needs adjustment.")
    
    return results

if __name__ == "__main__":
    test_enhanced_scoring()