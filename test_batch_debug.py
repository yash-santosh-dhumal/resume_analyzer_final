#!/usr/bin/env python3
"""Debug script to test batch analysis logic"""

import os
import sys
sys.path.append('.')

from src.simple_resume_analyzer import ResumeAnalyzer

def test_batch_analysis():
    """Test the batch analysis functionality"""
    
    print("Initializing Resume Analyzer...")
    analyzer = ResumeAnalyzer()
    
    # Test files
    jd_file = "sample_data/jds/JD/sample_jd_1.pdf"
    resume_files = [
        "sample_data/resumes/Resumes/resume - 1.pdf",
        "sample_data/resumes/Resumes/Resume - 10.pdf",
        "sample_data/resumes/Resumes/RESUME - 2.pdf"
    ]
    
    print(f"\nTesting with JD: {jd_file}")
    print(f"Testing with {len(resume_files)} resumes:")
    for i, resume in enumerate(resume_files, 1):
        print(f"  {i}. {resume}")
    
    # Run batch analysis
    results = []
    for i, resume_path in enumerate(resume_files):
        print(f"\nAnalyzing resume {i+1}/{len(resume_files)}: {resume_path}")
        
        try:
            result = analyzer.analyze_resume_for_job(resume_path, jd_file, save_to_db=False)
            
            # Add metadata like the app does
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['resume_filename'] = os.path.basename(resume_path)
            result['metadata']['success'] = True
            
            results.append(result)
            print(f"  ✅ Success: {result['resume_data'].get('candidate_name', 'Unknown')}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results.append({
                'metadata': {
                    'resume_filename': os.path.basename(resume_path),
                    'success': False,
                    'error': str(e)
                },
                'resume_data': {
                    'candidate_name': 'Unknown',
                    'filename': os.path.basename(resume_path)
                },
                'analysis_results': {
                    'overall_score': 0,
                    'match_level': 'poor'
                },
                'hiring_recommendation': {
                    'decision': 'REJECT',
                    'success_probability': 0
                }
            })
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total results: {len(results)}")
    
    # Filter like the app does
    successful_results = []
    failed_results = []
    
    for r in results:
        if r.get('metadata', {}).get('success', False):
            successful_results.append(r)
        else:
            failed_results.append(r)
    
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    print("\nSuccessful results:")
    for i, result in enumerate(successful_results, 1):
        name = result['resume_data'].get('candidate_name', 'Unknown')
        filename = result['metadata']['resume_filename']
        score = result['analysis_results'].get('overall_score', 0)
        print(f"  {i}. {name} ({filename}) - Score: {score}")
    
    print("\nFailed results:")
    for i, result in enumerate(failed_results, 1):
        filename = result['metadata']['resume_filename']
        error = result['metadata'].get('error', 'Unknown error')
        print(f"  {i}. {filename} - Error: {error}")

if __name__ == "__main__":
    test_batch_analysis()