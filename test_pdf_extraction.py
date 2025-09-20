"""
Test PDF extraction and candidate info extraction from Resumes folder
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_pdf_extraction():
    """Test PDF extraction with actual resume files"""
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Test with actual PDF files from Resumes folder
        resumes_dir = Path("sample_data/resumes/Resumes")
        pdf_files = list(resumes_dir.glob("*.pdf"))[:3]  # Test first 3 files
        
        print("🧪 Testing PDF Extraction from Resumes folder:")
        print("=" * 50)
        
        for pdf_file in pdf_files:
            print(f"\n📄 Testing: {pdf_file.name}")
            print("-" * 30)
            
            try:
                # Test PDF reading
                content = analyzer._read_file_content(str(pdf_file))
                
                if content.strip():
                    print(f"✅ PDF extraction successful")
                    print(f"📄 Content length: {len(content)} characters")
                    print(f"📄 First 200 chars: {repr(content[:200])}")
                    
                    # Test candidate info extraction
                    name = analyzer._extract_candidate_name(content)
                    email = analyzer._extract_email(content)
                    phone = analyzer._extract_phone(content)
                    experience = analyzer._extract_years_experience(content)
                    skills = list(analyzer._extract_skills(analyzer._clean_text(content)))[:5]
                    
                    print(f"\n👤 Extracted Information:")
                    print(f"   Name: {name}")
                    print(f"   Email: {email}")
                    print(f"   Phone: {phone}")
                    print(f"   Experience: {experience} years")
                    print(f"   Skills (top 5): {skills}")
                    
                else:
                    print(f"❌ PDF extraction failed - no content extracted")
                    
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
        
        # Test a complete analysis
        if pdf_files:
            print(f"\n🎯 Testing Complete Analysis:")
            print("=" * 50)
            
            # Use sample job description
            jd_file = "sample_data/batch_test/job_description.txt"
            
            if os.path.exists(jd_file):
                try:
                    result = analyzer.analyze_resume_for_job(
                        str(pdf_files[0]), 
                        jd_file, 
                        save_to_db=False
                    )
                    
                    if result['metadata']['success']:
                        print(f"✅ Complete analysis successful")
                        print(f"📊 Overall Score: {result['analysis_results']['overall_score']:.1f}")
                        print(f"👤 Candidate: {result['resume_data']['candidate_name']}")
                        print(f"📧 Email: {result['resume_data']['email']}")
                        print(f"📞 Phone: {result['resume_data']['phone']}")
                        print(f"💼 Experience: {result['resume_data']['experience_years']} years")
                        print(f"🎯 Skills: {result['resume_data']['skills'][:5]}")
                    else:
                        print(f"❌ Complete analysis failed")
                        
                except Exception as e:
                    print(f"❌ Complete analysis error: {e}")
            else:
                print(f"⚠️ Job description file not found: {jd_file}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_pdf_extraction()