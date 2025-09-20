"""
Test improved candidate name extraction
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_name_extraction():
    """Test the improved name extraction function"""
    try:
        from simple_resume_analyzer import ResumeAnalyzer
        
        analyzer = ResumeAnalyzer()
        
        # Test with sample files
        test_files = [
            "sample_data/batch_test/alex_python_senior.txt",
            "sample_data/batch_test/sarah_fullstack.txt", 
            "sample_data/batch_test/mike_java_developer.txt"
        ]
        
        print("🧪 Testing Name Extraction:")
        print("=" * 40)
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    name = analyzer._extract_candidate_name(content)
                    email = analyzer._extract_email(content)
                    phone = analyzer._extract_phone(content)
                    
                    print(f"📄 {os.path.basename(file_path)}:")
                    print(f"   Name: {name}")
                    print(f"   Email: {email}")
                    print(f"   Phone: {phone}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        # Test with some direct text samples
        print("🧪 Testing Direct Text Samples:")
        print("=" * 40)
        
        test_samples = [
            "John Smith\nSoftware Engineer\nemail: john@example.com",
            "JANE DOE\nData Scientist\nPhone: (555) 123-4567",
            "Mike Johnson Jr.\nProduct Manager\nmike.johnson@company.com\n+1-555-987-6543",
            "Sarah Williams\nEmail: sarah@email.com\nPhone: 555.123.4567"
        ]
        
        for i, sample in enumerate(test_samples, 1):
            name = analyzer._extract_candidate_name(sample)
            email = analyzer._extract_email(sample)
            phone = analyzer._extract_phone(sample)
            
            print(f"Sample {i}:")
            print(f"   Text: {repr(sample[:30])}...")
            print(f"   Name: {name}")
            print(f"   Email: {email}")
            print(f"   Phone: {phone}")
            print()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_name_extraction()