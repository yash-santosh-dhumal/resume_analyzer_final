"""
Quick Test for Improved Batch Analysis
Creates sample files to test with the web interface
"""

import os
from pathlib import Path

def create_sample_files():
    """Create sample files for testing batch analysis"""
    
    # Create sample directory
    sample_dir = Path("sample_data/batch_test")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample job description
    jd_content = """
Senior Python Developer Position
TechCorp Solutions

Requirements:
- 5+ years Python development experience
- Experience with Django/Flask frameworks
- Knowledge of PostgreSQL and MongoDB
- Familiarity with AWS cloud services
- Strong problem-solving skills
- Bachelor's degree in Computer Science

Responsibilities:
- Design and develop scalable web applications
- Lead technical architecture decisions
- Mentor junior developers
- Collaborate with product teams
"""
    
    # Sample resumes
    resumes = [
        {
            "name": "alex_python_senior.txt",
            "content": """
Alex Thompson
Senior Python Developer
Email: alex.thompson@email.com
Phone: +1-555-0123

EXPERIENCE:
Senior Python Developer at WebTech Inc (2019-2024)
- Led development of Django-based e-commerce platform
- Managed PostgreSQL databases with 10M+ records
- Deployed applications on AWS using EC2 and RDS
- Mentored team of 3 junior developers

Python Developer at StartupCo (2017-2019)
- Built RESTful APIs using Flask
- Worked with MongoDB for real-time analytics
- Implemented automated testing and CI/CD

EDUCATION:
Bachelor of Computer Science
University of Technology (2013-2017)

SKILLS:
Python, Django, Flask, PostgreSQL, MongoDB, AWS, Docker, Git, REST APIs
"""
        },
        {
            "name": "sarah_fullstack.txt", 
            "content": """
Sarah Kim
Full Stack Developer
Email: sarah.kim@email.com
Phone: +1-555-0456

EXPERIENCE:
Full Stack Developer at Digital Agency (2020-2024)
- Developed web applications using Python and React
- Experience with Django REST framework
- Used PostgreSQL for data storage
- Basic AWS deployment experience

Junior Developer at CodeCraft (2018-2020)
- Learned Python programming
- Built simple web applications
- Basic database knowledge

EDUCATION:
Bachelor of Information Technology
State College (2014-2018)

SKILLS:
Python, Django, React, PostgreSQL, HTML, CSS, JavaScript, Git
"""
        },
        {
            "name": "mike_java_developer.txt",
            "content": """
Mike Rodriguez
Java Developer
Email: mike.rodriguez@email.com
Phone: +1-555-0789

EXPERIENCE:
Java Developer at Enterprise Corp (2019-2024)
- Developed enterprise applications using Java Spring
- Worked with Oracle databases
- Experience with microservices architecture
- Used Jenkins for CI/CD

Software Engineer at TechStart (2017-2019)
- Built web services using Java
- Basic database management
- Agile development practices

EDUCATION:
Bachelor of Computer Engineering
Engineering University (2013-2017)

SKILLS:
Java, Spring Boot, Oracle, Microservices, Jenkins, Maven, Git
"""
        }
    ]
    
    # Write job description
    with open(sample_dir / "job_description.txt", "w") as f:
        f.write(jd_content)
    
    # Write resumes
    for resume in resumes:
        with open(sample_dir / resume["name"], "w") as f:
            f.write(resume["content"])
    
    print(f"✅ Sample files created in: {sample_dir}")
    print(f"📁 Files created:")
    print(f"   • job_description.txt")
    for resume in resumes:
        print(f"   • {resume['name']}")
    
    print(f"\n🎯 To test batch analysis:")
    print(f"1. Go to http://localhost:8501")
    print(f"2. Navigate to 'Batch Analysis' page")
    print(f"3. Upload job_description.txt as job description")
    print(f"4. Upload all 3 resume files (hold Ctrl and click each file)")
    print(f"5. Click 'Start Batch Analysis'")
    
    return sample_dir

if __name__ == "__main__":
    create_sample_files()