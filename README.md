# Resume Analyzer - Innomatics Research Labs

**🎯 AI-Powered Resume Relevance Analysis System**

A comprehensive, production-ready system for automated resume analysis and relevance scoring against job descriptions. Built for Innomatics Research Labs to streamline the placement process for students and hiring teams.

## 🌟 Features

### Core Functionality
- **Multi-format Document Support**: PDF and DOCX resume/JD processing
- **Advanced Text Extraction**: PyMuPDF, pdfplumber, python-docx integration
- **Intelligent Text Normalization**: spaCy and NLTK-powered preprocessing
- **Hybrid Matching Algorithm**: 
  - Hard matching (keywords, skills, TF-IDF, BM25)
  - Soft matching (semantic similarity, embeddings)
  - LLM-powered reasoning and gap analysis

### Analysis Components
- **Hard Matching (40% weight)**: Keyword extraction, skill matching, TF-IDF/BM25 scoring
- **Soft Matching (30% weight)**: Sentence transformers, vector similarity, semantic analysis
- **LLM Analysis (30% weight)**: GPT/LLaMA-powered gap analysis, personalized feedback, verdict generation

### Scoring & Recommendations
- **0-100 Relevance Scoring**: Weighted combination of all analysis components
- **Match Levels**: Excellent (80+), Good (65+), Fair (45+), Poor (<45)
- **Hiring Recommendations**: HIRE, INTERVIEW, MAYBE, REJECT with confidence levels
- **Detailed Feedback**: Gap analysis, improvement suggestions, risk factors

### Web Interface
- **Student Portal**: Upload resume, view analysis results, get personalized feedback
- **Placement Team Dashboard**: Batch processing, candidate ranking, export functionality
- **Analytics & Reporting**: Performance metrics, hiring insights, system statistics

### Data Management
- **Database Storage**: SQLite/PostgreSQL support with complete audit trails
- **Export Functionality**: CSV, Excel, JSON export with filtering capabilities
- **Batch Processing**: Analyze multiple resumes against single job descriptions
- **Historical Analysis**: Track performance trends and hiring outcomes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for LLM models)
- OpenAI API key (optional, for enhanced LLM features)

### Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd resume_analyzer
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLP Models**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. **Configure Settings**
```bash
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml with your API keys and preferences
```

5. **Initialize Database**
```bash
python -c "from src.database import DatabaseManager; from src.config.settings import load_config; DatabaseManager(load_config())"
```

### Running the Application

#### Simple Demo (Recommended for Testing)
```bash
streamlit run simple_app.py
```

#### Full Web Application
```bash
streamlit run web_app/app.py
```

#### Programmatic Usage
```python
from src import ResumeAnalyzer

# Initialize analyzer
analyzer = ResumeAnalyzer()

# Analyze single resume
results = analyzer.analyze_resume_for_job(
    'path/to/resume.pdf',
    'path/to/job_description.pdf'
)

print(f"Overall Score: {results['analysis_results']['overall_score']}")
print(f"Hiring Decision: {results['hiring_recommendation']['decision']}")
```

## 📁 Project Structure

```
resume_analyzer/
├── src/                          # Core application modules
│   ├── parsers/                  # Document parsing and text extraction
│   │   ├── document_parser.py    # PDF/DOCX parsing
│   │   └── text_normalizer.py    # Text preprocessing and normalization
│   ├── matching/                 # Matching algorithms
│   │   ├── hard_matcher.py       # Keyword and skill matching
│   │   ├── soft_matcher.py       # Semantic similarity matching
│   │   ├── embedding_generator.py # Vector embeddings
│   │   └── vector_database.py    # Vector storage and retrieval
│   ├── llm/                      # LLM integration
│   │   ├── llm_client.py         # Multi-provider LLM client
│   │   ├── langchain_analyzer.py # LangChain-based analysis
│   │   └── reasoning_engine.py   # Comprehensive LLM reasoning
│   ├── scoring/                  # Scoring and evaluation
│   │   └── scoring_engine.py     # Weighted scoring system
│   ├── database/                 # Data persistence
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── database_manager.py   # Database operations
│   │   └── export_manager.py     # Export functionality
│   ├── config/                   # Configuration management
│   │   └── settings.py           # Settings and configuration
│   └── resume_analyzer.py        # Main application orchestrator
├── web_app/                      # Streamlit web interface
│   └── app.py                    # Full-featured web application
├── data/                         # Data storage
│   ├── sample_resumes/           # Sample resume files (10 PDFs)
│   └── sample_jds/               # Sample job descriptions (2 PDFs)
├── config/                       # Configuration files
│   └── settings.yaml.example     # Example configuration
├── simple_app.py                 # Simple demo application
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Multi-container setup
└── README.md                     # This file
```

## ⚙️ Configuration

### settings.yaml Configuration Options

```yaml
# API Keys
api_keys:
  openai: "your-openai-api-key"
  huggingface: "your-hf-token"

# LLM Configuration
llm:
  primary_provider: "openai"
  fallback_provider: "huggingface"
  model_name: "gpt-3.5-turbo"
  max_tokens: 1000
  temperature: 0.3

# Scoring Weights
scoring:
  weights:
    hard_matching: 0.40
    soft_matching: 0.30
    llm_analysis: 0.30
  thresholds:
    excellent: 80.0
    good: 65.0
    fair: 45.0

# Database Configuration
database:
  type: "sqlite"  # or "postgresql"
  path: "data/resume_analyzer.db"
  # For PostgreSQL:
  # host: "localhost"
  # port: 5432
  # database: "resume_analyzer"
  # username: "postgres"
  # password: "password"

# Vector Database
vector_db:
  provider: "chroma"  # or "faiss"
  persist_directory: "data/vector_db"

# Text Processing
text_processing:
  spacy_model: "en_core_web_sm"
  max_text_length: 50000
```

## 📊 Analysis Workflow

1. **Document Parsing**
   - Extract text from PDF/DOCX files
   - Normalize and preprocess text
   - Extract candidate information and job requirements

2. **Hard Matching Analysis**
   - Keyword extraction and matching
   - Skills identification and scoring
   - TF-IDF and BM25 similarity calculation
   - Fuzzy string matching for variations

3. **Soft Matching Analysis**
   - Generate semantic embeddings using Sentence Transformers
   - Store and query vector databases (Chroma/FAISS)
   - Calculate contextual similarity scores

4. **LLM-Powered Analysis**
   - Gap analysis and missing skills identification
   - Personalized feedback generation
   - Verdict reasoning and explanation
   - Improvement recommendations

5. **Final Scoring**
   - Weighted combination of all analysis components
   - Confidence calculation based on score consistency
   - Match level assignment (Excellent/Good/Fair/Poor)

6. **Hiring Recommendation**
   - Decision generation (HIRE/INTERVIEW/MAYBE/REJECT)
   - Success probability estimation
   - Risk factor identification
   - Next steps recommendation

## 🔧 API Usage

### Single Resume Analysis
```python
from src import ResumeAnalyzer

analyzer = ResumeAnalyzer()

# Analyze resume against job description
results = analyzer.analyze_resume_for_job(
    resume_file_path="resume.pdf",
    job_description_file_path="job_description.pdf",
    save_to_db=True
)

# Access results
overall_score = results['analysis_results']['overall_score']
match_level = results['analysis_results']['match_level']
hiring_decision = results['hiring_recommendation']['decision']
recommendations = results['analysis_results']['recommendations']
```

### Batch Processing
```python
# Analyze multiple resumes against one job description
resume_files = ['resume1.pdf', 'resume2.pdf', 'resume3.pdf']
jd_file = 'job_description.pdf'

batch_results = analyzer.analyze_multiple_resumes(resume_files, jd_file)

# Rank candidates by score
sorted_results = sorted(batch_results, 
                       key=lambda x: x['analysis_results']['overall_score'], 
                       reverse=True)
```

### Export Results
```python
# Export to CSV
analyzer.export_results('csv', 'results.csv', job_id=1)

# Export to Excel with filters
analyzer.export_results('excel', 'results.xlsx', 
                        start_date=datetime(2024, 1, 1),
                        end_date=datetime(2024, 12, 31))

# Generate candidate report
report = analyzer.create_candidate_report(analysis_id=123)
```

## 📈 Web Interface Features

### Student Portal
- **Resume Upload**: Drag-and-drop interface for PDF/DOCX files
- **Job Matching**: Upload job descriptions for targeted analysis
- **Results Dashboard**: Comprehensive score breakdown and explanations
- **Personalized Feedback**: AI-generated improvement suggestions
- **Progress Tracking**: Historical analysis and improvement trends

### Placement Team Dashboard
- **Batch Processing**: Analyze multiple candidates simultaneously
- **Candidate Ranking**: Sort and filter by scores and criteria
- **Hiring Pipeline**: Track candidates through recruitment stages
- **Analytics & Reports**: Performance metrics and hiring insights
- **Export Tools**: Generate reports in multiple formats

### System Administration
- **Health Monitoring**: Component status and performance metrics
- **Configuration Management**: Adjust scoring weights and thresholds
- **Audit Trails**: Complete operation logging and tracking
- **Data Management**: Backup, cleanup, and maintenance tools

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t resume-analyzer .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data resume-analyzer
```

### Multi-Container with PostgreSQL
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test with Sample Data
```bash
# Test with provided sample files
python -c "
from src import ResumeAnalyzer
analyzer = ResumeAnalyzer()
results = analyzer.analyze_resume_for_job(
    'data/sample_resumes/resume_1.pdf',
    'data/sample_jds/software_engineer_jd.pdf'
)
print(f'Score: {results[\"analysis_results\"][\"overall_score\"]}')
"
```

## 📋 Sample Output

### Analysis Results Structure
```json
{
  "metadata": {
    "analysis_id": 123,
    "processing_time": 2.3,
    "timestamp": 1703123456.789,
    "success": true
  },
  "analysis_results": {
    "overall_score": 78.5,
    "match_level": "good",
    "confidence": 85.2,
    "explanation": "Candidate demonstrates strong alignment...",
    "recommendations": [
      "Strong technical background aligns well with requirements",
      "Consider highlighting project management experience"
    ],
    "risk_factors": [
      "Limited experience with cloud platforms"
    ]
  },
  "detailed_results": {
    "hard_matching": {
      "overall_score": 72.3,
      "keyword_score": 68.5,
      "skills_score": 76.1
    },
    "soft_matching": {
      "combined_semantic_score": 81.7,
      "semantic_score": 79.2,
      "embedding_score": 84.1
    },
    "llm_analysis": {
      "llm_score": 82.0,
      "llm_verdict": "good",
      "gap_analysis": "Candidate shows strong technical foundation...",
      "personalized_feedback": "Your background in software development..."
    }
  },
  "hiring_recommendation": {
    "decision": "INTERVIEW",
    "confidence": "medium",
    "success_probability": 78.5,
    "reasoning": "Candidate shows strong technical alignment...",
    "next_steps": [
      "Schedule technical interview",
      "Focus on cloud platform experience"
    ]
  }
}
```

## 🚀 Performance & Scalability

### Processing Performance
- **Single Resume Analysis**: ~2-5 seconds average
- **Batch Processing**: Scales linearly with parallel processing
- **Memory Usage**: ~1-2GB for full feature set
- **Storage**: ~10MB per 1000 analyses (excluding documents)

### Scalability Features
- **Database Support**: SQLite for development, PostgreSQL for production
- **Vector Database**: Chroma for development, FAISS for production scale
- **API Rate Limiting**: Built-in protection for external API calls
- **Caching**: Intelligent caching for repeated analyses
- **Async Processing**: Background job processing for batch operations

## 🔒 Security & Privacy

### Data Protection
- **Document Security**: Temporary file handling with automatic cleanup
- **Database Encryption**: Configurable encryption for sensitive data
- **API Key Management**: Secure configuration file handling
- **Audit Logging**: Complete operation tracking for compliance

### Privacy Features
- **Data Anonymization**: Option to remove personal information
- **Retention Policies**: Configurable data cleanup and archival
- **Access Controls**: Role-based access for different user types
- **GDPR Compliance**: Data deletion and export capabilities

## 🤝 Support & Maintenance

### Monitoring
- **Health Checks**: System component status monitoring
- **Performance Metrics**: Response time and accuracy tracking
- **Error Logging**: Comprehensive error reporting and alerting
- **Usage Analytics**: System utilization and user behavior insights

### Maintenance Tasks
- **Database Cleanup**: Automated old data archival
- **Model Updates**: Regular NLP model and embedding updates
- **Configuration Tuning**: Score threshold and weight optimization
- **Security Updates**: Regular dependency and security patches

## 📞 Support

For technical support, feature requests, or bug reports:

- **Email**: support@innomatics.in
- **Documentation**: [Internal Wiki/Confluence]
- **Issue Tracking**: [Internal JIRA/GitHub Issues]

## 📄 License

Internal use only - Innomatics Research Labs
Copyright © 2024 Innomatics Research Labs. All rights reserved.

---

**Built with ❤️ for Innomatics Research Labs**
*Empowering students and streamlining recruitment through AI*