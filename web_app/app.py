"""
Streamlit Web Application for Resume Analyzer
Main user interface for students and placement team
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
import json
from pathlib import Path
import sys
import os

# Add src and project root directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import config first
try:
    from config.settings import load_config
    print("✅ Config loaded successfully")
except Exception as e:
    print(f"⚠️ Config import failed: {e}")
    # Create a fallback config function
    def load_config():
        return {
            'openrouter': {
                'api_key': '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',
                'model': 'meta-llama/llama-3.1-8b-instruct:free'
            }
        }

# Try to import the fixed analyzer, fall back to simple one if it fails
try:
    from simple_resume_analyzer import ResumeAnalyzer
    ANALYZER_TYPE = "simple_enhanced"
    print("✅ Enhanced simple analyzer loaded")
except Exception as e:
    print(f"❌ Enhanced simple analyzer failed: {e}")
    ANALYZER_TYPE = "none"

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer - Innomatics Research Labs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-fair { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None

def initialize_analyzer():
    """Initialize the resume analyzer"""
    if ANALYZER_TYPE == "none":
        st.error("❌ No working analyzer found. Please check the installation.")
        return False
        
    if st.session_state.analyzer is None:
        try:
            with st.spinner(f"Initializing Resume Analyzer ({ANALYZER_TYPE} version)..."):
                st.session_state.analyzer = ResumeAnalyzer()
            st.success(f"✅ Resume Analyzer initialized successfully! (Using {ANALYZER_TYPE} version)")
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {str(e)}")
            return False
    return True

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Resume Analyzer - Innomatics Research Labs</div>', 
                unsafe_allow_html=True)
    
    # Initialize analyzer
    if not initialize_analyzer():
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Dashboard",
            "📝 Analyze Resume", 
            "📊 Batch Analysis",
            "🔍 View Results",
            "📈 Reports & Analytics",
            "⚙️ System Status"
        ]
    )
    
    # Route to different pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📝 Analyze Resume":
        show_single_analysis()
    elif page == "📊 Batch Analysis":
        show_batch_analysis()
    elif page == "🔍 View Results":
        show_results_viewer()
    elif page == "📈 Reports & Analytics":
        show_reports_analytics()
    elif page == "⚙️ System Status":
        show_system_status()

def show_dashboard():
    """Show main dashboard with overview"""
    st.markdown("## 📊 Dashboard Overview")
    
    # Get system statistics
    try:
        stats = st.session_state.analyzer.get_system_statistics(30)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Analyses (30 days)",
                value=stats.get('total_analyses', 0)
            )
        
        with col2:
            st.metric(
                label="Resumes Processed",
                value=stats.get('total_resumes', 0)
            )
        
        with col3:
            st.metric(
                label="Job Descriptions",
                value=stats.get('total_jobs', 0)
            )
        
        with col4:
            st.metric(
                label="Average Score",
                value=f"{stats.get('average_score', 0):.1f}"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Match Level Distribution")
            match_dist = stats.get('match_level_distribution', {})
            if match_dist:
                df_match = pd.DataFrame(list(match_dist.items()), columns=['Level', 'Count'])
                fig = px.pie(df_match, values='Count', names='Level', title="Match Quality Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Hiring Recommendations")
            hiring_dist = stats.get('hiring_decision_distribution', {})
            if hiring_dist:
                df_hiring = pd.DataFrame(list(hiring_dist.items()), columns=['Decision', 'Count'])
                fig = px.bar(df_hiring, x='Decision', y='Count', title="Hiring Decision Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Analyze New Resume", use_container_width=True):
                st.switch_page("📝 Analyze Resume")
        
        with col2:
            if st.button("📊 Batch Processing", use_container_width=True):
                st.switch_page("📊 Batch Analysis")
        
        with col3:
            if st.button("🔍 View Results", use_container_width=True):
                st.switch_page("🔍 View Results")
        
    except Exception as e:
        st.error(f"Failed to load dashboard data: {str(e)}")

def show_single_analysis():
    """Show single resume analysis interface"""
    st.markdown("## 📝 Single Resume Analysis")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload Resume")
        resume_file = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'docx'],
            help="Upload PDF or DOCX resume file"
        )
    
    with col2:
        st.markdown("### Upload Job Description")
        jd_file = st.file_uploader(
            "Choose job description file",
            type=['pdf', 'docx'],
            help="Upload PDF or DOCX job description file"
        )
    
    # Analysis options
    st.markdown("### Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        save_to_db = st.checkbox("Save results to database", value=True)
    
    with col2:
        generate_report = st.checkbox("Generate detailed report", value=True)
    
    # Analyze button
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        if resume_file and jd_file:
            analyze_single_resume(resume_file, jd_file, save_to_db, generate_report)
        else:
            st.error("Please upload both resume and job description files.")

def analyze_single_resume(resume_file, jd_file, save_to_db, generate_report):
    """Analyze a single resume"""
    try:
        # Save uploaded files temporarily
        with st.spinner("Processing files..."):
            resume_path = save_uploaded_file(resume_file, "resume")
            jd_path = save_uploaded_file(jd_file, "job_description")
        
        # Perform analysis
        with st.spinner("Analyzing resume... This may take a few minutes."):
            start_time = time.time()
            results = st.session_state.analyzer.analyze_resume_for_job(
                resume_path, jd_path, save_to_db
            )
            processing_time = time.time() - start_time
        
        # Display results
        if results['metadata']['success']:
            display_analysis_results(results)
            
            if generate_report:
                display_detailed_report(results)
        else:
            st.error(f"Analysis failed: {results['metadata'].get('error', 'Unknown error')}")
        
        # Cleanup temporary files
        os.unlink(resume_path)
        os.unlink(jd_path)
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

def display_analysis_results(results):
    """Display analysis results in a formatted way"""
    st.markdown("## 🎯 Analysis Results")
    
    # Overall score
    overall_score = results['analysis_results']['overall_score']
    match_level = results['analysis_results']['match_level']
    confidence = results['analysis_results']['confidence']
    
    # Score color based on level
    if match_level == 'excellent':
        score_class = 'score-excellent'
    elif match_level == 'good':
        score_class = 'score-good'
    elif match_level == 'fair':
        score_class = 'score-fair'
    else:
        score_class = 'score-poor'
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Score",
            value=f"{overall_score:.1f}/100"
        )
    
    with col2:
        st.markdown(f"**Match Level**: <span class='{score_class}'>{match_level.upper()}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric(
            label="Confidence",
            value=f"{confidence:.1f}%"
        )
    
    # Component scores
    st.markdown("### Component Scores")
    
    hard_score = results['detailed_results']['hard_matching'].get('overall_score', 0)
    soft_score = results['detailed_results']['soft_matching'].get('combined_semantic_score', 0)
    llm_score = results['detailed_results']['llm_analysis'].get('llm_score', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hard Matching", f"{hard_score:.1f}")
        st.progress(hard_score / 100)
    
    with col2:
        st.metric("Semantic Similarity", f"{soft_score:.1f}")
        st.progress(soft_score / 100)
    
    with col3:
        st.metric("LLM Analysis", f"{llm_score:.1f}")
        st.progress(llm_score / 100)
    
    # Hiring recommendation
    st.markdown("### 🎯 Hiring Recommendation")
    hiring_rec = results['hiring_recommendation']
    
    decision_colors = {
        'HIRE': '🟢',
        'INTERVIEW': '🟡', 
        'MAYBE': '🟠',
        'REJECT': '🔴'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        decision = hiring_rec['decision']
        st.markdown(f"**Decision**: {decision_colors.get(decision, '⚪')} {decision}")
        st.markdown(f"**Confidence**: {hiring_rec['confidence']}")
        st.markdown(f"**Success Probability**: {hiring_rec['success_probability']:.1f}%")
    
    with col2:
        st.markdown("**Reasoning**:")
        st.write(hiring_rec['reasoning'])
    
    # Recommendations and feedback
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Recommendations")
        recommendations = results['analysis_results']['recommendations']
        for i, rec in enumerate(recommendations[:5], 1):
            st.write(f"{i}. {rec}")
    
    with col2:
        st.markdown("### ⚠️ Risk Factors")
        risk_factors = results['analysis_results']['risk_factors']
        for i, risk in enumerate(risk_factors[:5], 1):
            st.write(f"{i}. {risk}")

def display_detailed_report(results):
    """Display detailed analysis report"""
    st.markdown("## 📋 Detailed Report")
    
    # Expandable sections
    with st.expander("📊 Detailed Component Analysis"):
        # Hard matching details
        st.markdown("#### Hard Matching Analysis")
        hard_details = results['detailed_results']['hard_matching']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Keyword Score", f"{hard_details.get('keyword_score', 0):.1f}")
            st.metric("Skills Score", f"{hard_details.get('skills_score', 0):.1f}")
        
        with col2:
            st.metric("TF-IDF Score", f"{hard_details.get('tfidf_score', 0):.1f}")
            st.metric("BM25 Score", f"{hard_details.get('bm25_score', 0):.1f}")
        
        # Soft matching details
        st.markdown("#### Semantic Similarity Analysis")
        soft_details = results['detailed_results']['soft_matching']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic Score", f"{soft_details.get('semantic_score', 0):.1f}")
        with col2:
            st.metric("Embedding Score", f"{soft_details.get('embedding_score', 0):.1f}")
    
    with st.expander("🤖 LLM Analysis Details"):
        llm_details = results['detailed_results']['llm_analysis']
        
        st.markdown("#### Gap Analysis")
        gap_analysis = llm_details.get('gap_analysis', 'No gap analysis available')
        if isinstance(gap_analysis, dict):
            st.write(gap_analysis.get('detailed_analysis', 'No detailed analysis'))
        else:
            st.write(gap_analysis)
        
        st.markdown("#### Personalized Feedback")
        st.write(llm_details.get('personalized_feedback', 'No feedback available'))
        
        if 'strengths' in llm_details:
            st.markdown("#### Strengths")
            for strength in llm_details['strengths']:
                st.write(f"• {strength}")
        
        if 'weaknesses' in llm_details:
            st.markdown("#### Areas for Improvement")
            for weakness in llm_details['weaknesses']:
                st.write(f"• {weakness}")
    
    with st.expander("👤 Candidate Profile"):
        candidate = results['resume_data']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name**: {candidate.get('candidate_name', 'N/A')}")
            st.write(f"**Email**: {candidate.get('email', 'N/A')}")
            st.write(f"**Phone**: {candidate.get('phone', 'N/A')}")
        
        with col2:
            st.write(f"**Experience**: {candidate.get('experience_years', 'N/A')} years")
            st.write(f"**File**: {candidate.get('filename', 'N/A')}")
        
        if candidate.get('skills'):
            st.markdown("**Skills**:")
            skills_text = ", ".join(candidate['skills'][:10])
            if len(candidate['skills']) > 10:
                skills_text += f" ... and {len(candidate['skills']) - 10} more"
            st.write(skills_text)
    
    # Export options
    st.markdown("### 📤 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as JSON"):
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export as CSV"):
            # Create simplified CSV data
            csv_data = create_csv_from_results([results])
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_batch_analysis():
    """Show batch analysis interface"""
    st.markdown("## 📊 Batch Analysis")
    st.write("Analyze multiple resumes against a single job description.")
    
    # Job description upload
    st.markdown("### 1. Upload Job Description")
    jd_file = st.file_uploader(
        "Choose job description file",
        type=['pdf', 'docx'],
        key="batch_jd"
    )
    
    # Multiple resume upload
    st.markdown("### 2. Upload Resume Files")
    resume_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        key="batch_resumes"
    )
    
    if resume_files:
        st.write(f"Selected {len(resume_files)} resume files:")
        for file in resume_files:
            st.write(f"• {file.name}")
    
    # Analysis options
    st.markdown("### 3. Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        save_batch_to_db = st.checkbox("Save all results to database", value=True, key="batch_save")
    
    with col2:
        export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON"], key="batch_export")
    
    # Start batch analysis
    if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
        if jd_file and resume_files:
            run_batch_analysis(jd_file, resume_files, save_batch_to_db, export_format)
        else:
            st.error("Please upload job description and at least one resume file.")

def run_batch_analysis(jd_file, resume_files, save_to_db, export_format):
    """Run batch analysis"""
    try:
        # Save job description temporarily
        jd_path = save_uploaded_file(jd_file, "batch_jd")
        
        # Save all resume files temporarily
        resume_paths = []
        for i, resume_file in enumerate(resume_files):
            resume_path = save_uploaded_file(resume_file, f"batch_resume_{i}")
            resume_paths.append(resume_path)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run batch analysis
        results = []
        for i, resume_path in enumerate(resume_paths):
            status_text.text(f"Analyzing resume {i+1}/{len(resume_paths)}: {Path(resume_path).name}")
            
            try:
                result = st.session_state.analyzer.analyze_resume_for_job(
                    resume_path, jd_path, save_to_db
                )
                results.append(result)
            except Exception as e:
                st.error(f"Failed to analyze {Path(resume_path).name}: {str(e)}")
                results.append({
                    'metadata': {
                        'resume_filename': Path(resume_path).name,
                        'success': False,
                        'error': str(e)
                    }
                })
            
            progress_bar.progress((i + 1) / len(resume_paths))
        
        status_text.text("Analysis completed!")
        
        # Display batch results
        display_batch_results(results, export_format)
        
        # Cleanup temporary files
        os.unlink(jd_path)
        for resume_path in resume_paths:
            os.unlink(resume_path)
        
    except Exception as e:
        st.error(f"Batch analysis failed: {str(e)}")

def display_batch_results(results, export_format):
    """Display batch analysis results"""
    st.markdown("## 🎯 Batch Analysis Results")
    
    # Filter successful results
    successful_results = [r for r in results if r['metadata']['success']]
    failed_results = [r for r in results if not r['metadata']['success']]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(results))
    
    with col2:
        st.metric("Successful", len(successful_results))
    
    with col3:
        st.metric("Failed", len(failed_results))
    
    with col4:
        if successful_results:
            avg_score = sum(r['analysis_results']['overall_score'] for r in successful_results) / len(successful_results)
            st.metric("Average Score", f"{avg_score:.1f}")
    
    if successful_results:
        # Create summary table
        st.markdown("### Results Summary")
        
        summary_data = []
        for result in successful_results:
            summary_data.append({
                'Candidate': result['resume_data'].get('candidate_name', 'N/A'),
                'Filename': result['metadata']['resume_filename'],
                'Overall Score': f"{result['analysis_results']['overall_score']:.1f}",
                'Match Level': result['analysis_results']['match_level'].title(),
                'Hiring Decision': result['hiring_recommendation']['decision'],
                'Success Probability': f"{result['hiring_recommendation']['success_probability']:.1f}%"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            scores = [r['analysis_results']['overall_score'] for r in successful_results]
            fig = px.histogram(x=scores, title="Score Distribution", nbins=10)
            fig.update_layout(xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Match level distribution
            match_levels = [r['analysis_results']['match_level'] for r in successful_results]
            match_counts = pd.Series(match_levels).value_counts()
            fig = px.pie(values=match_counts.values, names=match_counts.index, title="Match Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.markdown("### 📤 Export Results")
        
        if export_format == "CSV":
            csv_data = create_csv_from_results(successful_results)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif export_format == "JSON":
            json_str = json.dumps(successful_results, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Show failed analyses if any
    if failed_results:
        st.markdown("### ❌ Failed Analyses")
        for result in failed_results:
            st.error(f"{result['metadata']['resume_filename']}: {result['metadata'].get('error', 'Unknown error')}")

def show_results_viewer():
    """Show results viewer interface"""
    st.markdown("## 🔍 View Analysis Results")
    st.write("Browse and search previous analysis results.")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
    
    with col2:
        # Score filter
        min_score = st.slider("Minimum Score", 0, 100, 0)
    
    with col3:
        # Match level filter
        match_levels = st.multiselect(
            "Match Levels",
            ['excellent', 'good', 'fair', 'poor'],
            default=['excellent', 'good', 'fair', 'poor']
        )
    
    # Search functionality would require database queries
    # This is a placeholder for the interface
    st.info("Database search functionality would be implemented here based on the filters above.")

def show_reports_analytics():
    """Show reports and analytics interface"""
    st.markdown("## 📈 Reports & Analytics")
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        [
            "System Performance Overview",
            "Job-wise Analysis Report", 
            "Candidate Performance Trends",
            "Skills Gap Analysis"
        ]
    )
    
    if report_type == "System Performance Overview":
        show_system_performance_report()
    elif report_type == "Job-wise Analysis Report":
        show_job_analysis_report()
    
    # This would be expanded with more report types

def show_system_performance_report():
    """Show system performance report"""
    try:
        stats = st.session_state.analyzer.get_system_statistics(30)
        
        st.markdown("### System Performance (Last 30 Days)")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", stats.get('total_analyses', 0))
        
        with col2:
            st.metric("Average Score", f"{stats.get('average_score', 0):.1f}")
        
        with col3:
            st.metric("Processing Success Rate", "95%")  # This would be calculated
        
        with col4:
            st.metric("Average Processing Time", "2.3s")  # This would be calculated
        
        # Performance charts would go here
        
    except Exception as e:
        st.error(f"Failed to generate report: {str(e)}")

def show_job_analysis_report():
    """Show job-wise analysis report"""
    st.markdown("### Job-wise Analysis Report")
    st.info("Select a job to view detailed analysis report")
    
    # Job selection interface would go here

def show_system_status():
    """Show system status and health check"""
    st.markdown("## ⚙️ System Status")
    
    # Health check
    if st.button("🔍 Run Health Check"):
        with st.spinner("Checking system health..."):
            health_status = st.session_state.analyzer.health_check()
        
        # Display health status
        if health_status['status'] == 'healthy':
            st.success("🟢 System is healthy")
        elif health_status['status'] == 'degraded':
            st.warning("🟡 System is degraded")
        else:
            st.error("🔴 System is unhealthy")
        
        # Component status
        st.markdown("### Component Status")
        for component, status in health_status['components'].items():
            if 'error' in str(status):
                st.error(f"{component}: {status}")
            elif status == 'healthy':
                st.success(f"{component}: ✅ Healthy")
            else:
                st.warning(f"{component}: ⚠️ {status}")

def save_uploaded_file(uploaded_file, prefix):
    """Save uploaded file temporarily and return path"""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / f"{prefix}_{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def create_csv_from_results(results):
    """Create CSV data from analysis results"""
    csv_data = []
    
    for result in results:
        if result['metadata']['success']:
            csv_data.append({
                'Candidate Name': result['resume_data'].get('candidate_name', 'N/A'),
                'Email': result['resume_data'].get('email', 'N/A'),
                'Phone': result['resume_data'].get('phone', 'N/A'),
                'Resume Filename': result['metadata']['resume_filename'],
                'Overall Score': result['analysis_results']['overall_score'],
                'Match Level': result['analysis_results']['match_level'],
                'Confidence': result['analysis_results']['confidence'],
                'Hard Matching Score': result['detailed_results']['hard_matching'].get('overall_score', 0),
                'Soft Matching Score': result['detailed_results']['soft_matching'].get('combined_semantic_score', 0),
                'LLM Analysis Score': result['detailed_results']['llm_analysis'].get('llm_score', 0),
                'Hiring Decision': result['hiring_recommendation']['decision'],
                'Success Probability': result['hiring_recommendation']['success_probability'],
                'Processing Time': result['metadata']['processing_time'],
                'Analysis Date': datetime.fromtimestamp(result['metadata']['timestamp']).isoformat()
            })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    else:
        return "No data available"

if __name__ == "__main__":
    main()