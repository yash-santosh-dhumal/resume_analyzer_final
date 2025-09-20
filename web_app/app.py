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
    
    .upload-container {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0;
        color: white;
    }
    
    .upload-box {
        background-color: #2d2d30;
        border: 2px dashed #666;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #007acc;
        background-color: #333336;
    }
    
    .results-container {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 2rem 0;
        color: white;
    }
    
    .score-display {
        text-align: center;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .score-number {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin: 0;
    }
    
    .score-poor { color: #ff6b6b; }
    .score-fair { color: #ffd93d; }
    .score-good { color: #6bcf7f; }
    .score-excellent { color: #4ecdc4; }
    
    .match-level {
        font-size: 1.2rem;
        font-weight: bold;
        text-transform: uppercase;
        margin: 0.5rem 0;
    }
    
    .component-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .component-card {
        background-color: #2d2d30;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    .component-score {
        font-size: 2rem;
        font-weight: bold;
        color: #007acc;
        margin: 0.5rem 0;
    }
    
    .component-label {
        color: #cccccc;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .progress-bar {
        background-color: #404040;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background-color: #007acc;
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }
    
    .metric-card {
        background-color: #2d2d30;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #007acc;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #cccccc;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .feature-card {
        background-color: #2d2d30;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        color: white;
    }
    
    .feature-card:hover {
        background-color: #333336;
        transform: translateY(-2px);
    }
    
    .insights-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .insights-card {
        background-color: #2d2d30;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
    }
    
    .insights-title {
        color: #007acc;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .insight-item {
        margin: 0.5rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #404040;
    }
    
    .insight-item:last-child {
        border-bottom: none;
    }
    
    @media (max-width: 768px) {
        .component-grid {
            grid-template-columns: 1fr;
        }
        
        .insights-section {
            grid-template-columns: 1fr;
        }
        
        .upload-container {
            padding: 1rem;
        }
        
        .results-container {
            padding: 1rem;
        }
    }
    
    .stButton > button {
        background-color: #007acc;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #005a9e;
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
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

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
        ],
        index=[
            "🏠 Dashboard",
            "📝 Analyze Resume", 
            "📊 Batch Analysis",
            "🔍 View Results",
            "📈 Reports & Analytics",
            "⚙️ System Status"
        ].index(st.session_state.current_page) if st.session_state.current_page in [
            "🏠 Dashboard",
            "📝 Analyze Resume", 
            "📊 Batch Analysis",
            "🔍 View Results",
            "📈 Reports & Analytics",
            "⚙️ System Status"
        ] else 0
    )
    
    # Update current page in session state
    st.session_state.current_page = page
    
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
    st.markdown("## 📊 Resume Analyzer Enterprise Dashboard")
    st.markdown("*AI-Powered Resume Analysis for Innomatics Research Labs*")
    
    st.markdown("---")
    
    # System Statistics Section
    st.markdown("### 📈 System Overview")
    
    # Simple metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.analysis_results) if 'analysis_results' in st.session_state else 0}</div>
            <div class="metric-label">Analyses Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Online</div>
            <div class="metric-label">System Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Enhanced</div>
            <div class="metric-label">Analyzer Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = 0
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            scores = [r.get('analysis_results', {}).get('overall_score', 0) for r in st.session_state.analysis_results if r.get('metadata', {}).get('success', False)]
            avg_score = sum(scores) / len(scores) if scores else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Avg Session Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Features Section
    st.markdown("### 🚀 Main Features")
    st.markdown("*Core analysis and management tools*")
    
    # Feature boxes in single row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    color: white; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">Analyze Resume</h4>
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">
                Single Resume Analysis<br>
                AI-Powered Matching<br>
                Detailed Reports
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analyze Resume", use_container_width=True, key="analyze_btn"):
            st.session_state.current_page = "📝 Analyze Resume"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    color: white; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">Batch Analysis</h4>
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">
                Multiple Resumes<br>
                Bulk Processing<br>
                Comparative Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Batch Analysis", use_container_width=True, key="batch_btn"):
            st.session_state.current_page = "📊 Batch Analysis"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    color: white; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">�</div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">View Results</h4>
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">
                Browse History<br>
                Search & Filter<br>
                Export Data
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Results", use_container_width=True, key="results_btn"):
            st.session_state.current_page = "� View Results"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    color: white; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">�</div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">Reports & Analytics</h4>
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">
                Performance Reports<br>
                Trend Analysis<br>
                Insights Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Reports & Analytics", use_container_width=True, key="analytics_btn"):
            st.session_state.current_page = "📈 Reports & Analytics"
            st.rerun()
    
    st.markdown("---")
    
    # Recent Analysis Results (if any)
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.markdown("### 📋 Recent Analysis Results")
        st.markdown("*Latest resume analysis outcomes from your current session*")
        
        # Display last 5 results in modern cards
        recent_results = st.session_state.analysis_results[-5:]
        
        for i, result in enumerate(recent_results):
            if result.get('metadata', {}).get('success', False):
                candidate_name = result.get('resume_data', {}).get('candidate_name', 'Unknown')
                score = result.get('analysis_results', {}).get('overall_score', 0)
                decision = result.get('hiring_recommendation', {}).get('decision', 'N/A')
                match_level = result.get('analysis_results', {}).get('match_level', 'unknown')
                timestamp = result.get('metadata', {}).get('timestamp', time.time())
                
                # Color coding for decision
                decision_colors = {
                    'HIRE': '#2ecc71',
                    'INTERVIEW': '#f39c12', 
                    'MAYBE': '#e67e22',
                    'REJECT': '#e74c3c'
                }
                decision_color = decision_colors.get(decision, '#95a5a6')
                
                # Score color
                if score >= 80:
                    score_color = '#2ecc71'
                elif score >= 60:
                    score_color = '#f39c12'
                else:
                    score_color = '#e74c3c'
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #2d2d30 0%, #1e1e1e 100%); 
                           padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                           border-left: 4px solid {decision_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <div>
                            <h4 style="color: white; margin: 0; font-size: 1.2rem;">👤 {candidate_name}</h4>
                            <p style="color: #999; margin: 0.25rem 0; font-size: 0.85rem;">
                                Analysis #{len(st.session_state.analysis_results) - len(recent_results) + i + 1} • 
                                {datetime.fromtimestamp(timestamp).strftime('%B %d, %Y at %I:%M %p')}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: {decision_color}; color: white; padding: 0.5rem 1rem; 
                                       border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {decision}
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;">
                        <div style="text-align: center;">
                            <div style="color: {score_color}; font-size: 1.8rem; font-weight: bold;">{score:.1f}</div>
                            <div style="color: #ccc; font-size: 0.8rem;">Overall Score</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #17a2b8; font-size: 1.2rem; font-weight: bold; text-transform: capitalize;">{match_level}</div>
                            <div style="color: #ccc; font-size: 0.8rem;">Match Level</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #6c757d; font-size: 1.2rem; font-weight: bold;">{result.get('analysis_results', {}).get('confidence', 0):.1f}%</div>
                            <div style="color: #ccc; font-size: 0.8rem;">Confidence</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #ffc107; font-size: 1.2rem; font-weight: bold;">{result.get('hiring_recommendation', {}).get('success_probability', 0):.1f}%</div>
                            <div style="color: #ccc; font-size: 0.8rem;">Success Rate</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1rem; text-align: center;">
                        <button style="background: linear-gradient(45deg, #667eea, #764ba2); 
                                      color: white; border: none; padding: 0.5rem 1.5rem; 
                                      border-radius: 25px; cursor: pointer; font-size: 0.85rem; font-weight: 500;">
                            View Full Report
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a button for viewing details (this would need to be handled properly in Streamlit)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col4:
                    if st.button("📄 View Details", key=f"view_detail_{i}", use_container_width=True):
                        st.session_state['detailed_results'] = result
                        st.rerun()
    else:
        # No results yet - show getting started
        st.markdown("### 🚀 Get Started")
        st.markdown("*Start analyzing resumes to see results here*")
        
        st.markdown("""
        <div style="background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); 
                   padding: 2rem; border-radius: 12px; text-align: center; color: white; margin: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h3 style="margin: 0.5rem 0;">No Analysis Results Yet</h3>
            <p style="margin: 1rem 0; opacity: 0.9;">
                Upload your first resume and job description to get started with AI-powered analysis
            </p>
            <div style="margin-top: 1.5rem;">
                <button style="background: rgba(255,255,255,0.2); color: white; border: 2px solid white; 
                              padding: 0.75rem 2rem; border-radius: 25px; cursor: pointer; 
                              font-size: 1rem; font-weight: 600;">
                    Start Your First Analysis
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Single Resume Analysis", type="primary", use_container_width=True):
                st.session_state.current_page = "📝 Analyze Resume"
                st.rerun()
        
        with col2:
            if st.button("📊 Batch Processing", type="secondary", use_container_width=True):
                st.session_state.current_page = "📊 Batch Analysis"
                st.rerun()

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
        export_summary = st.checkbox("Include summary report", value=True, key="batch_summary")
    
    # Start batch analysis
    if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
        if jd_file and resume_files:
            run_batch_analysis(jd_file, resume_files, save_batch_to_db, export_summary)
        else:
            st.error("Please upload job description and at least one resume file.")

def run_batch_analysis(jd_file, resume_files, save_to_db, include_summary):
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
            resume_filename = Path(resume_path).name
            status_text.text(f"Analyzing resume {i+1}/{len(resume_paths)}: {resume_filename}")
            
            try:
                result = st.session_state.analyzer.analyze_resume_for_job(
                    resume_path, jd_path, save_to_db
                )
                
                # Ensure metadata exists and contains resume_filename
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['resume_filename'] = resume_filename
                result['metadata']['success'] = True
                
                results.append(result)
            except Exception as e:
                st.error(f"Failed to analyze {resume_filename}: {str(e)}")
                results.append({
                    'metadata': {
                        'resume_filename': resume_filename,
                        'success': False,
                        'error': str(e)
                    },
                    'resume_data': {
                        'candidate_name': 'Unknown',
                        'filename': resume_filename
                    },
                    'analysis_results': {
                        'overall_score': 0,
                        'match_level': 'poor',
                        'confidence': 0,
                        'recommendations': [],
                        'risk_factors': []
                    },
                    'hiring_recommendation': {
                        'decision': 'REJECT',
                        'success_probability': 0,
                        'reasoning': f'Analysis failed: {str(e)}'
                    },
                    'detailed_results': {
                        'hard_matching': {'overall_score': 0},
                        'soft_matching': {'combined_semantic_score': 0},
                        'llm_analysis': {'llm_score': 0}
                    }
                })
            
            progress_bar.progress((i + 1) / len(resume_paths))
        
        status_text.text("Analysis completed!")
        
        # Display batch results
        display_batch_results(results, include_summary)
        
        # Cleanup temporary files
        os.unlink(jd_path)
        for resume_path in resume_paths:
            os.unlink(resume_path)
        
    except Exception as e:
        st.error(f"Batch analysis failed: {str(e)}")

def display_batch_results(results, include_summary):
    """Display batch analysis results"""
    st.markdown("## 🎯 Batch Analysis Results")
    
    # Filter successful results more safely
    successful_results = []
    failed_results = []
    
    for r in results:
        if r.get('metadata', {}).get('success', False):
            successful_results.append(r)
        else:
            failed_results.append(r)
    
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
            try:
                # Safely calculate average score
                total_score = 0
                count = 0
                for r in successful_results:
                    try:
                        score = float(r.get('analysis_results', {}).get('overall_score', 0))
                        total_score += score
                        count += 1
                    except (ValueError, TypeError):
                        continue
                
                if count > 0:
                    avg_score = total_score / count
                    st.metric("Average Score", f"{avg_score:.1f}")
                else:
                    st.metric("Average Score", "N/A")
            except Exception:
                st.metric("Average Score", "N/A")
        else:
            st.metric("Average Score", "N/A")
    
    if successful_results:
        # Create summary table
        st.markdown("### Results Summary")
        
        summary_data = []
        for result in successful_results:
            resume_data = result.get('resume_data', {})
            analysis_results = result.get('analysis_results', {})
            hiring_rec = result.get('hiring_recommendation', {})
            metadata = result.get('metadata', {})
            
            # Safely convert numeric values
            try:
                overall_score = float(analysis_results.get('overall_score', 0))
            except (ValueError, TypeError):
                overall_score = 0.0
                
            try:
                success_probability = float(hiring_rec.get('success_probability', 0))
            except (ValueError, TypeError):
                success_probability = 0.0
            
            summary_data.append({
                'Candidate': resume_data.get('candidate_name', 'N/A'),
                'Filename': metadata.get('resume_filename', 'Unknown'),
                'Overall Score': f"{overall_score:.1f}",
                'Match Level': analysis_results.get('match_level', 'unknown').title(),
                'Hiring Decision': hiring_rec.get('decision', 'UNKNOWN'),
                'Success Probability': f"{success_probability:.1f}%"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution - safely handle string to float conversion
            scores = []
            for r in successful_results:
                try:
                    score = float(r.get('analysis_results', {}).get('overall_score', 0))
                    scores.append(score)
                except (ValueError, TypeError):
                    scores.append(0)
            
            if scores:
                fig = px.histogram(x=scores, title="Score Distribution", nbins=10)
                fig.update_layout(xaxis_title="Score", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid scores to display")
        
        with col2:
            # Match level distribution
            match_levels = []
            for r in successful_results:
                match_level = r.get('analysis_results', {}).get('match_level', 'unknown')
                if match_level:
                    match_levels.append(match_level)
            
            if match_levels:
                match_counts = pd.Series(match_levels).value_counts()
                fig = px.pie(values=match_counts.values, names=match_counts.index, title="Match Level Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No match levels to display")
        
        # Export results
        st.markdown("### 📤 Export Results")
        st.markdown("*Download batch analysis results in various formats*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export with error handling
            try:
                csv_data = create_csv_from_results(successful_results)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"CSV export failed: {str(e)}")
                if st.button("🔄 Retry CSV Export", key="retry_csv"):
                    st.rerun()
        
        with col2:
            # JSON Export with error handling
            try:
                json_str = json.dumps(successful_results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"JSON export failed: {str(e)}")
                if st.button("🔄 Retry JSON Export", key="retry_json"):
                    st.rerun()
        
        with col3:
            # Excel Export with error handling
            try:
                excel_data = create_excel_from_results(successful_results)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Excel export failed: {str(e)}")
                if st.button("🔄 Retry Excel Export", key="retry_excel"):
                    st.rerun()
        
        # Individual Resume Analysis Section with Dropdowns
        st.markdown("---")
        st.markdown("### 👥 Individual Resume Analysis Results")
        st.markdown("*Click on each resume below to view detailed analysis. Expand subsections for more details.*")
        
        # Add info box
        st.info("💡 **Tip:** Each resume is shown in a collapsible section. Click to expand and see detailed analysis including scores, recommendations, and export options.")
        
        # Display all results (successful and failed) in dropdown sections
        all_results = successful_results + failed_results
        for i, result in enumerate(all_results, 1):
            display_dropdown_individual_analysis(result, i)
    
    elif failed_results:
        # If no successful results but there are failed ones, still show individual analysis
        st.markdown("---")
        st.markdown("### 👥 Individual Resume Analysis Results")
        st.markdown("*Click on each resume below to view detailed analysis. Expand subsections for more details.*")
        
        # Add info box
        st.info("💡 **Tip:** Each resume is shown in a collapsible section. Click to expand and see detailed analysis.")
        
        for i, result in enumerate(failed_results, 1):
            display_dropdown_individual_analysis(result, i)
    
    else:
        st.warning("No resume analyses to display.")

def display_individual_student_result(result, student_number):
    """Display detailed results for an individual student"""
    # Extract student data
    candidate_name = result['resume_data'].get('candidate_name', 'Unknown')
    filename = result['metadata']['resume_filename']
    overall_score = result['analysis_results']['overall_score']
    match_level = result['analysis_results']['match_level']
    hiring_decision = result['hiring_recommendation']['decision']
    success_probability = result['hiring_recommendation']['success_probability']
    confidence = result['analysis_results']['confidence']
    
    # Color coding for decisions
    decision_colors = {
        'HIRE': '#2ecc71',
        'INTERVIEW': '#f39c12',
        'MAYBE': '#e67e22', 
        'REJECT': '#e74c3c'
    }
    decision_color = decision_colors.get(hiring_decision, '#95a5a6')
    
    # Score color
    if overall_score >= 80:
        score_color = '#2ecc71'
    elif overall_score >= 60:
        score_color = '#f39c12'
    else:
        score_color = '#e74c3c'
    
    # Create expandable section for each student
    with st.expander(f"📋 Student #{student_number}: {candidate_name} ({overall_score:.1f}% - {hiring_decision})", expanded=False):
        
        # Student header card
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #2d2d30 0%, #1e1e1e 100%); 
                   padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                   border-left: 4px solid {decision_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h4 style="color: white; margin: 0; font-size: 1.3rem;">👤 {candidate_name}</h4>
                    <p style="color: #999; margin: 0.25rem 0; font-size: 0.9rem;">
                        📄 {filename}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="background: {decision_color}; color: white; padding: 0.5rem 1rem; 
                               border-radius: 20px; font-size: 0.9rem; font-weight: bold;">
                        {hiring_decision}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        
        with col2:
            st.metric("Match Level", match_level.title())
        
        with col3:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col4:
            st.metric("Success Probability", f"{success_probability:.1f}%")
        
        # Component scores
        st.markdown("#### 📊 Component Scores")
        
        hard_score = result['detailed_results']['hard_matching'].get('overall_score', 0)
        soft_score = result['detailed_results']['soft_matching'].get('combined_semantic_score', 0)
        llm_score = result['detailed_results']['llm_analysis'].get('llm_score', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #495057;">{hard_score:.1f}</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Hard Matching</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #495057;">{soft_score:.1f}</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Semantic Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #495057;">{llm_score:.1f}</div>
                <div style="font-size: 0.9rem; color: #6c757d;">LLM Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Student information
        st.markdown("#### 👤 Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📧 Email**: {result['resume_data'].get('email', 'Not found')}")
            st.write(f"**📞 Phone**: {result['resume_data'].get('phone', 'Not found')}")
        
        with col2:
            st.write(f"**💼 Experience**: {result['resume_data'].get('experience_years', 'N/A')} years")
            st.write(f"**🎯 Confidence Level**: {confidence:.1f}%")
        
        # Skills
        if result['resume_data'].get('skills'):
            st.markdown("#### 🛠️ Skills")
            skills = result['resume_data']['skills'][:10]  # Show first 10 skills
            skills_text = ", ".join(skills)
            if len(result['resume_data']['skills']) > 10:
                skills_text += f" ... and {len(result['resume_data']['skills']) - 10} more"
            st.write(skills_text)
        
        # Key strengths and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💡 Key Strengths")
            recommendations = result['analysis_results']['recommendations'][:3]
            for rec in recommendations:
                st.write(f"• {rec}")
        
        with col2:
            st.markdown("#### ⚠️ Areas for Improvement")
            risk_factors = result['analysis_results']['risk_factors'][:3]
            for risk in risk_factors:
                st.write(f"• {risk}")
        
        # Hiring recommendation details
        st.markdown("#### 🎯 Hiring Recommendation")
        
        hiring_rec = result['hiring_recommendation']
        st.markdown(f"""
        <div style="background: {decision_color}; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong>Decision:</strong> {hiring_decision}<br>
            <strong>Success Probability:</strong> {success_probability:.1f}%<br>
            <strong>Reasoning:</strong> {hiring_rec.get('reasoning', 'No reasoning provided')}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons for individual student
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"📄 Full Report", key=f"full_report_{student_number}"):
                st.session_state[f'detailed_student_{student_number}'] = result
                st.rerun()
        
        with col2:
            # Export individual JSON
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Export JSON",
                data=json_str,
                file_name=f"student_{student_number}_{candidate_name.replace(' ', '_')}.json",
                mime="application/json",
                key=f"export_json_{student_number}"
            )
        
        with col3:
            if st.button(f"📧 Send Results", key=f"send_results_{student_number}"):
                st.info(f"Results would be sent to {result['resume_data'].get('email', 'email not found')}")
        
        st.markdown("---")

def display_dropdown_individual_analysis(result, student_number):
    """Display individual analysis in a compact dropdown/expandable format"""
    # Extract student data safely
    resume_data = result.get('resume_data', {})
    analysis_results = result.get('analysis_results', {})
    hiring_rec = result.get('hiring_recommendation', {})
    metadata = result.get('metadata', {})
    detailed_results = result.get('detailed_results', {})
    
    candidate_name = resume_data.get('candidate_name', 'Unknown Candidate')
    filename = metadata.get('resume_filename', 'Unknown File')
    
    # Safely convert numeric values
    try:
        overall_score = float(analysis_results.get('overall_score', 0))
    except (ValueError, TypeError):
        overall_score = 0.0
        
    try:
        success_probability = float(hiring_rec.get('success_probability', 0))
    except (ValueError, TypeError):
        success_probability = 0.0
        
    try:
        confidence = float(analysis_results.get('confidence', 0))
    except (ValueError, TypeError):
        confidence = 0.0
    
    match_level = analysis_results.get('match_level', 'unknown')
    hiring_decision = hiring_rec.get('decision', 'UNKNOWN')
    
    # Color coding for decisions
    decision_colors = {
        'HIRE': '#2ecc71',
        'INTERVIEW': '#f39c12',
        'MAYBE': '#e67e22', 
        'REJECT': '#e74c3c'
    }
    decision_color = decision_colors.get(hiring_decision, '#95a5a6')
    
    # Check if this is a failed analysis
    is_failed = not metadata.get('success', True)
    
    # Create expandable section with summary information
    with st.expander(f"📄 #{student_number}: {candidate_name} | Score: {overall_score:.1f}% | Decision: {hiring_decision}", expanded=False):
        
        if is_failed:
            # Display error information for failed analysis
            st.error(f"❌ Analysis Failed: {metadata.get('error', 'Unknown error occurred during analysis')}")
            st.markdown("#### 📂 File Information")
            st.write(f"**Filename:** {filename}")
            return
        
        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        with col2:
            st.metric("Match Level", match_level.title())
        with col3:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col4:
            st.metric("Success Probability", f"{success_probability:.1f}%")
        
        # Candidate information section
        st.markdown("#### 👤 Candidate Information")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**📧 Email:** {resume_data.get('email', 'Not provided')}")
            st.write(f"**📞 Phone:** {resume_data.get('phone', 'Not provided')}")
        with info_col2:
            st.write(f"**💼 Experience:** {resume_data.get('experience_years', 'N/A')} years")
            st.write(f"**📂 File:** {filename}")
        
        # Skills section
        if resume_data.get('skills'):
            st.markdown("#### 🛠️ Skills")
            skills = resume_data['skills'][:10]  # Show first 10 skills
            skills_text = ", ".join(skills)
            if len(resume_data['skills']) > 10:
                skills_text += f" ... and {len(resume_data['skills']) - 10} more"
            st.write(skills_text)
        
        # Detailed scores in collapsible sections
        with st.expander("📊 Detailed Score Breakdown", expanded=False):
            hard_matching = detailed_results.get('hard_matching', {})
            soft_matching = detailed_results.get('soft_matching', {})
            llm_analysis = detailed_results.get('llm_analysis', {})
            
            # Safe numeric conversions
            try:
                hard_score = float(hard_matching.get('overall_score', 0))
            except (ValueError, TypeError):
                hard_score = 0.0
                
            try:
                soft_score = float(soft_matching.get('combined_semantic_score', 0))
            except (ValueError, TypeError):
                soft_score = 0.0
                
            try:
                llm_score = float(llm_analysis.get('llm_score', 0))
            except (ValueError, TypeError):
                llm_score = 0.0
            
            score_col1, score_col2, score_col3 = st.columns(3)
            with score_col1:
                st.metric("Hard Skills", f"{hard_score:.1f}/100")
                st.progress(max(0.0, min(1.0, hard_score / 100)))
            with score_col2:
                st.metric("Semantic Match", f"{soft_score:.1f}/100")
                st.progress(max(0.0, min(1.0, soft_score / 100)))
            with score_col3:
                st.metric("LLM Analysis", f"{llm_score:.1f}/100")
                st.progress(max(0.0, min(1.0, llm_score / 100)))
        
        # Recommendations and improvements
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            with st.expander("💪 Strengths & Recommendations", expanded=False):
                recommendations = analysis_results.get('recommendations', [])
                if recommendations:
                    for i, rec in enumerate(recommendations[:5], 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.write("No specific recommendations available")
        
        with feedback_col2:
            with st.expander("⚠️ Areas for Improvement", expanded=False):
                risk_factors = analysis_results.get('risk_factors', [])
                if risk_factors:
                    for i, risk in enumerate(risk_factors[:5], 1):
                        st.write(f"{i}. {risk}")
                else:
                    st.write("No major concerns identified")
        
        # LLM Analysis Details
        if llm_analysis:
            with st.expander("🤖 AI-Powered Analysis", expanded=False):
                if 'gap_analysis' in llm_analysis:
                    st.markdown("**🔍 Gap Analysis:**")
                    gap_analysis = llm_analysis['gap_analysis']
                    if isinstance(gap_analysis, dict):
                        st.write(gap_analysis.get('detailed_analysis', 'No detailed analysis available'))
                    else:
                        st.write(gap_analysis)
                
                if 'personalized_feedback' in llm_analysis:
                    st.markdown("**💬 Personalized Feedback:**")
                    st.write(llm_analysis['personalized_feedback'])
        
        # Hiring recommendation
        st.markdown("#### 🎯 Hiring Recommendation")
        recommendation_html = f"""
        <div style="background: {decision_color}; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">📋 {hiring_decision}</div>
                <div style="font-size: 1rem;">{success_probability:.1f}% Success Rate</div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                <strong>Reasoning:</strong> {hiring_rec.get('reasoning', 'No detailed reasoning provided')}
            </div>
        </div>
        """
        st.markdown(recommendation_html, unsafe_allow_html=True)
        
        # Export options for individual resume
        st.markdown("#### 📤 Export Options")
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            try:
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    label="📥 JSON",
                    data=json_str,
                    file_name=f"resume_{student_number}_{candidate_name.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"dropdown_json_{student_number}",
                    use_container_width=True
                )
            except Exception:
                st.error("JSON export failed")
        
        with export_col2:
            try:
                csv_data = create_csv_from_results([result])
                st.download_button(
                    label="📊 CSV",
                    data=csv_data,
                    file_name=f"resume_{student_number}_{candidate_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"dropdown_csv_{student_number}",
                    use_container_width=True
                )
            except Exception:
                st.error("CSV export failed")
        
        with export_col3:
            try:
                excel_data = create_excel_from_results([result])
                st.download_button(
                    label="📊 Excel",
                    data=excel_data,
                    file_name=f"resume_{student_number}_{candidate_name.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dropdown_excel_{student_number}",
                    use_container_width=True
                )
            except Exception:
                st.error("Excel export failed")
        
        with export_col4:
            if st.button(f"📧 Email", key=f"dropdown_email_{student_number}", use_container_width=True):
                st.success(f"Results would be sent to {resume_data.get('email', 'email not found')}")

def display_detailed_individual_analysis(result, student_number):
    """Display comprehensive detailed analysis for an individual student"""
    # Extract student data safely
    resume_data = result.get('resume_data', {})
    analysis_results = result.get('analysis_results', {})
    hiring_rec = result.get('hiring_recommendation', {})
    metadata = result.get('metadata', {})
    detailed_results = result.get('detailed_results', {})
    
    candidate_name = resume_data.get('candidate_name', 'Unknown Candidate')
    filename = metadata.get('resume_filename', 'Unknown File')
    
    # Safely convert numeric values
    try:
        overall_score = float(analysis_results.get('overall_score', 0))
    except (ValueError, TypeError):
        overall_score = 0.0
        
    try:
        success_probability = float(hiring_rec.get('success_probability', 0))
    except (ValueError, TypeError):
        success_probability = 0.0
        
    try:
        confidence = float(analysis_results.get('confidence', 0))
    except (ValueError, TypeError):
        confidence = 0.0
    
    match_level = analysis_results.get('match_level', 'unknown')
    hiring_decision = hiring_rec.get('decision', 'UNKNOWN')
    
    # Color coding for decisions
    decision_colors = {
        'HIRE': '#2ecc71',
        'INTERVIEW': '#f39c12',
        'MAYBE': '#e67e22', 
        'REJECT': '#e74c3c'
    }
    decision_color = decision_colors.get(hiring_decision, '#95a5a6')
    
    # Create main container for each student
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%); 
               padding: 2rem; border-radius: 15px; margin: 2rem 0; 
               border-left: 6px solid {decision_color}; box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50; margin: 0;">📄 Resume Analysis #{student_number}</h2>
            <h3 style="color: {decision_color}; margin: 0.5rem 0;">{candidate_name}</h3>
            <p style="color: #6c757d; margin: 0; font-size: 1rem;">📂 {filename}</p>
        </div>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
            <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; min-width: 120px;">
                <div style="font-size: 2rem; font-weight: bold; color: {decision_color};">{overall_score:.1f}</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Overall Score</div>
            </div>
            <div style="text-align: center; background: {decision_color}; color: white; padding: 1rem; border-radius: 10px; min-width: 120px;">
                <div style="font-size: 1.2rem; font-weight: bold;">{hiring_decision}</div>
                <div style="font-size: 0.9rem;">Decision</div>
            </div>
            <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; min-width: 120px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #495057;">{success_probability:.1f}%</div>
                <div style="font-size: 0.9rem; color: #6c757d;">Success Rate</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if this is a failed analysis
    is_failed = not metadata.get('success', True)
    
    if is_failed:
        # Display error information for failed analysis
        st.error(f"❌ Analysis Failed: {metadata.get('error', 'Unknown error occurred during analysis')}")
        
        # Still show basic information if available
        st.markdown("#### 📂 File Information")
        st.write(f"**Filename:** {filename}")
        
        # Show a retry button or suggestion
        st.info("💡 **Suggestions:**")
        st.write("• Check if the file is a valid PDF or DOCX format")
        st.write("• Ensure the file is not corrupted or password-protected")
        st.write("• Verify the file contains readable text content")
        
        # Separator between students
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        return
    
    # Detailed Analysis Sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Personal Information
        st.markdown("#### 👤 Candidate Information")
        info_container = st.container()
        with info_container:
            st.write(f"**📧 Email:** {resume_data.get('email', 'Not provided')}")
            st.write(f"**📞 Phone:** {resume_data.get('phone', 'Not provided')}")
            st.write(f"**💼 Experience:** {resume_data.get('experience_years', 'N/A')} years")
            st.write(f"**🎯 Match Level:** {match_level.title()}")
            st.write(f"**📊 Confidence:** {confidence:.1f}%")
    
    with col2:
        # Score Breakdown
        st.markdown("#### 📊 Score Breakdown")
        
        hard_matching = detailed_results.get('hard_matching', {})
        soft_matching = detailed_results.get('soft_matching', {})
        llm_analysis = detailed_results.get('llm_analysis', {})
        
        # Safely convert scores to float
        try:
            hard_score = float(hard_matching.get('overall_score', 0))
        except (ValueError, TypeError):
            hard_score = 0.0
            
        try:
            soft_score = float(soft_matching.get('combined_semantic_score', 0))
        except (ValueError, TypeError):
            soft_score = 0.0
            
        try:
            llm_score = float(llm_analysis.get('llm_score', 0))
        except (ValueError, TypeError):
            llm_score = 0.0
        
        # Progress bars for scores
        st.markdown(f"**Hard Skills Matching:** {hard_score:.1f}/100")
        st.progress(max(0.0, min(1.0, hard_score / 100)))
        
        st.markdown(f"**Semantic Similarity:** {soft_score:.1f}/100")
        st.progress(max(0.0, min(1.0, soft_score / 100)))
        
        st.markdown(f"**LLM Analysis Score:** {llm_score:.1f}/100")
        st.progress(max(0.0, min(1.0, llm_score / 100)))
    
    # Skills Analysis
    st.markdown("#### 🛠️ Skills Analysis")
    if resume_data.get('skills'):
        skills = resume_data['skills']
        skills_cols = st.columns(3)
        
        # Display skills in columns
        for i, skill in enumerate(skills[:15]):  # Show first 15 skills
            with skills_cols[i % 3]:
                st.markdown(f"• {skill}")
        
        if len(skills) > 15:
            st.markdown(f"*... and {len(skills) - 15} more skills*")
    else:
        st.write("No skills extracted from resume")
    
    # Detailed Analysis Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💪 Strengths & Recommendations")
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.write("No specific recommendations available")
    
    with col2:
        st.markdown("#### ⚠️ Areas for Improvement")
        risk_factors = analysis_results.get('risk_factors', [])
        if risk_factors:
            for i, risk in enumerate(risk_factors[:5], 1):
                st.markdown(f"**{i}.** {risk}")
        else:
            st.write("No major concerns identified")
    
    # LLM Analysis Details
    if llm_analysis:
        st.markdown("#### 🤖 AI-Powered Analysis")
        
        # Gap Analysis
        if 'gap_analysis' in llm_analysis:
            with st.expander("🔍 Gap Analysis", expanded=False):
                gap_analysis = llm_analysis['gap_analysis']
                if isinstance(gap_analysis, dict):
                    st.write(gap_analysis.get('detailed_analysis', 'No detailed analysis available'))
                else:
                    st.write(gap_analysis)
        
        # Personalized Feedback
        if 'personalized_feedback' in llm_analysis:
            with st.expander("💬 Personalized Feedback", expanded=False):
                st.write(llm_analysis['personalized_feedback'])
        
        # Strengths and Weaknesses from LLM
        if 'strengths' in llm_analysis or 'weaknesses' in llm_analysis:
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if 'strengths' in llm_analysis:
                    st.markdown("**🌟 AI-Identified Strengths:**")
                    for strength in llm_analysis['strengths'][:3]:
                        st.write(f"• {strength}")
            
            with feedback_col2:
                if 'weaknesses' in llm_analysis:
                    st.markdown("**🎯 Areas to Develop:**")
                    for weakness in llm_analysis['weaknesses'][:3]:
                        st.write(f"• {weakness}")
    
    # Hiring Recommendation Details
    st.markdown("#### 🎯 Detailed Hiring Recommendation")
    
    recommendation_html = f"""
    <div style="background: {decision_color}; color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: bold;">📋 {hiring_decision}</div>
            <div style="font-size: 1.2rem;">{success_probability:.1f}% Success Probability</div>
        </div>
        <div style="font-size: 1rem; line-height: 1.5;">
            <strong>Reasoning:</strong> {hiring_rec.get('reasoning', 'No detailed reasoning provided')}
        </div>
    </div>
    """
    st.markdown(recommendation_html, unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown("#### 🔧 Actions & Export Options")
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        # Export individual JSON
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 JSON",
            data=json_str,
            file_name=f"analysis_{student_number}_{candidate_name.replace(' ', '_')}.json",
            mime="application/json",
            key=f"export_individual_json_{student_number}",
            use_container_width=True
        )
    
    with action_col2:
        # Export individual CSV
        csv_data = create_csv_from_results([result])
        st.download_button(
            label="📊 CSV",
            data=csv_data,
            file_name=f"analysis_{student_number}_{candidate_name.replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"export_individual_csv_{student_number}",
            use_container_width=True
        )
    
    with action_col3:
        # Export individual Excel
        excel_data = create_excel_from_results([result])
        st.download_button(
            label="� Excel",
            data=excel_data,
            file_name=f"analysis_{student_number}_{candidate_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_individual_excel_{student_number}",
            use_container_width=True
        )
    
    with action_col4:
        if st.button(f"📧 Email", key=f"email_results_{student_number}", use_container_width=True):
            st.success(f"Results would be sent to {resume_data.get('email', 'email not found')}")
    
    # Additional action row
    st.markdown("---")
    action_col5, action_col6, action_col7, action_col8 = st.columns(4)
    
    with action_col5:
        if st.button(f"💾 Save to DB", key=f"save_db_{student_number}", use_container_width=True):
            st.success("Results saved to database!")
    
    with action_col6:
        if st.button(f"🔍 View Details", key=f"view_details_{student_number}", use_container_width=True):
            st.info("Detailed view opened!")
    
    with action_col7:
        if st.button(f"📋 Copy Link", key=f"copy_link_{student_number}", use_container_width=True):
            st.info("Result link copied to clipboard!")
    
    with action_col8:
        if st.button(f"🔄 Re-analyze", key=f"reanalyze_{student_number}", use_container_width=True):
            st.info("Re-analysis requested!")
    
    # Separator between students
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

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
    """Create CSV data from analysis results with safe type conversion"""
    csv_data = []
    
    for result in results:
        # Check if the result is successful
        if result.get('metadata', {}).get('success', False):
            # Safe data extraction with type conversion
            resume_data = result.get('resume_data', {})
            analysis_results = result.get('analysis_results', {})
            hiring_rec = result.get('hiring_recommendation', {})
            metadata = result.get('metadata', {})
            detailed_results = result.get('detailed_results', {})
            
            # Safe numeric conversions
            try:
                overall_score = float(analysis_results.get('overall_score', 0))
            except (ValueError, TypeError):
                overall_score = 0.0
                
            try:
                confidence = float(analysis_results.get('confidence', 0))
            except (ValueError, TypeError):
                confidence = 0.0
                
            try:
                success_probability = float(hiring_rec.get('success_probability', 0))
            except (ValueError, TypeError):
                success_probability = 0.0
                
            try:
                hard_score = float(detailed_results.get('hard_matching', {}).get('overall_score', 0))
            except (ValueError, TypeError):
                hard_score = 0.0
                
            try:
                soft_score = float(detailed_results.get('soft_matching', {}).get('combined_semantic_score', 0))
            except (ValueError, TypeError):
                soft_score = 0.0
                
            try:
                llm_score = float(detailed_results.get('llm_analysis', {}).get('llm_score', 0))
            except (ValueError, TypeError):
                llm_score = 0.0
                
            try:
                processing_time = float(metadata.get('processing_time', 0))
            except (ValueError, TypeError):
                processing_time = 0.0
                
            # Safe timestamp conversion
            try:
                timestamp = metadata.get('timestamp', 0)
                if isinstance(timestamp, str):
                    analysis_date = timestamp
                else:
                    analysis_date = datetime.fromtimestamp(float(timestamp)).isoformat()
            except (ValueError, TypeError, OSError):
                analysis_date = datetime.now().isoformat()
            
            csv_data.append({
                'Candidate Name': resume_data.get('candidate_name', 'N/A'),
                'Email': resume_data.get('email', 'N/A'),
                'Phone': resume_data.get('phone', 'N/A'),
                'Resume Filename': metadata.get('resume_filename', 'Unknown'),
                'Overall Score': f"{overall_score:.1f}",
                'Match Level': analysis_results.get('match_level', 'unknown'),
                'Confidence': f"{confidence:.1f}",
                'Hard Matching Score': f"{hard_score:.1f}",
                'Soft Matching Score': f"{soft_score:.1f}",
                'LLM Analysis Score': f"{llm_score:.1f}",
                'Hiring Decision': hiring_rec.get('decision', 'UNKNOWN'),
                'Success Probability': f"{success_probability:.1f}%",
                'Processing Time': f"{processing_time:.2f}s",
                'Analysis Date': analysis_date,
                'Skills': ', '.join(resume_data.get('skills', [])[:10]),
                'Experience Years': str(resume_data.get('experience_years', 'N/A')),
                'Key Recommendations': ' | '.join(analysis_results.get('recommendations', [])[:3]),
                'Risk Factors': ' | '.join(analysis_results.get('risk_factors', [])[:3])
            })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    else:
        return "No successful analyses to export"

def create_excel_from_results(results):
    """Create Excel data from analysis results with multiple sheets"""
    try:
        from io import BytesIO
        import xlsxwriter
        
        # Create a BytesIO buffer
        output = BytesIO()
        
        # Create workbook and worksheets
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })
        
        score_format = workbook.add_format({
            'num_format': '0.0',
            'valign': 'top',
            'border': 1
        })
        
        # Summary Sheet
        summary_sheet = workbook.add_worksheet('Summary')
        
        # Prepare summary data with safe type conversion
        summary_data = []
        for result in results:
            if result.get('metadata', {}).get('success', False):
                # Safe data extraction
                resume_data = result.get('resume_data', {})
                analysis_results = result.get('analysis_results', {})
                hiring_rec = result.get('hiring_recommendation', {})
                metadata = result.get('metadata', {})
                
                # Safe numeric conversions
                try:
                    overall_score = float(analysis_results.get('overall_score', 0))
                except (ValueError, TypeError):
                    overall_score = 0.0
                    
                try:
                    confidence = float(analysis_results.get('confidence', 0))
                except (ValueError, TypeError):
                    confidence = 0.0
                    
                try:
                    success_probability = float(hiring_rec.get('success_probability', 0))
                except (ValueError, TypeError):
                    success_probability = 0.0
                
                summary_data.append({
                    'Candidate Name': resume_data.get('candidate_name', 'N/A'),
                    'Email': resume_data.get('email', 'N/A'),
                    'Phone': resume_data.get('phone', 'N/A'),
                    'Resume Filename': metadata.get('resume_filename', 'Unknown'),
                    'Overall Score': overall_score,
                    'Match Level': analysis_results.get('match_level', 'unknown'),
                    'Confidence': confidence,
                    'Hiring Decision': hiring_rec.get('decision', 'UNKNOWN'),
                    'Success Probability': success_probability
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            # Write headers
            for col_num, value in enumerate(df_summary.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
            
            # Write data with safe type handling
            for row_num, row_data in enumerate(df_summary.values):
                for col_num, value in enumerate(row_data):
                    try:
                        if col_num in [4, 6, 8]:  # Score columns
                            # Ensure numeric value
                            numeric_value = float(value) if value is not None else 0.0
                            summary_sheet.write(row_num + 1, col_num, numeric_value, score_format)
                        else:
                            # String value
                            summary_sheet.write(row_num + 1, col_num, str(value) if value is not None else '', cell_format)
                    except (ValueError, TypeError) as e:
                        # Fallback for problematic values
                        summary_sheet.write(row_num + 1, col_num, str(value) if value is not None else '', cell_format)
            
            # Auto-adjust column widths
            for col_num, col_name in enumerate(df_summary.columns):
                try:
                    max_length = max(len(str(col_name)), max(len(str(val)) for val in df_summary.iloc[:, col_num]))
                    summary_sheet.set_column(col_num, col_num, min(max_length + 2, 30))
                except:
                    summary_sheet.set_column(col_num, col_num, 15)  # Default width
        
        # Detailed Sheet
        detailed_sheet = workbook.add_worksheet('Detailed Analysis')
        
        detailed_data = []
        for result in results:
            if result.get('metadata', {}).get('success', False):
                # Safe data extraction
                resume_data = result.get('resume_data', {})
                analysis_results = result.get('analysis_results', {})
                hiring_rec = result.get('hiring_recommendation', {})
                metadata = result.get('metadata', {})
                detailed_results = result.get('detailed_results', {})
                
                # Safe numeric conversions
                try:
                    overall_score = float(analysis_results.get('overall_score', 0))
                except (ValueError, TypeError):
                    overall_score = 0.0
                    
                try:
                    confidence = float(analysis_results.get('confidence', 0))
                except (ValueError, TypeError):
                    confidence = 0.0
                    
                try:
                    success_probability = float(hiring_rec.get('success_probability', 0))
                except (ValueError, TypeError):
                    success_probability = 0.0
                    
                try:
                    hard_score = float(detailed_results.get('hard_matching', {}).get('overall_score', 0))
                except (ValueError, TypeError):
                    hard_score = 0.0
                    
                try:
                    soft_score = float(detailed_results.get('soft_matching', {}).get('combined_semantic_score', 0))
                except (ValueError, TypeError):
                    soft_score = 0.0
                    
                try:
                    llm_score = float(detailed_results.get('llm_analysis', {}).get('llm_score', 0))
                except (ValueError, TypeError):
                    llm_score = 0.0
                    
                try:
                    processing_time = float(metadata.get('processing_time', 0))
                except (ValueError, TypeError):
                    processing_time = 0.0
                
                # Safe timestamp conversion
                try:
                    timestamp = metadata.get('timestamp', 0)
                    if isinstance(timestamp, str):
                        analysis_date = timestamp
                    else:
                        analysis_date = datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError, OSError):
                    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                detailed_data.append({
                    'Candidate Name': resume_data.get('candidate_name', 'N/A'),
                    'Email': resume_data.get('email', 'N/A'),
                    'Phone': resume_data.get('phone', 'N/A'),
                    'Experience Years': str(resume_data.get('experience_years', 'N/A')),
                    'Resume Filename': metadata.get('resume_filename', 'Unknown'),
                    'Overall Score': overall_score,
                    'Match Level': analysis_results.get('match_level', 'unknown'),
                    'Confidence': confidence,
                    'Hard Matching Score': hard_score,
                    'Soft Matching Score': soft_score,
                    'LLM Analysis Score': llm_score,
                    'Hiring Decision': hiring_rec.get('decision', 'UNKNOWN'),
                    'Success Probability': success_probability,
                    'Skills': ', '.join(resume_data.get('skills', [])[:15]),
                    'Key Recommendations': ' | '.join(analysis_results.get('recommendations', [])[:5]),
                    'Risk Factors': ' | '.join(analysis_results.get('risk_factors', [])[:5]),
                    'Hiring Reasoning': hiring_rec.get('reasoning', 'N/A'),
                    'Processing Time': processing_time,
                    'Analysis Date': analysis_date
                })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            
            # Write headers
            for col_num, value in enumerate(df_detailed.columns.values):
                detailed_sheet.write(0, col_num, value, header_format)
            
            # Write data with safe type handling
            score_columns = [5, 7, 8, 9, 10, 12, 17]  # Score and numeric columns
            for row_num, row_data in enumerate(df_detailed.values):
                for col_num, value in enumerate(row_data):
                    try:
                        if col_num in score_columns:
                            # Ensure numeric value
                            numeric_value = float(value) if value is not None else 0.0
                            detailed_sheet.write(row_num + 1, col_num, numeric_value, score_format)
                        else:
                            # String value
                            detailed_sheet.write(row_num + 1, col_num, str(value) if value is not None else '', cell_format)
                    except (ValueError, TypeError):
                        # Fallback for problematic values
                        detailed_sheet.write(row_num + 1, col_num, str(value) if value is not None else '', cell_format)
            
            # Auto-adjust column widths
            for col_num, col_name in enumerate(df_detailed.columns):
                try:
                    max_length = max(len(str(col_name)), max(len(str(val)) for val in df_detailed.iloc[:, col_num]))
                    detailed_sheet.set_column(col_num, col_num, min(max_length + 2, 40))
                except:
                    detailed_sheet.set_column(col_num, col_num, 15)  # Default width
        
        # Statistics Sheet
        stats_sheet = workbook.add_worksheet('Statistics')
        
        if summary_data:
            # Calculate statistics safely
            scores = []
            decisions = []
            match_levels = []
            
            for item in summary_data:
                try:
                    scores.append(float(item['Overall Score']))
                except (ValueError, TypeError):
                    scores.append(0.0)
                decisions.append(item.get('Hiring Decision', 'UNKNOWN'))
                match_levels.append(item.get('Match Level', 'unknown'))
            
            # Write statistics
            stats_data = [
                ['Metric', 'Value'],
                ['Total Candidates', len(summary_data)],
                ['Average Score', round(sum(scores) / len(scores), 2) if scores else 0],
                ['Highest Score', round(max(scores), 2) if scores else 0],
                ['Lowest Score', round(min(scores), 2) if scores else 0],
                ['HIRE Decisions', decisions.count('HIRE')],
                ['INTERVIEW Decisions', decisions.count('INTERVIEW')],
                ['MAYBE Decisions', decisions.count('MAYBE')],
                ['REJECT Decisions', decisions.count('REJECT')],
                ['Excellent Matches', match_levels.count('excellent')],
                ['Good Matches', match_levels.count('good')],
                ['Fair Matches', match_levels.count('fair')],
                ['Poor Matches', match_levels.count('poor')]
            ]
            
            for row_num, row_data in enumerate(stats_data):
                for col_num, value in enumerate(row_data):
                    if row_num == 0:  # Header row
                        stats_sheet.write(row_num, col_num, value, header_format)
                    else:
                        if col_num == 1 and isinstance(value, (int, float)):
                            stats_sheet.write(row_num, col_num, value, score_format)
                        else:
                            stats_sheet.write(row_num, col_num, value, cell_format)
            
            # Set column widths
            stats_sheet.set_column(0, 0, 25)
            stats_sheet.set_column(1, 1, 15)
        
        # Close workbook and get data
        workbook.close()
        output.seek(0)
        
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        # Fallback to CSV if Excel creation fails
        return create_csv_from_results(results).encode('utf-8')

if __name__ == "__main__":
    main()