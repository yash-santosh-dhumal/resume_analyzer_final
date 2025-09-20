"""
Simple Working Streamlit Web Application for Resume Analyzer
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the simplified analyzer
try:
    from simple_resume_analyzer import ResumeAnalyzer
    ANALYZER_AVAILABLE = True
except Exception as e:
    st.error(f"Could not import ResumeAnalyzer: {e}")
    ANALYZER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer - Innomatics Research Labs",
    page_icon="📄",
    layout="wide"
)

# Title and header
st.title("📄 Resume Analyzer")
st.markdown("**AI-Powered Resume Relevance Analysis**")
st.markdown("---")

if not ANALYZER_AVAILABLE:
    st.error("⚠️ Resume Analyzer is not available. Please check the installation.")
    st.stop()

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    try:
        return ResumeAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {e}")
        return None

analyzer = get_analyzer()

if analyzer is None:
    st.error("❌ Failed to initialize Resume Analyzer")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload Resume**: Paste or upload your resume text
    2. **Job Description**: Paste the target job description  
    3. **Analyze**: Click analyze to get relevance score
    4. **Review**: Check recommendations and improve your resume
    """)
    
    st.header("ℹ️ About")
    st.markdown("""
    This tool analyzes how well your resume matches a specific job posting using AI-powered analysis.
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Resume")
    resume_input_method = st.radio("Input method:", ["Text Input", "File Upload"])
    
    resume_text = ""
    if resume_input_method == "Text Input":
        resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Copy and paste your resume content here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload resume file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a text, PDF, or Word document"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    resume_text = str(uploaded_file.read(), "utf-8")
                else:
                    resume_text = "File uploaded successfully. Text extraction from PDF/DOCX files is not yet implemented in this demo."
                    
                st.success(f"✅ File uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

with col2:
    st.header("💼 Job Description")
    job_text = st.text_area(
        "Paste the job description here:",
        height=300,
        placeholder="Copy and paste the target job description here..."
    )

# Analysis button and results
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if analyze_button:
    if not resume_text.strip():
        st.error("❌ Please provide resume text")
    elif not job_text.strip():
        st.error("❌ Please provide job description")
    else:
        with st.spinner("🔄 Analyzing resume..."):
            try:
                # Perform analysis
                result = analyzer.analyze_resume(resume_text, job_text)
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Score display
                score = result.get("overall_score", 0)
                match_level = result.get("match_level", "unknown")
                
                # Color coding for score
                if score >= 80:
                    score_color = "🟢"
                elif score >= 60:
                    score_color = "🟡"
                else:
                    score_color = "🔴"
                
                # Results layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Overall Score",
                        value=f"{score:.1f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Match Level",
                        value=match_level.upper(),
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Status",
                        value=f"{score_color} {match_level.title()}",
                        delta=None
                    )
                
                # Detailed breakdown
                st.markdown("### 📊 Detailed Analysis")
                
                explanation = result.get("explanation", "No explanation available")
                st.info(f"**Analysis:** {explanation}")
                
                # Component scores
                if "component_scores" in result and result["component_scores"]:
                    st.markdown("#### Component Scores")
                    for component, comp_score in result["component_scores"].items():
                        st.progress(comp_score / 100, text=f"{component.replace('_', ' ').title()}: {comp_score:.1f}%")
                
                # Recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.markdown("#### 💡 Recommendations")
                    for i, rec in enumerate(result["recommendations"], 1):
                        st.write(f"{i}. {rec}")
                
                # Report generation
                st.markdown("---")
                with st.expander("📄 Generate Report"):
                    report = analyzer.generate_report(result)
                    st.text_area("Report:", value=report, height=200)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"resume_analysis_report_{score:.0f}pct.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Resume Analyzer | Powered by AI | © 2025 Innomatics Research Labs
    </div>
    """,
    unsafe_allow_html=True
)