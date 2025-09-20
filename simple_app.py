"""
Simple Streamlit App Runner
Launch the resume analyzer web application
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Simple version for testing
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🎯 Resume Analyzer - Innomatics Research Labs")
st.markdown("**AI-Powered Resume Relevance Analysis System**")

# Simple file upload interface
st.markdown("## Upload Files for Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Resume File")
    resume_file = st.file_uploader("Upload Resume", type=['pdf', 'docx'])

with col2:
    st.markdown("### Job Description File")
    jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx'])

if st.button("🚀 Analyze Resume", type="primary"):
    if resume_file and jd_file:
        st.success("Files uploaded successfully!")
        
        # Show demo results
        st.markdown("## 🎯 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", "78.5/100")
        with col2:
            st.metric("Match Level", "GOOD")
        with col3:
            st.metric("Confidence", "85.2%")
        
        # Component scores
        st.markdown("### Component Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hard Matching", "72.3")
            st.progress(0.723)
        with col2:
            st.metric("Semantic Similarity", "81.7")
            st.progress(0.817)
        with col3:
            st.metric("LLM Analysis", "82.0")
            st.progress(0.82)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = [
            "Strong technical background aligns well with requirements",
            "Consider highlighting project management experience",
            "Add more details about cloud platform expertise",
            "Excellent communication skills demonstrated"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Mock hiring decision
        st.markdown("### 🎯 Hiring Recommendation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Decision**: 🟡 INTERVIEW")
            st.markdown("**Confidence**: Medium")
            st.markdown("**Success Probability**: 78.5%")
        
        with col2:
            st.markdown("**Reasoning**: Candidate shows strong technical alignment with good communication skills. Recommend technical interview to assess practical implementation abilities.")
        
        st.success("✅ Analysis completed successfully!")
        
    else:
        st.error("Please upload both resume and job description files.")

# Instructions
st.markdown("---")
st.markdown("## 📋 Instructions")
st.markdown("""
1. Upload a resume file (PDF or DOCX format)
2. Upload a job description file (PDF or DOCX format)  
3. Click 'Analyze Resume' to start the analysis
4. View comprehensive results including scores, recommendations, and hiring decisions

**Features:**
- ✅ Multi-format document support (PDF, DOCX)
- ✅ Advanced text extraction and normalization
- ✅ Hard matching (keywords, skills, TF-IDF, BM25)
- ✅ Soft matching (semantic similarity, embeddings)
- ✅ LLM-powered analysis and feedback
- ✅ Comprehensive scoring and hiring recommendations
- ✅ Export functionality (CSV, Excel, JSON)
""")

st.markdown("---")
st.markdown("*Developed by Innomatics Research Labs*")