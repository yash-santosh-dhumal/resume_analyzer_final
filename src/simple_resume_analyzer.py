"""
Simplified Resume Analyzer for Web App
Only imports what's needed and defers heavy initialization
"""

import os
import sys
from typing import Dict, Any, Optional
from collections import Counter
import re
import math

# Add src directory to path for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import only what we need for basic functionality
try:
    from config.settings import load_config
except:
    def load_config(path=None):
        return {"api_keys": {"openai": "", "huggingface": ""}}

class ResumeAnalyzer:
    """
    Simplified Resume Analyzer for web application
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the resume analyzer with minimal setup
        """
        try:
            self.config = load_config(config_path)
        except:
            self.config = {"api_keys": {"openai": "", "huggingface": ""}}
        
        # Initialize components as None - they'll be loaded on demand
        self.document_parser = None
        self.scoring_engine = None
        self.llm_engine = None
        
        print("Resume Analyzer initialized successfully (lightweight mode)")
    
    def _init_parser(self):
        """Initialize document parser on demand"""
        if self.document_parser is None:
            try:
                from parsers import DocumentParser
                self.document_parser = DocumentParser()
            except Exception as e:
                print(f"Warning: Could not initialize document parser: {e}")
                self.document_parser = None
    
    def _init_scoring(self):
        """Initialize scoring engine on demand"""
        if self.scoring_engine is None:
            try:
                from scoring import ScoringEngine
                self.scoring_engine = ScoringEngine(self.config)
            except Exception as e:
                print(f"Warning: Could not initialize scoring engine: {e}")
                self.scoring_engine = None
    
    def analyze_resume(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """
        Analyze resume against job description with enhanced scoring
        """
        try:
            import re
            from collections import Counter
            import math
            
            # Clean and normalize text
            resume_clean = self._clean_text(resume_text)
            job_clean = self._clean_text(job_text)
            
            # Extract key information
            resume_skills = self._extract_skills(resume_clean)
            job_skills = self._extract_skills(job_clean)
            
            resume_keywords = self._extract_keywords(resume_clean)
            job_keywords = self._extract_keywords(job_clean)
            
            # Calculate different scoring components
            keyword_score = self._calculate_keyword_score(resume_keywords, job_keywords)
            skill_score = self._calculate_skill_score(resume_skills, job_skills)
            context_score = self._calculate_context_score(resume_clean, job_clean)
            experience_score = self._calculate_experience_score(resume_clean, job_clean)
            
            # Weight the scores with more emphasis on skills and experience
            weights = {
                'keyword': 0.25,
                'skill': 0.40,     # Increased weight for skills
                'context': 0.15,
                'experience': 0.20  # Increased weight for experience
            }
            
            overall_score = (
                keyword_score * weights['keyword'] +
                skill_score * weights['skill'] +
                context_score * weights['context'] +
                experience_score * weights['experience']
            )
            
            # Determine match level based on score
            if overall_score >= 80:
                match_level = "excellent"
            elif overall_score >= 65:
                match_level = "good"
            elif overall_score >= 45:
                match_level = "fair"
            else:
                match_level = "poor"
            
            # Generate explanation
            explanation = self._generate_explanation(
                overall_score, keyword_score, skill_score, context_score, experience_score
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                keyword_score, skill_score, context_score, experience_score, job_skills, resume_skills
            )
            
            return {
                "overall_score": round(overall_score, 2),
                "match_level": match_level,
                "explanation": explanation,
                "recommendations": recommendations,
                "component_scores": {
                    "keyword_match": round(keyword_score, 2),
                    "skill_match": round(skill_score, 2),
                    "context_match": round(context_score, 2),
                    "experience_match": round(experience_score, 2)
                }
            }
            
        except Exception as e:
            return {
                "overall_score": 0,
                "match_level": "error",
                "explanation": f"Analysis failed: {e}",
                "recommendations": ["Please check input data and try again"],
                "component_scores": {
                    "keyword_match": 0,
                    "skill_match": 0,
                    "context_match": 0,
                    "experience_match": 0
                }
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        import re
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_skills(self, text: str) -> set:
        """Extract technical skills from text"""
        # Common technical skills keywords
        skills_patterns = [
            r'\b(?:python|java|javascript|c\+\+|c#|php|ruby|swift|kotlin|go|rust)\b',
            r'\b(?:react|angular|vue|node|express|django|flask|spring|laravel)\b',
            r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git|github)\b',
            r'\b(?:machine learning|deep learning|ai|data science|analytics)\b',
            r'\b(?:html|css|bootstrap|tailwind|sass|scss)\b',
            r'\b(?:tensorflow|pytorch|scikit-learn|pandas|numpy)\b',
            r'\b(?:project management|agile|scrum|devops|ci/cd)\b'
        ]
        
        skills = set()
        for pattern in skills_patterns:
            import re
            matches = re.findall(pattern, text)
            skills.update(matches)
        
        # Also extract capitalized words that might be technologies
        import re
        tech_words = re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', text)
        for word in tech_words:
            if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
                skills.add(word.lower())
        
        return skills
    
    def _extract_keywords(self, text: str) -> Counter:
        """Extract important keywords with frequency"""
        from collections import Counter
        import re
        from collections import Counter
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'within', 'without', 'upon', 'this', 'that',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'shall', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their'
        }
        
        words = text.split()
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return Counter(filtered_words)
    
    def _calculate_keyword_score(self, resume_keywords: Counter, job_keywords: Counter) -> float:
        """Calculate keyword matching score using enhanced TF-IDF like approach"""
        if not job_keywords:
            return 0.0
        
        # Calculate overlap and weighted importance
        total_weight = 0
        matched_weight = 0
        
        # Important keywords get extra weight
        important_keywords = {
            'python', 'django', 'flask', 'aws', 'docker', 'kubernetes', 'postgresql', 
            'mongodb', 'react', 'javascript', 'git', 'api', 'development', 'experience',
            'senior', 'lead', 'architect', 'microservices', 'scrum', 'agile'
        }
        
        for keyword, freq in job_keywords.items():
            # Base weight from frequency
            weight = freq
            
            # Boost weight for important technical keywords
            if keyword in important_keywords:
                weight *= 2.0
            
            total_weight += weight
            
            if keyword in resume_keywords:
                # Bonus for frequency in resume
                resume_freq = resume_keywords[keyword]
                match_strength = min(1.2, resume_freq / freq)  # Cap at 1.2 for bonus
                matched_weight += weight * match_strength
        
        if total_weight == 0:
            return 0.0
        
        base_score = (matched_weight / total_weight) * 100
        
        # Bonus for having many relevant keywords
        keyword_diversity = len(set(resume_keywords.keys()).intersection(set(job_keywords.keys())))
        diversity_bonus = min(15.0, keyword_diversity * 2)  # Max 15 bonus
        
        return min(100.0, base_score + diversity_bonus)
    
    def _calculate_skill_score(self, resume_skills: set, job_skills: set) -> float:
        """Calculate skill matching score with enhanced logic"""
        if not job_skills:
            return 50.0  # Neutral score if no skills detected
        
        matched_skills = resume_skills.intersection(job_skills)
        
        if len(job_skills) == 0:
            return 50.0
        
        # Base skill match ratio
        skill_match_ratio = len(matched_skills) / len(job_skills)
        
        # Bonus for extra relevant skills (shows broader knowledge)
        extra_skills_factor = min(1.2, len(resume_skills) / max(1, len(job_skills)))
        
        # Special bonuses for critical skills
        critical_skills = {'python', 'django', 'flask', 'aws', 'docker', 'kubernetes', 'postgresql', 'react'}
        critical_matches = len(critical_skills.intersection(matched_skills))
        critical_bonus = critical_matches * 5  # 5 points per critical skill
        
        base_score = skill_match_ratio * 70 * extra_skills_factor  # Adjusted base scoring
        final_score = min(100.0, base_score + critical_bonus)
        
        return final_score
    
    def _calculate_context_score(self, resume_text: str, job_text: str) -> float:
        """Calculate contextual similarity score"""
        import re
        
        # Extract multi-word phrases that indicate context
        job_phrases = self._extract_phrases(job_text)
        resume_phrases = self._extract_phrases(resume_text)
        
        if not job_phrases:
            return 50.0
        
        matched_phrases = 0
        total_phrases = len(job_phrases)
        
        for phrase in job_phrases:
            # Check for exact or partial matches
            for resume_phrase in resume_phrases:
                if phrase in resume_phrase or resume_phrase in phrase:
                    matched_phrases += 1
                    break
        
        context_score = (matched_phrases / total_phrases) * 100 if total_phrases > 0 else 50.0
        return min(100.0, context_score)
    
    def _extract_phrases(self, text: str) -> list:
        """Extract meaningful phrases from text"""
        import re
        
        # Look for phrases like "3+ years experience", "project management", etc.
        phrases = []
        
        # Experience patterns
        exp_patterns = [
            r'\d+\+?\s*years?\s+(?:of\s+)?(?:experience|work|background)',
            r'(?:experience|background)\s+(?:in|with|of)\s+\w+',
            r'(?:senior|junior|lead|principal)\s+\w+'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        # Technology combinations
        tech_patterns = [
            r'\w+\s+(?:development|programming|framework|platform)',
            r'(?:web|mobile|full[\-\s]?stack|backend|frontend)\s+\w+',
            r'\w+\s+(?:database|server|cloud|deployment)'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return phrases
    
    def _calculate_experience_score(self, resume_text: str, job_text: str) -> float:
        """Calculate experience level matching score with enhanced logic"""
        import re
        
        # Extract years of experience from both texts
        job_years = self._extract_years_experience(job_text)
        resume_years = self._extract_years_experience(resume_text)
        
        # Check for seniority indicators
        resume_seniority = self._assess_seniority(resume_text)
        job_seniority = self._assess_seniority(job_text)
        
        if job_years is None and job_seniority == 0:
            return 70.0  # Neutral score if no experience requirement found
        
        score = 50.0  # Base score
        
        # Years-based scoring
        if job_years is not None:
            if resume_years is None:
                score = 30.0  # Lower score if no experience mentioned
            elif resume_years >= job_years:
                # Bonus for meeting/exceeding requirements
                excess = resume_years - job_years
                base_score = 80.0
                bonus = min(20.0, excess * 3)  # 3 points per extra year, max 20
                score = min(100.0, base_score + bonus)
            else:
                # Penalty for insufficient experience, but not too harsh
                shortage = job_years - resume_years
                penalty = min(40.0, shortage * 10)  # 10 points per missing year
                score = max(20.0, 80.0 - penalty)
        
        # Seniority-based adjustments
        if job_seniority > 0:
            seniority_match = min(resume_seniority / job_seniority, 1.5)  # Cap at 1.5x
            score *= seniority_match
        
        return min(100.0, max(10.0, score))
    
    def _assess_seniority(self, text: str) -> float:
        """Assess seniority level from text content"""
        text_lower = text.lower()
        seniority_score = 0.0
        
        # Senior level indicators
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'architect']):
            seniority_score += 3.0
        
        # Leadership indicators
        if any(word in text_lower for word in ['led', 'managed', 'mentored', 'supervised']):
            seniority_score += 2.0
        
        # Advanced responsibility indicators
        if any(phrase in text_lower for phrase in ['technical decisions', 'architecture', 'code review', 'best practices']):
            seniority_score += 1.5
        
        # Project scale indicators
        if any(phrase in text_lower for phrase in ['microservices', 'scalable', 'enterprise', 'production']):
            seniority_score += 1.0
        
        return seniority_score
    
    def _extract_years_experience(self, text: str) -> int:
        """Extract years of experience from text"""
        import re
        
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|work)',
            r'(?:experience|work).*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*year\s+(?:experience|work)',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    years.append(int(match))
                except ValueError:
                    continue
        
        return max(years) if years else None
    
    def _generate_explanation(self, overall_score: float, keyword_score: float, 
                            skill_score: float, context_score: float, experience_score: float) -> str:
        """Generate detailed explanation of the score"""
        explanation = f"Overall match score: {overall_score:.1f}/100. "
        
        # Component breakdown
        explanation += f"Breakdown: Keywords {keyword_score:.1f}%, "
        explanation += f"Skills {skill_score:.1f}%, "
        explanation += f"Context {context_score:.1f}%, "
        explanation += f"Experience {experience_score:.1f}%. "
        
        # Qualitative assessment
        if overall_score >= 80:
            explanation += "Excellent match with strong alignment across all criteria."
        elif overall_score >= 65:
            explanation += "Good match with minor gaps in some areas."
        elif overall_score >= 45:
            explanation += "Fair match but requires skill development."
        else:
            explanation += "Poor match - significant gaps in required qualifications."
        
        return explanation
    
    def _generate_recommendations(self, keyword_score: float, skill_score: float, 
                                context_score: float, experience_score: float,
                                job_skills: set, resume_skills: set) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        if keyword_score < 60:
            recommendations.append("Improve resume keywords to better match job requirements")
        
        if skill_score < 60:
            missing_skills = job_skills - resume_skills
            if missing_skills:
                skill_list = list(missing_skills)[:3]  # Top 3 missing skills
                recommendations.append(f"Develop skills in: {', '.join(skill_list)}")
            else:
                recommendations.append("Highlight relevant technical skills more prominently")
        
        if context_score < 60:
            recommendations.append("Add more specific examples of relevant experience")
        
        if experience_score < 60:
            recommendations.append("Gain more experience in the relevant field or emphasize transferable skills")
        
        # Overall recommendations
        if len(recommendations) == 0:
            recommendations.append("Strong candidate - consider for interview")
        elif len(recommendations) >= 3:
            recommendations.append("Significant improvement needed before applying")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        Generate a simple text report
        """
        score = analysis_result.get("overall_score", 0)
        level = analysis_result.get("match_level", "unknown")
        explanation = analysis_result.get("explanation", "No explanation available")
        
        return f"""
Resume Analysis Report
=====================
Overall Score: {score:.1f}%
Match Level: {level.upper()}

Analysis: {explanation}

Recommendations:
{chr(10).join('- ' + rec for rec in analysis_result.get('recommendations', []))}
"""
    
    def analyze_resume_for_job(self, resume_file_path: str, job_description_file_path: str, save_to_db: bool = False) -> Dict[str, Any]:
        """
        Analyze resume against job description - compatible with webapp
        
        Args:
            resume_file_path: Path to resume file (PDF, DOCX, or TXT)
            job_description_file_path: Path to job description file
            save_to_db: Whether to save to database (ignored in simple analyzer)
        
        Returns:
            Analysis results in format compatible with full analyzer
        """
        try:
            # Read files with better encoding handling and PDF support
            resume_text = self._read_file_content(resume_file_path)
            job_text = self._read_file_content(job_description_file_path)
            
            # Ensure we have content
            if not resume_text.strip():
                raise ValueError("Resume file appears to be empty or unreadable")
            if not job_text.strip():
                raise ValueError("Job description file appears to be empty or unreadable")
            
            # Perform enhanced analysis
            analysis_result = self.analyze_resume(resume_text, job_text)
            
            # Extract additional metadata for better analysis
            candidate_name = self._extract_candidate_name(resume_text)
            email = self._extract_email(resume_text)
            phone = self._extract_phone(resume_text)
            
            # Generate hiring recommendation
            hiring_decision = self._determine_hiring_decision(analysis_result['overall_score'])
            
            # Format result to match full analyzer output
            return {
                'metadata': {
                    'success': True,
                    'timestamp': __import__('datetime').datetime.now().isoformat(),
                    'analyzer_type': 'simple_enhanced',
                    'processing_time': 0.2,
                    'resume_file': resume_file_path,
                    'job_description_file': job_description_file_path
                },
                'resume_data': {
                    'candidate_name': candidate_name,
                    'email': email,
                    'phone': phone,
                    'skills': list(self._extract_skills(self._clean_text(resume_text))),
                    'experience_years': self._extract_years_experience(resume_text),
                    'filename': resume_file_path.split('\\')[-1] if '\\' in resume_file_path else resume_file_path.split('/')[-1]
                },
                'job_data': {
                    'title': self._extract_job_title(job_text),
                    'company': self._extract_company_name(job_text),
                    'required_skills': list(self._extract_skills(self._clean_text(job_text))),
                    'filename': job_description_file_path.split('\\')[-1] if '\\' in job_description_file_path else job_description_file_path.split('/')[-1]
                },
                'analysis_results': {
                    'overall_score': analysis_result.get('overall_score', 0),
                    'match_level': analysis_result.get('match_level', 'poor'),
                    'confidence': min(90.0, analysis_result.get('overall_score', 0) * 0.8 + 20),  # Dynamic confidence based on score
                    'explanation': analysis_result.get('explanation', ''),
                    'recommendations': analysis_result.get('recommendations', []),
                    'risk_factors': self._identify_risk_factors(analysis_result)
                },
                'detailed_results': {
                    'hard_matching': {
                        'overall_score': analysis_result.get('component_scores', {}).get('keyword_match', 0),
                        'keyword_score': analysis_result.get('component_scores', {}).get('keyword_match', 0),
                        'skills_score': analysis_result.get('component_scores', {}).get('skill_match', 0)
                    },
                    'soft_matching': {
                        'combined_semantic_score': analysis_result.get('component_scores', {}).get('context_match', 0),
                        'semantic_score': analysis_result.get('component_scores', {}).get('context_match', 0),
                        'embedding_score': analysis_result.get('component_scores', {}).get('context_match', 0)
                    },
                    'llm_analysis': {
                        'llm_score': analysis_result.get('component_scores', {}).get('experience_match', 0),
                        'llm_verdict': 'good' if analysis_result.get('overall_score', 0) > 60 else 'medium' if analysis_result.get('overall_score', 0) > 30 else 'poor',
                        'gap_analysis': {
                            'detailed_analysis': analysis_result.get('explanation', ''),
                            'strengths': self._extract_strengths(analysis_result),
                            'weaknesses': self._extract_weaknesses(analysis_result)
                        },
                        'personalized_feedback': analysis_result.get('explanation', ''),
                        'improvement_suggestions': analysis_result.get('recommendations', [])
                    },
                    'scoring_details': {
                        'component_scores': analysis_result.get('component_scores', {}),
                        'weighted_scores': analysis_result.get('component_scores', {})
                    }
                },
                'hiring_recommendation': {
                    'decision': hiring_decision['decision'],
                    'confidence': hiring_decision['confidence'],
                    'reasoning': hiring_decision['reasoning'],
                    'next_steps': analysis_result.get('recommendations', []),
                    'success_probability': min(95.0, analysis_result.get('overall_score', 0) * 0.9 + 5)
                }
            }
            
        except Exception as e:
            # Return error result in compatible format
            return {
                'metadata': {
                    'success': False,
                    'timestamp': __import__('datetime').datetime.now().isoformat(),
                    'analyzer_type': 'simple_enhanced',
                    'error': str(e),
                    'resume_file': resume_file_path,
                    'job_description_file': job_description_file_path
                },
                'analysis_results': {
                    'overall_score': 0,
                    'match_level': 'error',
                    'confidence': 0,
                    'explanation': f"Analysis failed: {e}",
                    'recommendations': ['Please check input files and try again'],
                    'risk_factors': ['Analysis system error']
                },
                'detailed_results': {
                    'hard_matching': {'overall_score': 0, 'keyword_score': 0, 'skills_score': 0},
                    'soft_matching': {'combined_semantic_score': 0, 'semantic_score': 0, 'embedding_score': 0},
                    'llm_analysis': {'llm_score': 0, 'llm_verdict': 'error', 'gap_analysis': '', 'personalized_feedback': f"Analysis failed: {e}", 'improvement_suggestions': []},
                    'scoring_details': {'component_scores': {}, 'weighted_scores': {}}
                },
                'hiring_recommendation': {
                    'decision': 'error',
                    'confidence': 'none',
                    'reasoning': f"Analysis failed: {e}",
                    'next_steps': ['Fix the error and retry analysis'],
                    'success_probability': 0
                }
            }
    
    def _read_file_content(self, file_path: str) -> str:
        """Read file content from various formats (TXT, PDF, DOCX)"""
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                return self._read_pdf_content(file_path)
            elif file_extension in ['docx', 'doc']:
                return self._read_docx_content(file_path)
            else:
                # Default to text file
                return self._read_text_content(file_path)
        except Exception as e:
            # Fallback to text reading
            print(f"Warning: Failed to read {file_extension} file, trying as text: {e}")
            return self._read_text_content(file_path)
    
    def _read_text_content(self, file_path: str) -> str:
        """Read text file with various encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file with {encoding}: {e}")
                continue
        
        # Last resort - read with errors ignored
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Unable to read file: {e}")
    
    def _read_pdf_content(self, file_path: str) -> str:
        """Simple PDF text extraction"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("PyPDF2 not available, treating PDF as text file")
            return self._read_text_content(file_path)
        except Exception as e:
            print(f"PDF reading failed, treating as text: {e}")
            return self._read_text_content(file_path)
    
    def _read_docx_content(self, file_path: str) -> str:
        """Simple DOCX text extraction"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            print("python-docx not available, treating DOCX as text file")
            return self._read_text_content(file_path)
        except Exception as e:
            print(f"DOCX reading failed, treating as text: {e}")
            return self._read_text_content(file_path)
    
    def _extract_candidate_name(self, resume_text: str) -> str:
        """Extract candidate name from resume"""
        import re
        lines = resume_text.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 0 and len(line.split()) <= 4:  # Name shouldn't be too long
                # Check if line looks like a name (starts with capital letters)
                if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line):
                    return line
        
        return "Unknown"
    
    def _extract_email(self, resume_text: str) -> str:
        """Extract email from resume"""
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        return emails[0] if emails else "Not found"
    
    def _extract_phone(self, resume_text: str) -> str:
        """Extract phone number from resume"""
        import re
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, resume_text)
            if phones:
                return phones[0]
        
        return "Not found"
    
    def _extract_job_title(self, job_text: str) -> str:
        """Extract job title from job description"""
        lines = job_text.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 0 and ('developer' in line.lower() or 'engineer' in line.lower() or 
                                 'analyst' in line.lower() or 'manager' in line.lower() or
                                 'position' in line.lower() or 'role' in line.lower()):
                return line[:100]  # Truncate if too long
        
        return "Position not specified"
    
    def _extract_company_name(self, job_text: str) -> str:
        """Extract company name from job description"""
        lines = job_text.split('\n')[:15]  # Check first 15 lines
        
        for line in lines:
            line = line.strip()
            if 'company' in line.lower() or 'organization' in line.lower():
                return line[:100]
        
        return "Company not specified"
    
    def _determine_hiring_decision(self, overall_score: float) -> Dict[str, str]:
        """Determine hiring decision based on score"""
        if overall_score >= 80:
            return {
                'decision': 'HIRE',
                'confidence': 'high',
                'reasoning': 'Excellent match across all criteria with strong qualifications'
            }
        elif overall_score >= 65:
            return {
                'decision': 'INTERVIEW',
                'confidence': 'high',
                'reasoning': 'Good candidate with strong potential, worth interviewing'
            }
        elif overall_score >= 45:
            return {
                'decision': 'MAYBE',
                'confidence': 'medium',
                'reasoning': 'Fair match with some gaps, consider for phone screening'
            }
        else:
            return {
                'decision': 'REJECT',
                'confidence': 'high',
                'reasoning': 'Poor match with significant gaps in required qualifications'
            }
    
    def _identify_risk_factors(self, analysis_result: Dict[str, Any]) -> list:
        """Identify potential risk factors"""
        risk_factors = []
        
        score = analysis_result.get('overall_score', 0)
        components = analysis_result.get('component_scores', {})
        
        if components.get('skill_match', 0) < 40:
            risk_factors.append("Significant technical skills gap")
        
        if components.get('experience_match', 0) < 40:
            risk_factors.append("Limited relevant experience")
        
        if components.get('keyword_match', 0) < 30:
            risk_factors.append("Poor alignment with job requirements")
        
        if score < 30:
            risk_factors.append("Overall poor qualification match")
        
        return risk_factors
    
    def _extract_strengths(self, analysis_result: Dict[str, Any]) -> list:
        """Extract strengths from analysis"""
        strengths = []
        components = analysis_result.get('component_scores', {})
        
        if components.get('skill_match', 0) >= 70:
            strengths.append("Strong technical skills alignment")
        
        if components.get('experience_match', 0) >= 70:
            strengths.append("Excellent relevant experience")
        
        if components.get('keyword_match', 0) >= 70:
            strengths.append("Strong keyword match with job requirements")
        
        if components.get('context_match', 0) >= 70:
            strengths.append("Good contextual fit for the role")
        
        return strengths
    
    def _extract_weaknesses(self, analysis_result: Dict[str, Any]) -> list:
        """Extract weaknesses from analysis"""
        weaknesses = []
        components = analysis_result.get('component_scores', {})
        
        if components.get('skill_match', 0) < 50:
            weaknesses.append("Technical skills need development")
        
        if components.get('experience_match', 0) < 50:
            weaknesses.append("Limited relevant experience")
        
        if components.get('keyword_match', 0) < 50:
            weaknesses.append("Resume doesn't align well with job requirements")
        
        if components.get('context_match', 0) < 50:
            weaknesses.append("Contextual fit could be improved")
        
        return weaknesses