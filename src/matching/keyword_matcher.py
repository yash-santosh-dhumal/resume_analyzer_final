"""
Keyword Matching Module
Implements exact and fuzzy keyword matching for resumes and job descriptions
"""

from typing import List, Dict, Set, Any, Tuple
import re
from collections import Counter
from fuzzywuzzy import fuzz, process
import logging

logger = logging.getLogger(__name__)

class KeywordMatcher:
    """Handles keyword-based matching between resume and job description"""
    
    def __init__(self, fuzzy_threshold: int = 80):
        """
        Initialize keyword matcher
        
        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        
        # Common skill categories
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 
                'rust', 'kotlin', 'swift', 'typescript', 'scala', 'r', 'matlab'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 
                'django', 'flask', 'spring', 'laravel', 'rails', 'asp.net'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 
                'oracle', 'sqlite', 'cassandra', 'dynamodb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 
                'microsoft azure', 'docker', 'kubernetes', 'terraform'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data science', 'tensorflow', 
                'pytorch', 'scikit-learn', 'pandas', 'numpy', 'jupyter', 'tableau'
            ],
            'tools_frameworks': [
                'git', 'jenkins', 'jira', 'confluence', 'slack', 'figma', 
                'photoshop', 'visual studio', 'intellij', 'eclipse'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd', 
                'microservices', 'rest api', 'graphql'
            ]
        }
        
        # Flatten all skills for easy lookup
        self.all_skills = set()
        for category_skills in self.skill_categories.values():
            self.all_skills.update([skill.lower() for skill in category_skills])
        
        # Common soft skills
        self.soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving', 
            'analytical thinking', 'creativity', 'adaptability', 'time management',
            'project management', 'critical thinking', 'collaboration'
        }
    
    def extract_keywords(self, text: str, include_skills: bool = True) -> Dict[str, List[str]]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            include_skills: Whether to identify technical skills
        
        Returns:
            Dictionary with categorized keywords
        """
        keywords = {
            'all_keywords': [],
            'technical_skills': [],
            'soft_skills': [],
            'tools_technologies': [],
            'programming_languages': [],
            'unique_terms': []
        }
        
        if not text:
            return keywords
        
        text_lower = text.lower()
        
        # Extract all meaningful words (3+ characters, alphanumeric)
        word_pattern = r'\b[a-zA-Z][a-zA-Z0-9+#\-.]{2,}\b'
        all_words = re.findall(word_pattern, text_lower)
        
        # Remove common stop words but keep technical terms
        tech_stop_words = {
            'and', 'the', 'for', 'with', 'using', 'experience', 'knowledge', 
            'skills', 'ability', 'strong', 'good', 'excellent', 'years'
        }
        
        filtered_words = [word for word in all_words if word not in tech_stop_words]
        keywords['all_keywords'] = list(set(filtered_words))
        
        if include_skills:
            # Identify technical skills
            for word in filtered_words:
                if word in self.all_skills:
                    keywords['technical_skills'].append(word)
                    
                    # Categorize by skill type
                    for category, skills in self.skill_categories.items():
                        if word in [skill.lower() for skill in skills]:
                            if category == 'programming_languages':
                                keywords['programming_languages'].append(word)
                            else:
                                keywords['tools_technologies'].append(word)
            
            # Identify soft skills
            for skill in self.soft_skills:
                if skill in text_lower:
                    keywords['soft_skills'].append(skill)
            
            # Look for multi-word technical terms
            multiword_skills = [
                'machine learning', 'deep learning', 'data science', 'web development',
                'software development', 'full stack', 'front end', 'back end',
                'rest api', 'web services', 'cloud computing', 'big data'
            ]
            
            for skill in multiword_skills:
                if skill in text_lower:
                    keywords['technical_skills'].append(skill)
        
        # Remove duplicates
        for key in keywords:
            keywords[key] = list(set(keywords[key]))
        
        return keywords
    
    def exact_match(self, resume_keywords: List[str], jd_keywords: List[str]) -> Dict[str, Any]:
        """
        Perform exact keyword matching
        
        Args:
            resume_keywords: Keywords from resume
            jd_keywords: Keywords from job description
        
        Returns:
            Match results with statistics
        """
        resume_set = set([kw.lower() for kw in resume_keywords])
        jd_set = set([kw.lower() for kw in jd_keywords])
        
        matched = resume_set.intersection(jd_set)
        missing = jd_set - resume_set
        extra = resume_set - jd_set
        
        match_percentage = (len(matched) / len(jd_set) * 100) if jd_set else 0
        
        return {
            'matched_keywords': list(matched),
            'missing_keywords': list(missing),
            'extra_keywords': list(extra),
            'match_count': len(matched),
            'total_jd_keywords': len(jd_set),
            'match_percentage': round(match_percentage, 2),
            'precision': len(matched) / len(resume_set) if resume_set else 0,
            'recall': len(matched) / len(jd_set) if jd_set else 0
        }
    
    def fuzzy_match(self, resume_keywords: List[str], jd_keywords: List[str]) -> Dict[str, Any]:
        """
        Perform fuzzy keyword matching
        
        Args:
            resume_keywords: Keywords from resume
            jd_keywords: Keywords from job description
        
        Returns:
            Fuzzy match results
        """
        fuzzy_matches = []
        matched_jd_keywords = set()
        
        for jd_keyword in jd_keywords:
            # Find best match in resume keywords
            match_result = process.extractOne(
                jd_keyword, 
                resume_keywords, 
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] >= self.fuzzy_threshold:
                fuzzy_matches.append({
                    'jd_keyword': jd_keyword,
                    'resume_keyword': match_result[0],
                    'similarity': match_result[1]
                })
                matched_jd_keywords.add(jd_keyword)
        
        missing_keywords = [kw for kw in jd_keywords if kw not in matched_jd_keywords]
        
        avg_similarity = (
            sum(match['similarity'] for match in fuzzy_matches) / len(fuzzy_matches)
            if fuzzy_matches else 0
        )
        
        match_percentage = (len(fuzzy_matches) / len(jd_keywords) * 100) if jd_keywords else 0
        
        return {
            'fuzzy_matches': fuzzy_matches,
            'missing_keywords': missing_keywords,
            'match_count': len(fuzzy_matches),
            'total_jd_keywords': len(jd_keywords),
            'match_percentage': round(match_percentage, 2),
            'average_similarity': round(avg_similarity, 2)
        }
    
    def weighted_keyword_match(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Perform weighted keyword matching with category-specific scoring
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Weighted match results
        """
        # Extract keywords by category
        resume_keywords = self.extract_keywords(resume_text)
        jd_keywords = self.extract_keywords(jd_text)
        
        # Category weights
        weights = {
            'technical_skills': 0.4,
            'programming_languages': 0.3,
            'tools_technologies': 0.2,
            'soft_skills': 0.1
        }
        
        category_results = {}
        overall_score = 0
        
        for category, weight in weights.items():
            if category in resume_keywords and category in jd_keywords:
                # Perform exact match for this category
                exact_result = self.exact_match(
                    resume_keywords[category], 
                    jd_keywords[category]
                )
                
                # Perform fuzzy match for this category
                fuzzy_result = self.fuzzy_match(
                    resume_keywords[category], 
                    jd_keywords[category]
                )
                
                # Combine exact and fuzzy scores
                category_score = (exact_result['match_percentage'] * 0.7 + 
                                fuzzy_result['match_percentage'] * 0.3)
                
                category_results[category] = {
                    'exact_match': exact_result,
                    'fuzzy_match': fuzzy_result,
                    'combined_score': round(category_score, 2),
                    'weight': weight,
                    'weighted_score': round(category_score * weight, 2)
                }
                
                overall_score += category_score * weight
        
        return {
            'category_results': category_results,
            'overall_score': round(overall_score, 2),
            'resume_keywords': resume_keywords,
            'jd_keywords': jd_keywords
        }
    
    def keyword_frequency_analysis(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Analyze keyword frequency and importance
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Frequency analysis results
        """
        resume_keywords = self.extract_keywords(resume_text)['all_keywords']
        jd_keywords = self.extract_keywords(jd_text)['all_keywords']
        
        # Count frequencies
        resume_freq = Counter([kw.lower() for kw in resume_keywords])
        jd_freq = Counter([kw.lower() for kw in jd_keywords])
        
        # Find common keywords with frequencies
        common_keywords = set(resume_freq.keys()).intersection(set(jd_freq.keys()))
        
        keyword_analysis = []
        for keyword in common_keywords:
            analysis = {
                'keyword': keyword,
                'resume_frequency': resume_freq[keyword],
                'jd_frequency': jd_freq[keyword],
                'importance_score': min(resume_freq[keyword], jd_freq[keyword]) * jd_freq[keyword]
            }
            keyword_analysis.append(analysis)
        
        # Sort by importance score
        keyword_analysis.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return {
            'keyword_analysis': keyword_analysis,
            'resume_keyword_count': len(resume_freq),
            'jd_keyword_count': len(jd_freq),
            'common_keyword_count': len(common_keywords),
            'top_keywords': keyword_analysis[:10]
        }