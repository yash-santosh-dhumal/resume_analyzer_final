"""
Export Utilities
Functions for exporting analysis results to various formats
"""

import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import io

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExportManager:
    """
    Manager for exporting analysis data in various formats
    """
    
    def __init__(self, database_manager):
        """
        Initialize export manager
        
        Args:
            database_manager: DatabaseManager instance
        """
        self.db_manager = database_manager
    
    def export_to_csv(self, output_path: str, job_id: Optional[int] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> str:
        """
        Export analysis results to CSV file
        
        Args:
            output_path: Path for output CSV file
            job_id: Optional job ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Path to created CSV file
        """
        try:
            # Get export data
            export_data = self.db_manager.export_analyses_to_dict(
                job_id=job_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not export_data:
                raise ValueError("No data found for export")
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(export_data)
            
            logger.info(f"CSV export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            raise
    
    def export_to_excel(self, output_path: str, job_id: Optional[int] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> str:
        """
        Export analysis results to Excel file
        
        Args:
            output_path: Path for output Excel file
            job_id: Optional job ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Path to created Excel file
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel export. Install with: pip install pandas openpyxl")
        
        try:
            # Get export data
            export_data = self.db_manager.export_analyses_to_dict(
                job_id=job_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not export_data:
                raise ValueError("No data found for export")
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main analysis sheet
                df.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Summary statistics sheet
                stats = self._create_summary_statistics(df)
                stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Top candidates sheet (if job_id specified)
                if job_id:
                    top_candidates = df.nlargest(10, 'overall_score')
                    top_candidates.to_excel(writer, sheet_name='Top Candidates', index=False)
            
            logger.info(f"Excel export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Excel export failed: {str(e)}")
            raise
    
    def export_to_json(self, output_path: str, job_id: Optional[int] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> str:
        """
        Export analysis results to JSON file
        
        Args:
            output_path: Path for output JSON file
            job_id: Optional job ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Path to created JSON file
        """
        try:
            # Get export data
            export_data = self.db_manager.export_analyses_to_dict(
                job_id=job_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Create export structure
            export_structure = {
                'export_metadata': {
                    'export_date': datetime.utcnow().isoformat(),
                    'total_records': len(export_data),
                    'job_id_filter': job_id,
                    'start_date_filter': start_date.isoformat() if start_date else None,
                    'end_date_filter': end_date.isoformat() if end_date else None
                },
                'analysis_results': export_data,
                'summary_statistics': self._create_summary_statistics_from_data(export_data) if export_data else {}
            }
            
            # Write JSON
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_structure, jsonfile, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            raise
    
    def export_to_string_csv(self, job_id: Optional[int] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> str:
        """
        Export analysis results to CSV string (for web downloads)
        
        Args:
            job_id: Optional job ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            CSV content as string
        """
        try:
            # Get export data
            export_data = self.db_manager.export_analyses_to_dict(
                job_id=job_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not export_data:
                return "No data available for export"
            
            # Create CSV in memory
            output = io.StringIO()
            fieldnames = export_data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(export_data)
            
            csv_content = output.getvalue()
            output.close()
            
            return csv_content
            
        except Exception as e:
            logger.error(f"CSV string export failed: {str(e)}")
            raise
    
    def create_candidate_report(self, analysis_id: int) -> Dict[str, Any]:
        """
        Create detailed report for a specific candidate analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Detailed candidate report
        """
        try:
            # Get analysis with related data
            analysis = self.db_manager.get_analysis(analysis_id)
            if not analysis:
                raise ValueError(f"Analysis not found: {analysis_id}")
            
            resume = self.db_manager.get_resume(analysis.resume_id)
            job_desc = self.db_manager.get_job_description(analysis.job_description_id)
            
            # Create comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'analysis_id': analysis_id
                },
                'candidate_profile': {
                    'name': resume.candidate_name,
                    'email': resume.email,
                    'phone': resume.phone,
                    'experience_years': resume.experience_years,
                    'skills': resume.skills,
                    'education': resume.education,
                    'certifications': resume.certifications,
                    'resume_filename': resume.filename
                },
                'job_details': {
                    'title': job_desc.title,
                    'company': job_desc.company,
                    'department': job_desc.department,
                    'location': job_desc.location,
                    'required_skills': job_desc.required_skills,
                    'preferred_skills': job_desc.preferred_skills
                },
                'analysis_results': {
                    'overall_score': analysis.overall_score,
                    'match_level': analysis.match_level,
                    'confidence': analysis.confidence,
                    'component_scores': {
                        'hard_matching': analysis.hard_matching_score,
                        'soft_matching': analysis.soft_matching_score,
                        'llm_analysis': analysis.llm_analysis_score
                    },
                    'hiring_recommendation': {
                        'decision': analysis.hiring_decision,
                        'confidence': analysis.hiring_confidence,
                        'success_probability': analysis.success_probability
                    }
                },
                'detailed_analysis': {
                    'gap_analysis': analysis.gap_analysis,
                    'personalized_feedback': analysis.personalized_feedback,
                    'recommendations': analysis.recommendations,
                    'risk_factors': analysis.risk_factors,
                    'hard_matching_details': analysis.hard_matching_details,
                    'soft_matching_details': analysis.soft_matching_details,
                    'llm_analysis_details': analysis.llm_analysis_details
                },
                'processing_info': {
                    'analysis_date': analysis.created_at.isoformat() if analysis.created_at else None,
                    'processing_time': analysis.processing_time,
                    'analysis_version': analysis.analysis_version
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create candidate report: {str(e)}")
            raise
    
    def create_job_summary_report(self, job_id: int) -> Dict[str, Any]:
        """
        Create summary report for all candidates for a specific job
        
        Args:
            job_id: Job description ID
            
        Returns:
            Job summary report
        """
        try:
            # Get job details
            job_desc = self.db_manager.get_job_description(job_id)
            if not job_desc:
                raise ValueError(f"Job description not found: {job_id}")
            
            # Get all analyses for this job
            analyses = self.db_manager.get_analyses_for_job(job_id)
            
            # Get top candidates with resume details
            top_candidates = self.db_manager.get_top_candidates(job_id, limit=20)
            
            # Calculate statistics
            scores = [a.overall_score for a in analyses]
            match_levels = [a.match_level for a in analyses]
            hiring_decisions = [a.hiring_decision for a in analyses]
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'job_id': job_id
                },
                'job_details': {
                    'title': job_desc.title,
                    'company': job_desc.company,
                    'department': job_desc.department,
                    'location': job_desc.location,
                    'required_skills': job_desc.required_skills,
                    'preferred_skills': job_desc.preferred_skills,
                    'created_at': job_desc.created_at.isoformat() if job_desc.created_at else None
                },
                'candidate_pool_statistics': {
                    'total_candidates': len(analyses),
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'highest_score': max(scores) if scores else 0,
                    'lowest_score': min(scores) if scores else 0,
                    'match_level_distribution': self._count_occurrences(match_levels),
                    'hiring_decision_distribution': self._count_occurrences(hiring_decisions)
                },
                'top_candidates': [
                    {
                        'rank': i + 1,
                        'candidate_name': candidate['resume']['candidate_name'],
                        'overall_score': candidate['analysis']['overall_score'],
                        'match_level': candidate['analysis']['match_level'],
                        'hiring_decision': candidate['analysis']['hiring_decision'],
                        'key_strengths': candidate['analysis'].get('recommendations', [])[:3],
                        'analysis_id': candidate['analysis']['id']
                    }
                    for i, candidate in enumerate(top_candidates[:10])
                ],
                'hiring_recommendations': {
                    'excellent_candidates': len([a for a in analyses if a.match_level == 'excellent']),
                    'recommended_for_interview': len([a for a in analyses if a.hiring_decision in ['HIRE', 'INTERVIEW']]),
                    'high_potential_candidates': len([a for a in analyses if a.overall_score >= 70]),
                    'top_skills_gap': self._identify_common_gaps(analyses)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create job summary report: {str(e)}")
            raise
    
    def _create_summary_statistics(self, df) -> Dict[str, Any]:
        """Create summary statistics from DataFrame"""
        if df.empty:
            return {}
        
        return {
            'Total Candidates': len(df),
            'Average Score': round(df['overall_score'].mean(), 2),
            'Highest Score': round(df['overall_score'].max(), 2),
            'Lowest Score': round(df['overall_score'].min(), 2),
            'Median Score': round(df['overall_score'].median(), 2),
            'Standard Deviation': round(df['overall_score'].std(), 2),
            'Excellent Matches': len(df[df['match_level'] == 'excellent']),
            'Good Matches': len(df[df['match_level'] == 'good']),
            'Fair Matches': len(df[df['match_level'] == 'fair']),
            'Poor Matches': len(df[df['match_level'] == 'poor']),
            'Recommended for Hire': len(df[df['hiring_decision'] == 'HIRE']),
            'Recommended for Interview': len(df[df['hiring_decision'] == 'INTERVIEW'])
        }
    
    def _create_summary_statistics_from_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from raw data"""
        if not data:
            return {}
        
        scores = [d['overall_score'] for d in data]
        match_levels = [d['match_level'] for d in data]
        hiring_decisions = [d['hiring_decision'] for d in data]
        
        return {
            'total_candidates': len(data),
            'average_score': round(sum(scores) / len(scores), 2),
            'highest_score': round(max(scores), 2),
            'lowest_score': round(min(scores), 2),
            'match_level_distribution': self._count_occurrences(match_levels),
            'hiring_decision_distribution': self._count_occurrences(hiring_decisions)
        }
    
    def _count_occurrences(self, items: List[str]) -> Dict[str, int]:
        """Count occurrences of items in list"""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts
    
    def _identify_common_gaps(self, analyses: List) -> List[str]:
        """Identify common skill gaps across analyses"""
        all_risks = []
        for analysis in analyses:
            if analysis.risk_factors:
                all_risks.extend(analysis.risk_factors)
        
        # Count occurrences and return top gaps
        gap_counts = self._count_occurrences(all_risks)
        sorted_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [gap for gap, count in sorted_gaps[:5]]