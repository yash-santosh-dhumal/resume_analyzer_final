-- Initialize Resume Analyzer Database
-- Create necessary extensions and initial data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial system configuration
INSERT INTO system_configurations (config_key, config_value, config_type, description, is_active) VALUES
    ('scoring_weights_hard', '{"hard_matching": 0.40}', 'scoring', 'Weight for hard matching component', true),
    ('scoring_weights_soft', '{"soft_matching": 0.30}', 'scoring', 'Weight for soft matching component', true),
    ('scoring_weights_llm', '{"llm_analysis": 0.30}', 'scoring', 'Weight for LLM analysis component', true),
    ('score_thresholds', '{"excellent": 80.0, "good": 65.0, "fair": 45.0, "poor": 0.0}', 'scoring', 'Score thresholds for match levels', true),
    ('system_version', '"1.0.0"', 'general', 'Current system version', true),
    ('deployment_date', '"2024-01-15T00:00:00Z"', 'general', 'System deployment date', true)
ON CONFLICT (config_key) DO NOTHING;

-- Create initial audit log entry
INSERT INTO analysis_audit_logs (operation_type, status, timestamp, operation_data) VALUES
    ('system_initialization', 'success', NOW(), '{"message": "System initialized successfully", "version": "1.0.0"}');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_resumes_file_hash ON resumes(file_hash);
CREATE INDEX IF NOT EXISTS idx_resumes_uploaded_at ON resumes(uploaded_at);
CREATE INDEX IF NOT EXISTS idx_job_descriptions_file_hash ON job_descriptions(file_hash);
CREATE INDEX IF NOT EXISTS idx_job_descriptions_created_at ON job_descriptions(created_at);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_overall_score ON resume_analyses(overall_score);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_created_at ON resume_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_resume_id ON resume_analyses(resume_id);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_job_description_id ON resume_analyses(job_description_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON analysis_audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_operation_type ON analysis_audit_logs(operation_type);