output "s3_bucket_name" {
  description = "S3 bucket for DVC and MLflow artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint for MLflow metadata"
  value       = aws_db_instance.mlflow.address
}

output "rds_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.mlflow.port
}

output "pgvector_enable_sql" {
  description = "Run this SQL once after provisioning to enable pgvector"
  value       = "CREATE EXTENSION IF NOT EXISTS vector;"
}

output "ecr_repository_url" {
  description = "ECR repository URL for Docker image push"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer"
  value       = aws_lb.app.dns_name
}

output "drift_retrain_state_machine_arn" {
  description = "ARN of drift-retrain Step Functions state machine"
  value       = aws_sfn_state_machine.drift_retrain.arn
}

output "drift_schedule_rule_name" {
  description = "EventBridge rule that triggers the daily drift workflow"
  value       = aws_cloudwatch_event_rule.daily_drift_workflow.name
}

output "llm_degradation_sns_topic_arn" {
  description = "SNS topic ARN for LLM degradation notifications"
  value       = aws_sns_topic.llm_degradation_alerts.arn
}

output "llm_error_rate_alarm_name" {
  description = "CloudWatch alarm name for LLM error rate"
  value       = aws_cloudwatch_metric_alarm.llm_error_rate_high.alarm_name
}

output "db_password_secret_arn" {
  description = "Secrets Manager ARN for DB password"
  value       = aws_secretsmanager_secret.db_password.arn
}
