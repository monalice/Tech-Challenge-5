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

output "openai_secret_arn" {
  description = "Secrets Manager ARN for OPENAI_API_KEY"
  value       = aws_secretsmanager_secret.openai_api_key.arn
}

output "db_password_secret_arn" {
  description = "Secrets Manager ARN for DB password"
  value       = aws_secretsmanager_secret.db_password.arn
}
