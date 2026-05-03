variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "tech-challenge-5"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "db_name" {
  description = "MLflow metadata PostgreSQL database name"
  type        = string
  default     = "mlflow"
}

variable "db_username" {
  description = "Master username for PostgreSQL"
  type        = string
  default     = "mlflowadmin"
}

variable "db_password" {
  description = "Master password for PostgreSQL (also stored in Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "Allocated storage in GiB"
  type        = number
  default     = 20
}

variable "container_port" {
  description = "Application container port"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task CPU units"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory (MiB)"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Desired ECS service tasks"
  type        = number
  default     = 1
}

variable "container_image_tag" {
  description = "Docker image tag in ECR"
  type        = string
  default     = "latest"
}

variable "health_check_path" {
  description = "Health check path for ALB target group"
  type        = string
  default     = "/health"
}

variable "drift_schedule_expression" {
  description = "EventBridge cron/rate expression for daily drift workflow"
  type        = string
  default     = "cron(0 3 * * ? *)"
}

variable "drift_ticker" {
  description = "Ticker used in drift checks orchestrated by Step Functions"
  type        = string
  default     = "BTC-USD"
}

variable "llm_metrics_namespace" {
  description = "CloudWatch namespace for custom LLM observability metrics"
  type        = string
  default     = "StockCast/LLM"
}

variable "llm_alarm_email" {
  description = "Email endpoint for SNS notifications of LLM degradation alarms"
  type        = string
  default     = "mlops-alerts@example.com"
}

# ---------------------------------------------------------------------------
# Training pipeline (champion-challenger) — isolated compute
# ---------------------------------------------------------------------------

variable "training_task_cpu" {
  description = "Fargate CPU units for the dedicated training task (2 vCPU = 2048)"
  type        = number
  default     = 2048
}

variable "training_task_memory" {
  description = "Memory (MiB) for the dedicated training task"
  type        = number
  default     = 4096
}

variable "training_log_retention_days" {
  description = "CloudWatch log retention in days for training runs"
  type        = number
  default     = 30
}

variable "champion_min_improvement" {
  description = "Minimum relative MAE improvement (0.005 = 0.5%) for challenger promotion"
  type        = string
  default     = "0.005"
}

variable "mlflow_model_name" {
  description = "Registered model name in MLflow Model Registry"
  type        = string
  default     = "btc_hourly_forecaster"
}

variable "mlflow_champion_alias" {
  description = "MLflow alias that identifies the production champion model"
  type        = string
  default     = "champion"
}

variable "mlflow_candidate_alias" {
  description = "MLflow alias assigned to challengers that passed evaluation but await promotion"
  type        = string
  default     = "candidate"
}
