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
  default     = "cron(0 * * * ? *)"
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
  description = "Email endpoint for SNS notifications of LLM degradation alarms (configure in terraform.tfvars)"
  type        = string
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

# ---------------------------------------------------------------------------
# Monitoring stack (Prometheus + Grafana)
# ---------------------------------------------------------------------------

variable "grafana_admin_password" {
  description = "Admin password for Grafana. Overwrite in terraform.tfvars — never commit the real value."
  type        = string
  sensitive   = true
}

variable "monitoring_task_cpu" {
  description = "Fargate CPU units for the monitoring task (Prometheus + Grafana). 1024 = 1 vCPU."
  type        = number
  default     = 1024
}

variable "monitoring_task_memory" {
  description = "Memory (MiB) for the monitoring task. Must be compatible with monitoring_task_cpu."
  type        = number
  default     = 2048
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

# ---------------------------------------------------------------------------
# GitHub Actions OIDC (CI/CD deploy role)
# ---------------------------------------------------------------------------

variable "agent_llm_model" {
  description = "Bedrock inference profile ARN or model ID for the ReAct agent LLM (configure in terraform.tfvars)"
  type        = string
}

variable "github_repository_owner" {
  description = "GitHub repository owner used in OIDC trust policy"
  type        = string
  default     = "monalice"
}

variable "github_repository_name" {
  description = "GitHub repository name used in OIDC trust policy"
  type        = string
  default     = "Tech-Challenge-5"
}

variable "github_allowed_branches" {
  description = "Git branches allowed to assume the GitHub Actions OIDC role"
  type        = list(string)
  default     = ["main", "develop", "dev"]
}

variable "github_allowed_environments" {
  description = "GitHub Environments allowed to assume the GitHub Actions OIDC role"
  type        = list(string)
  default     = ["dev", "prod"]
}

variable "github_oidc_thumbprint" {
  description = "SHA1 thumbprint for token.actions.githubusercontent.com OIDC provider"
  type        = string
  default     = "6938fd4d98bab03faadb97b34396831e3780aea1"
}
