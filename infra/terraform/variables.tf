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

variable "openai_api_key" {
  description = "OpenAI API key stored in Secrets Manager"
  type        = string
  sensitive   = true
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
