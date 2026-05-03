resource "aws_ecr_repository" "app" {
  name                 = "${var.project_name}/${var.environment}/app"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecr"
  })
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 14

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-logs"
  })
}

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cluster"
  })
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${local.name_prefix}-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = "${aws_ecr_repository.app.repository_url}:${var.container_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "BEDROCK_AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow.address}:5432/${var.db_name}"
        },
        {
          name  = "MLFLOW_ARTIFACT_URI"
          value = "s3://${aws_s3_bucket.artifacts.id}/mlflow-artifacts"
        },
        {
          name  = "CW_LLM_METRICS_ENABLED"
          value = "true"
        },
        {
          name  = "CW_LLM_METRICS_NAMESPACE"
          value = var.llm_metrics_namespace
        },
        {
          name  = "CW_METRIC_SERVICE_NAME"
          value = var.project_name
        },
        {
          name  = "CW_METRIC_ENVIRONMENT"
          value = var.environment
        }
      ]

      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_secretsmanager_secret.db_password.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  tags = local.common_tags
}

resource "aws_ecs_service" "app" {
  name            = "${local.name_prefix}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = var.container_port
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [
    aws_iam_role_policy_attachment.ecs_execution_managed,
    aws_lb_listener.http
  ]

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Dedicated CloudWatch log group for training runs
# Kept separate from inference logs to avoid SPOF and to simplify retention
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "training" {
  name              = "/ecs/${local.name_prefix}/training"
  retention_in_days = var.training_log_retention_days

  tags = merge(local.common_tags, {
    Name      = "${local.name_prefix}-training-logs"
    Component = "training"
  })
}

# ---------------------------------------------------------------------------
# Dedicated ECS Task Definition for champion-challenger training pipeline
#
# Intentionally decoupled from aws_ecs_task_definition.app (inference):
#   - No port mappings    → batch job, never shares ALB/target group
#   - Higher CPU/memory   → LSTM training is compute-intensive
#   - Separate log group  → training noise doesn't flood inference logs
#   - Never attached to aws_ecs_service.app → eliminates SPOF on the
#     inference service if training fails or is slow
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "training" {
  family                   = "${local.name_prefix}-training-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.training_task_cpu)
  memory                   = tostring(var.training_task_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "training"
      image     = "${aws_ecr_repository.app.repository_url}:${var.container_image_tag}"
      essential = true

      # No portMappings: training is a batch workload, never receives traffic

      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow.address}:5432/${var.db_name}"
        },
        {
          name  = "MLFLOW_ARTIFACT_URI"
          value = "s3://${aws_s3_bucket.artifacts.id}/mlflow-artifacts"
        },
        {
          name  = "MLFLOW_MODEL_NAME"
          value = var.mlflow_model_name
        },
        {
          name  = "MLFLOW_CHAMPION_ALIAS"
          value = var.mlflow_champion_alias
        },
        {
          name  = "MLFLOW_CANDIDATE_ALIAS"
          value = var.mlflow_candidate_alias
        },
        {
          name  = "CHAMPION_MIN_IMPROVEMENT"
          value = var.champion_min_improvement
        },
        # Disable auto-promotion: Step Functions controls the promotion gate
        {
          name  = "MLFLOW_AUTO_PROMOTE_VALIDATED"
          value = "false"
        },
        {
          name  = "TF_CPP_MIN_LOG_LEVEL"
          value = "2"
        },
        {
          name  = "TF_ENABLE_ONEDNN_OPTS"
          value = "0"
        }
      ]

      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_secretsmanager_secret.db_password.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.training.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "training"
        }
      }
    }
  ])

  tags = merge(local.common_tags, {
    Name      = "${local.name_prefix}-training-task"
    Component = "training"
  })
}
