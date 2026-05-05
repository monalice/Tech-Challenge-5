# ============================================================================
# Monitoring Stack — Prometheus + Grafana on ECS Fargate
#
# Architecture
# ────────────
#   Single ECS task definition containing four containers:
#
#   1. prometheus-init  (amazon/aws-cli)            ← init, essential=false
#      Downloads prometheus.yml from S3 → ephemeral volume "prometheus-config"
#
#   2. prometheus       (prom/prometheus:v2.54.1)   ← essential=true
#      Reads config from volume; starts only after prometheus-init SUCCESS.
#      Scrapes /metrics via the public ALB DNS (no ECS service discovery needed).
#      Runs on localhost:9090 within the task's shared network namespace.
#
#   3. grafana-init     (amazon/aws-cli)            ← init, essential=false
#      Downloads Grafana provisioning YAML + dashboard JSON from S3.
#
#   4. grafana          (grafana/grafana:10.4.0)    ← essential=true, port 3000
#      Reads provisioning from volume; starts only after grafana-init SUCCESS.
#      Datasource URL = http://localhost:9090 (same task namespace as prometheus).
#
#   Because all containers share a network namespace in an ECS Fargate task,
#   Grafana reaches Prometheus via localhost — no Cloud Map or inter-service
#   routing required.
#
# Configuration files
# ───────────────────
#   Stored as S3 objects in the existing artifacts bucket (managed by Terraform).
#   ECS task role already has s3:GetObject on that bucket (see iam.tf), so the
#   amazon/aws-cli init containers can pull them at task startup.
#
# Exposure
# ────────
#   ALB listener on port 3000  →  Grafana target group  →  container port 3000
#   Prometheus is internal-only (no ALB rule).
# ============================================================================

# ---------------------------------------------------------------------------
# Local values — configuration file contents
# ---------------------------------------------------------------------------

locals {
  monitoring_name = "${local.name_prefix}-monitoring"

  # prometheus.yml — scrapes the app via the existing public ALB
  prometheus_yml = <<-YAML
    global:
      scrape_interval:     15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: stockcast_api
        metrics_path: /metrics
        static_configs:
          - targets:
              - "${aws_lb.app.dns_name}:80"
    YAML

  # Grafana datasource — points to localhost:9090 (same task network ns)
  grafana_datasource_yml = <<-YAML
    apiVersion: 1

    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://localhost:9090
        isDefault: true
        editable: false
    YAML

  # Grafana dashboard provider — tells Grafana to load JSONs from a path
  grafana_dashboard_provider_yml = <<-YAML
    apiVersion: 1

    providers:
      - name: stockcast-dashboards
        orgId: 1
        folder: Stockcast
        type: file
        disableDeletion: true
        updateIntervalSeconds: 30
        options:
          path: /var/lib/grafana/dashboards
    YAML
}

# ---------------------------------------------------------------------------
# S3 — configuration objects (created / updated by Terraform)
# ---------------------------------------------------------------------------

resource "aws_s3_object" "prometheus_config" {
  bucket       = aws_s3_bucket.artifacts.id
  key          = "monitoring/prometheus.yml"
  content      = local.prometheus_yml
  content_type = "text/plain"
  etag         = md5(local.prometheus_yml)

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-prometheus-config" })
}

resource "aws_s3_object" "grafana_datasource" {
  bucket       = aws_s3_bucket.artifacts.id
  key          = "monitoring/grafana/datasources/prometheus.yaml"
  content      = local.grafana_datasource_yml
  content_type = "text/plain"
  etag         = md5(local.grafana_datasource_yml)

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-grafana-datasource" })
}

resource "aws_s3_object" "grafana_dashboard_provider" {
  bucket       = aws_s3_bucket.artifacts.id
  key          = "monitoring/grafana/dashboards/dashboards.yaml"
  content      = local.grafana_dashboard_provider_yml
  content_type = "text/plain"
  etag         = md5(local.grafana_dashboard_provider_yml)

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-grafana-dashboard-provider" })
}

resource "aws_s3_object" "grafana_dashboard_stockcast" {
  bucket       = aws_s3_bucket.artifacts.id
  key          = "monitoring/grafana/dashboards/stockcast.json"
  source       = "${path.root}/../../monitoring/grafana/dashboards/stockcast.json"
  content_type = "application/json"
  etag         = filemd5("${path.root}/../../monitoring/grafana/dashboards/stockcast.json")

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-grafana-dashboard-stockcast" })
}

# ---------------------------------------------------------------------------
# CloudWatch — log group for all monitoring containers
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "monitoring" {
  name              = "/ecs/${local.monitoring_name}"
  retention_in_days = 14

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-logs" })
}

# ---------------------------------------------------------------------------
# ALB — target group + listener for Grafana (port 3000)
# ---------------------------------------------------------------------------

resource "aws_lb_target_group" "grafana" {
  # ALB TG names are limited to 32 characters
  name        = substr("${local.name_prefix}-grafana", 0, 32)
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/api/health"
    protocol            = "HTTP"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
  }

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-grafana-tg" })
}

resource "aws_lb_listener" "grafana" {
  load_balancer_arn = aws_lb.app.arn
  port              = 3000
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.grafana.arn
  }
}

# ---------------------------------------------------------------------------
# ECS Task Definition — multi-container monitoring task
#
# Startup order enforced via dependsOn / condition:
#   prometheus-init (SUCCESS) → prometheus
#   grafana-init    (SUCCESS) → grafana
#
# Ephemeral shared volumes (within-task bind mounts):
#   prometheus-config  → prometheus-init writes, prometheus reads
#   grafana-prov       → grafana-init writes, grafana reads (/provisioning)
#   grafana-dashboards → grafana-init writes, grafana reads (/var/lib/grafana/dashboards)
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "monitoring" {
  family                   = "${local.monitoring_name}-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.monitoring_task_cpu)
  memory                   = tostring(var.monitoring_task_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  # Ephemeral volumes — shared within the task, no EFS required
  volume { name = "prometheus-config" }
  volume { name = "grafana-prov" }
  volume { name = "grafana-dashboards" }

  container_definitions = jsonencode([

    # ── prometheus-init ────────────────────────────────────────────────
    # Downloads prometheus.yml from S3 into the shared prometheus-config
    # volume, then exits with code 0 (triggers SUCCESS condition).
    {
      name      = "prometheus-init"
      image     = "amazon/aws-cli:latest"
      essential = false

      # Override ENTRYPOINT so we can run a multi-step shell script
      entryPoint = ["/bin/sh", "-c"]
      command = [
        join(" && ", [
          "aws s3 cp s3://${aws_s3_bucket.artifacts.id}/${aws_s3_object.prometheus_config.key} /config/prometheus.yml --no-paginate",
          "echo '[prometheus-init] config downloaded OK'"
        ])
      ]

      environment = [
        { name = "AWS_DEFAULT_REGION", value = var.aws_region }
      ]

      mountPoints = [
        { sourceVolume = "prometheus-config", containerPath = "/config", readOnly = false }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.monitoring.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "prometheus-init"
        }
      }
    },

    # ── prometheus ─────────────────────────────────────────────────────
    # Reads config from the shared volume written by prometheus-init.
    # Listens on localhost:9090 (visible to grafana inside the same task).
    {
      name      = "prometheus"
      image     = "prom/prometheus:v2.54.1"
      essential = true

      # Overrides the image's default CMD (ENTRYPOINT stays /bin/prometheus)
      command = [
        "--config.file=/config/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
        "--storage.tsdb.retention.time=7d",
        "--web.console.libraries=/usr/share/prometheus/console_libraries",
        "--web.console.templates=/usr/share/prometheus/consoles"
      ]

      mountPoints = [
        { sourceVolume = "prometheus-config", containerPath = "/config", readOnly = true }
      ]

      dependsOn = [
        { containerName = "prometheus-init", condition = "SUCCESS" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.monitoring.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "prometheus"
        }
      }
    },

    # ── grafana-init ───────────────────────────────────────────────────
    # Downloads all Grafana provisioning YAML files + dashboard JSON from
    # S3 into the shared ephemeral volumes, then exits with code 0.
    {
      name      = "grafana-init"
      image     = "amazon/aws-cli:latest"
      essential = false

      entryPoint = ["/bin/sh", "-c"]
      command = [
        join(" && ", [
          "mkdir -p /provisioning/datasources /provisioning/dashboards /dashboards",
          "aws s3 cp s3://${aws_s3_bucket.artifacts.id}/${aws_s3_object.grafana_datasource.key} /provisioning/datasources/prometheus.yaml --no-paginate",
          "aws s3 cp s3://${aws_s3_bucket.artifacts.id}/${aws_s3_object.grafana_dashboard_provider.key} /provisioning/dashboards/dashboards.yaml --no-paginate",
          "aws s3 cp s3://${aws_s3_bucket.artifacts.id}/${aws_s3_object.grafana_dashboard_stockcast.key} /dashboards/stockcast.json --no-paginate",
          "echo '[grafana-init] provisioning files downloaded OK'"
        ])
      ]

      environment = [
        { name = "AWS_DEFAULT_REGION", value = var.aws_region }
      ]

      mountPoints = [
        { sourceVolume = "grafana-prov",       containerPath = "/provisioning", readOnly = false },
        { sourceVolume = "grafana-dashboards", containerPath = "/dashboards",   readOnly = false }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.monitoring.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "grafana-init"
        }
      }
    },

    # ── grafana ────────────────────────────────────────────────────────
    # Reads datasource + dashboard provisioning from shared volumes.
    # Exposed on port 3000 via the ALB listener.
    {
      name      = "grafana"
      image     = "grafana/grafana:10.4.0"
      essential = true

      portMappings = [
        { containerPort = 3000, hostPort = 3000, protocol = "tcp" }
      ]

      environment = [
        { name = "GF_SECURITY_ADMIN_PASSWORD", value = var.grafana_admin_password },
        { name = "GF_USERS_ALLOW_SIGN_UP",     value = "false" },
        { name = "GF_AUTH_ANONYMOUS_ENABLED",  value = "false" },
        # Tell Grafana to look for provisioning under our mounted volume
        { name = "GF_PATHS_PROVISIONING",      value = "/provisioning" },
        { name = "GF_SERVER_ROOT_URL",         value = "http://%(domain)s:3000/" }
      ]

      mountPoints = [
        # Provisioning YAML files (datasources + dashboard provider config)
        { sourceVolume = "grafana-prov",       containerPath = "/provisioning",               readOnly = true },
        # Dashboard JSON files loaded by the provider above
        { sourceVolume = "grafana-dashboards", containerPath = "/var/lib/grafana/dashboards", readOnly = true }
      ]

      dependsOn = [
        { containerName = "grafana-init", condition = "SUCCESS" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.monitoring.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "grafana"
        }
      }
    }
  ])

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-task" })
}

# ---------------------------------------------------------------------------
# ECS Service — 1 monitoring task, Fargate, same cluster as the API
# ---------------------------------------------------------------------------

resource "aws_ecs_service" "monitoring" {
  name            = "${local.monitoring_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.monitoring.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.monitoring.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.grafana.arn
    container_name   = "grafana"
    container_port   = 3000
  }

  # Allow full replacement during Terraform deploys (monitoring downtime acceptable)
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  # Ensure all config files exist in S3 before the task starts
  depends_on = [
    aws_lb_listener.grafana,
    aws_s3_object.prometheus_config,
    aws_s3_object.grafana_datasource,
    aws_s3_object.grafana_dashboard_provider,
    aws_s3_object.grafana_dashboard_stockcast,
  ]

  tags = merge(local.common_tags, { Name = "${local.monitoring_name}-service" })
}
