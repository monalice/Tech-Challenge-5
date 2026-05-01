data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name               = "${local.name_prefix}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

  tags = local.common_tags
}

resource "aws_iam_role" "ecs_task_role" {
  name               = "${local.name_prefix}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "ecs_execution_secrets" {
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue"
    ]

    resources = [
      aws_secretsmanager_secret.google_api_key.arn,
      aws_secretsmanager_secret.db_password.arn
    ]
  }
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name   = "${local.name_prefix}-ecs-secrets-policy"
  role   = aws_iam_role.ecs_task_execution_role.id
  policy = data.aws_iam_policy_document.ecs_execution_secrets.json
}

data "aws_iam_policy_document" "ecs_task_bedrock" {
  statement {
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]

    resources = [
      "arn:aws:bedrock:${var.aws_region}::foundation-model/*",
      "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
      "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:application-inference-profile/*"
    ]
  }
}

resource "aws_iam_role_policy" "ecs_task_bedrock" {
  name   = "${local.name_prefix}-ecs-bedrock-policy"
  role   = aws_iam_role.ecs_task_role.id
  policy = data.aws_iam_policy_document.ecs_task_bedrock.json
}
