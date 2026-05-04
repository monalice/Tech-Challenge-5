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

  statement {
    effect = "Allow"
    actions = [
      "cloudwatch:PutMetricData"
    ]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [var.llm_metrics_namespace]
    }
  }
}

resource "aws_iam_role_policy" "ecs_task_bedrock" {
  name   = "${local.name_prefix}-ecs-bedrock-policy"
  role   = aws_iam_role.ecs_task_role.id
  policy = data.aws_iam_policy_document.ecs_task_bedrock.json
}

# ---------------------------------------------------------------------------
# S3 policy: grants the ECS task role read/write access to the artifacts
# bucket so that MLflow can store model artifacts and the training script
# can upload model files via S3ModelManager.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "ecs_task_s3" {
  statement {
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:DeleteObject"
    ]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name   = "${local.name_prefix}-ecs-s3-policy"
  role   = aws_iam_role.ecs_task_role.id
  policy = data.aws_iam_policy_document.ecs_task_s3.json
}

data "aws_iam_policy_document" "sfn_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "sfn_drift_retrain" {
  name               = "${local.name_prefix}-sfn-drift-retrain-role"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume_role.json

  tags = local.common_tags
}

data "aws_iam_policy_document" "sfn_drift_retrain" {
  statement {
    effect = "Allow"
    actions = [
      "ecs:RunTask"
    ]
    resources = [
      aws_ecs_task_definition.app.arn,
      aws_ecs_task_definition.training.arn
    ]

    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.main.arn]
    }
  }

  statement {
    effect = "Allow"
    actions = [
      "ecs:DescribeTasks",
      "ecs:StopTask"
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "iam:PassRole"
    ]
    resources = [
      aws_iam_role.ecs_task_execution_role.arn,
      aws_iam_role.ecs_task_role.arn
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule"
    ]
    resources = [
      "arn:aws:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"
    ]
  }
}

resource "aws_iam_role_policy" "sfn_drift_retrain" {
  name   = "${local.name_prefix}-sfn-drift-retrain-policy"
  role   = aws_iam_role.sfn_drift_retrain.id
  policy = data.aws_iam_policy_document.sfn_drift_retrain.json
}

data "aws_iam_policy_document" "eventbridge_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "eventbridge_invoke_sfn" {
  name               = "${local.name_prefix}-eventbridge-sfn-role"
  assume_role_policy = data.aws_iam_policy_document.eventbridge_assume_role.json

  tags = local.common_tags
}

data "aws_iam_policy_document" "eventbridge_invoke_sfn" {
  statement {
    effect = "Allow"
    actions = [
      "states:StartExecution"
    ]
    resources = [
      aws_sfn_state_machine.drift_retrain.arn
    ]
  }
}

resource "aws_iam_role_policy" "eventbridge_invoke_sfn" {
  name   = "${local.name_prefix}-eventbridge-sfn-policy"
  role   = aws_iam_role.eventbridge_invoke_sfn.id
  policy = data.aws_iam_policy_document.eventbridge_invoke_sfn.json
}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC deploy role
# ---------------------------------------------------------------------------

resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [var.github_oidc_thumbprint]

  tags = merge(local.common_tags, {
    Name      = "${local.name_prefix}-github-oidc-provider"
    Component = "cicd"
  })
}

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    effect = "Allow"
    actions = [
      "sts:AssumeRoleWithWebIdentity"
    ]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = concat(
        formatlist(
          "repo:%s/%s:ref:refs/heads/%s",
          var.github_repository_owner,
          var.github_repository_name,
          var.github_allowed_branches,
        ),
        formatlist(
          "repo:%s/%s:environment:%s",
          var.github_repository_owner,
          var.github_repository_name,
          var.github_allowed_environments,
        )
      )
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "${local.name_prefix}-github-actions-deploy-role"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json

  tags = merge(local.common_tags, {
    Name      = "${local.name_prefix}-github-actions-deploy-role"
    Component = "cicd"
  })
}

data "aws_iam_policy_document" "github_actions_deploy" {
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken"
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart"
    ]
    resources = [aws_ecr_repository.app.arn]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecs:DescribeServices",
      "ecs:UpdateService"
    ]
    resources = [aws_ecs_service.app.id]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecs:DescribeTaskDefinition"
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecs:ListTasks",
      "ecs:DescribeTasks"
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject"
    ]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:ListBucket"
    ]
    resources = [aws_s3_bucket.artifacts.arn]
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${local.name_prefix}-github-actions-deploy-policy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
