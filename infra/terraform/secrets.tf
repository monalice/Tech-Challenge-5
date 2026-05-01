resource "aws_secretsmanager_secret" "google_api_key" {
  name = "${local.name_prefix}/google-api-key"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-google-secret"
  })
}

resource "aws_secretsmanager_secret_version" "google_api_key" {
  secret_id     = aws_secretsmanager_secret.google_api_key.id
  secret_string = var.google_api_key
}

resource "aws_secretsmanager_secret" "db_password" {
  name = "${local.name_prefix}/db-password"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-password-secret"
  })
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.db_password
}
