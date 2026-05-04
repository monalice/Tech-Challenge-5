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
