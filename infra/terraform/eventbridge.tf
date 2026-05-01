resource "aws_cloudwatch_event_rule" "daily_drift_workflow" {
  name                = "${local.name_prefix}-daily-drift"
  description         = "Triggers Step Functions for daily drift check and conditional retraining"
  schedule_expression = var.drift_schedule_expression

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-daily-drift"
  })
}

resource "aws_cloudwatch_event_target" "daily_drift_workflow" {
  rule      = aws_cloudwatch_event_rule.daily_drift_workflow.name
  target_id = "drift-retrain-sfn"
  arn       = aws_sfn_state_machine.drift_retrain.arn
  role_arn  = aws_iam_role.eventbridge_invoke_sfn.arn
}
