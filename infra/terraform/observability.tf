resource "aws_sns_topic" "llm_degradation_alerts" {
  name = "${local.name_prefix}-llm-degradation-alerts"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-llm-degradation-alerts"
  })
}

resource "aws_sns_topic_subscription" "llm_degradation_email" {
  topic_arn = aws_sns_topic.llm_degradation_alerts.arn
  protocol  = "email"
  endpoint  = var.llm_alarm_email
}

resource "aws_cloudwatch_metric_alarm" "llm_error_rate_high" {
  alarm_name          = "${local.name_prefix}-llm-error-rate-high"
  alarm_description   = "Triggers when LLM error rate is above 5 percent for 5 minutes"
  namespace           = var.llm_metrics_namespace
  metric_name         = "llm_error_rate"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    Service     = var.project_name
    Environment = var.environment
    Endpoint    = "/chat"
  }

  alarm_actions = [aws_sns_topic.llm_degradation_alerts.arn]
  ok_actions    = [aws_sns_topic.llm_degradation_alerts.arn]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-llm-error-rate-high"
  })
}
