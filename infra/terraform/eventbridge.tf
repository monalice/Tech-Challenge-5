# ---------------------------------------------------------------------------
# EventBridge cron rule — Avaliação diária de drift com Evidently + PSI
#
# Dispara diariamente às 03:00 UTC a Step Function de drift detection:
#   cron(0 3 * * ? *)  →  todos os dias às 03:00 UTC
#
# Fluxo acionado:
#   EventBridge (cron diário)
#     → aws_sfn_state_machine.drift_retrain
#       → RunDriftDetection (ECS task: run_drift_scheduler.py)
#           → Evidently DataDriftPreset + PSI sobre preço real vs predito
#           → PSI > 0.1 → warning / PSI > 0.2 → TrainChallenger
#
# Métricas publicadas por cada execução:
#   CloudWatch Namespace : MLOps/DriftDetection
#   - PSI_DataDrift       (dim: Ticker)
#   - PSI_PredictionDrift (dim: Ticker)
#   - DriftActionCode     (0=monitor | 1=alert | 2=retrain)
#
#   MLflow Experiment    : btc-hourly-serving
#   - psi_btc_usd, psi_data_drift, psi_prediction_drift, drift_share
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "daily_drift_workflow" {
  name        = "${local.name_prefix}-daily-drift"
  description = "Cron diário (03:00 UTC): dispara avaliação de drift Evidently/PSI + champion-challenger retraining pipeline"

  # Cron AWS EventBridge: cron(<min> <hour> <day> <month> <weekday> <year>)
  # O campo weekday usa "?" quando day-of-month é especificado.
  # Default: todos os dias às 03:00 UTC  (var.drift_schedule_expression = "cron(0 3 * * ? *)")
  schedule_expression = var.drift_schedule_expression

  # Mantém a regra ativa imediatamente após o terraform apply
  state = "ENABLED"

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
