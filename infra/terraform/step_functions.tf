resource "aws_sfn_state_machine" "drift_retrain" {
	name     = "${local.name_prefix}-drift-retrain"
	role_arn = aws_iam_role.sfn_drift_retrain.arn

	definition = jsonencode({
		Comment = "Daily drift detection and conditional retraining workflow"
		StartAt = "RunDriftDetection"
		States = {
			RunDriftDetection = {
				Type     = "Task"
				Resource = "arn:aws:states:::ecs:runTask.sync"
				Parameters = {
					LaunchType     = "FARGATE"
					Cluster        = aws_ecs_cluster.main.arn
					TaskDefinition = aws_ecs_task_definition.app.arn
					NetworkConfiguration = {
						AwsvpcConfiguration = {
							Subnets        = data.aws_subnets.default.ids
							SecurityGroups = [aws_security_group.ecs.id]
							AssignPublicIp = "ENABLED"
						}
					}
					Overrides = {
						ContainerOverrides = [
							{
								Name    = "app"
								Command = ["python", "-u", "scripts/run_drift_scheduler.py"]
								Environment = [
									{
										Name  = "DRIFT_AUTOMATION_API_URL"
										Value = "http://${aws_lb.app.dns_name}"
									},
									{
										Name  = "DRIFT_RETRAIN_ENABLED"
										Value = "false"
									},
									{
										Name  = "DRIFT_AUTOMATION_TICKER"
										Value = var.drift_ticker
									}
								]
							}
						]
					}
				}
				ResultPath = "$.drift"
				Next       = "EvaluateDriftOutcome"
			}
			EvaluateDriftOutcome = {
				Type = "Choice"
				Choices = [
					{
						Variable      = "$.drift.tasks[0].containers[0].exitCode"
						NumericEquals = 20
						Next          = "RunRetraining"
					}
				]
				Default = "NoRetrainingNeeded"
			}
			RunRetraining = {
				Type     = "Task"
				Resource = "arn:aws:states:::ecs:runTask.sync"
				Parameters = {
					LaunchType     = "FARGATE"
					Cluster        = aws_ecs_cluster.main.arn
					TaskDefinition = aws_ecs_task_definition.app.arn
					NetworkConfiguration = {
						AwsvpcConfiguration = {
							Subnets        = data.aws_subnets.default.ids
							SecurityGroups = [aws_security_group.ecs.id]
							AssignPublicIp = "ENABLED"
						}
					}
					Overrides = {
						ContainerOverrides = [
							{
								Name    = "app"
								Command = ["python", "-u", "src/train_model.py"]
							}
						]
					}
				}
				End = true
			}
			NoRetrainingNeeded = {
				Type = "Succeed"
			}
		}
	})

	tags = merge(local.common_tags, {
		Name = "${local.name_prefix}-drift-retrain-sfn"
	})
}
