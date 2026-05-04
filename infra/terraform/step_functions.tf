# ---------------------------------------------------------------------------
# Champion-Challenger Orchestration — AWS Step Functions (ASL)
#
# Fluxo de retreino desacoplado (MLOps Nível 2, sem SPOF):
#
#   EventBridge → TrainChallenger(ECS)
#                     ↓
#                 EvaluateChampionChallenger(ECS)
#                    (resolve challenger pelo alias no Model Registry)
#                     ↓
#                 Choice por exit code
#                     ├─ exit 0  -> EvaluationPassed (Success)
#                     ├─ exit 10 -> ModelRejected (Fail)
#                     └─ exit 1  -> EvaluationFailed (Fail)
#
# Observação de integração:
#   - `ecs:runTask.sync` retorna metadados da task ECS (não o stdout do container).
#   - A etapa de avaliação resolve run/version do challenger via alias no Model Registry.
# ---------------------------------------------------------------------------

locals {
  sfn_network_config = {
    AwsvpcConfiguration = {
      Subnets        = data.aws_subnets.default.ids
      SecurityGroups = [aws_security_group.ecs.id]
      AssignPublicIp = "ENABLED"
    }
  }
}

resource "aws_sfn_state_machine" "drift_retrain" {
  name     = "${local.name_prefix}-drift-retrain"
  role_arn = aws_iam_role.sfn_drift_retrain.arn

  definition = jsonencode({
    Comment = "Champion-Challenger retraining workflow with strict evaluation gate"
    StartAt = "TrainChallenger"

    States = {

      TrainChallenger = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType           = "FARGATE"
          Cluster              = aws_ecs_cluster.main.arn
          TaskDefinition       = aws_ecs_task_definition.training.arn
          NetworkConfiguration = local.sfn_network_config
          Overrides = {
            ContainerOverrides = [
              {
                Name    = "training"
                Command = ["python", "-m", "training.train_model"]
                Environment = [
                  {
                    Name  = "MLFLOW_AUTO_PROMOTE_VALIDATED"
                    Value = "false"
                  }
                ]
              }
            ]
          }
        }
        ResultPath = "$.training"
        Next       = "EvaluateChampionChallenger"
        Catch = [
          {
            ErrorEquals = ["States.TaskFailed"]
            ResultPath  = "$.training_error"
            Next        = "TrainingFailed"
          }
        ]
      }

      EvaluateChampionChallenger = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType           = "FARGATE"
          Cluster              = aws_ecs_cluster.main.arn
          TaskDefinition       = aws_ecs_task_definition.training.arn
          NetworkConfiguration = local.sfn_network_config
          Overrides = {
            ContainerOverrides = [
              {
                Name    = "training"
                Command = ["python", "-u", "scripts/evaluate_champion_challenger.py"]
                Environment = [
                  {
                    Name  = "CHAMPION_MIN_IMPROVEMENT"
                    Value = var.champion_min_improvement
                  },
                  {
                    Name  = "MLFLOW_MODEL_NAME"
                    Value = var.mlflow_model_name
                  },
                  {
                    Name  = "MLFLOW_CHAMPION_ALIAS"
                    Value = var.mlflow_champion_alias
                  },
                  {
                    Name  = "MLFLOW_CANDIDATE_ALIAS"
                    Value = var.mlflow_candidate_alias
                  }
                ]
              }
            ]
          }
        }
        ResultPath = "$.evaluation"
        Next       = "EvaluationPassed"
        Catch = [
          {
            ErrorEquals = ["States.TaskFailed"]
            ResultPath  = "$.evaluation_error"
            Next        = "EvaluationDecision"
          }
        ]
      }

      EvaluationDecision = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.evaluation_error.Cause"
            StringMatches = "*\"exitCode\":10*"
            Next          = "ModelRejected"
          },
          {
            Variable      = "$.evaluation_error.Cause"
            StringMatches = "*\"exitCode\":1*"
            Next          = "EvaluationFailed"
          }
        ]
        Default = "EvaluationFailed"
      }

      EvaluationPassed = {
        Type    = "Succeed"
        Comment = "Modelo aprovado no quality gate e promovido pela etapa de avaliação."
      }

      ModelRejected = {
        Type  = "Fail"
        Error = "ModelRejected"
        Cause = "Challenger rejeitado no quality gate (delta_auc abaixo do limiar)."
      }

      TrainingFailed = {
        Type  = "Fail"
        Error = "TrainingError"
        Cause = "O treinamento do challenger falhou. Verifique os logs do ECS em /ecs/${local.name_prefix}/training."
      }

      EvaluationFailed = {
        Type  = "Fail"
        Error = "EvaluationSystemError"
        Cause = "Falha de infraestrutura/API durante avaliação champion-challenger."
      }
    }
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-drift-retrain-sfn"
  })
}

