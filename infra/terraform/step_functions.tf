# ---------------------------------------------------------------------------
# Champion-Challenger Retraining Pipeline — AWS Step Functions
#
# Fluxo completo (MLOps Nível 2, Gap 07):
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  EventBridge (cron diário)                                         │
#   │        │                                                            │
#   │        ▼                                                            │
#   │  RunDriftDetection   ──(exit 0)──► NoRetrainingNeeded (Succeed)    │
#   │        │ (exit≠0)                                                   │
#   │        ▼                                                            │
#   │  EvaluateDriftOutcome                                               │
#   │        │ (Cause contém exitCode 20)                                 │
#   │        ▼                    ▼ (outro erro)                          │
#   │  TrainChallenger      DriftSystemError (Succeed)                   │
#   │  [training task def]                                                │
#   │        │ (sucesso)          │ (falha infra)                         │
#   │        ▼                   ▼                                        │
#   │  EvaluateChampionChallenger  TrainingFailed (Fail)                 │
#   │  [training task def]                                                │
#   │  scripts/evaluate_champion_challenger.py                            │
#   │        │ (exit 0 → promovido)   │ (exit≠0 → abaixo threshold/erro) │
#   │        ▼                        ▼                                   │
#   │  ChallengerPromoted (Succeed)   ChallengerNotPromoted (Succeed)    │
#   └─────────────────────────────────────────────────────────────────────┘
#
# Garantias de isolamento (sem SPOF):
#   - RunDriftDetection usa aws_ecs_task_definition.app com override de
#     comando — tarefa efêmera, não altera o serviço de inferência.
#   - TrainChallenger e EvaluateChampionChallenger usam
#     aws_ecs_task_definition.training — task def dedicada com CPU/memória
#     maiores, sem port mappings, nunca associada ao ECS Service de inferência.
#   - Falhas no pipeline de treinamento resultam em estados terminais
#     Succeed ou Fail que não afetam o serviço em execução.
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
    Comment = "Daily drift detection → champion-challenger retraining pipeline (MLOps Level 2)"
    StartAt = "RunDriftDetection"

    States = {

      # ──────────────────────────────────────────────────────────────────
      # Estado 1: Drift Detection
      # Usa a task de inferência com override de comando para não criar
      # compute dedicado para uma verificação leve.
      # Exit code 20 → retrain necessário; exit 0 → sem drift significativo.
      # ──────────────────────────────────────────────────────────────────
      RunDriftDetection = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.app.arn
          NetworkConfiguration = local.sfn_network_config
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
        # Exit code 0 → sem retrain necessário
        Next = "NoRetrainingNeeded"
        # Exit code 20 (ou outro ≠ 0) → ECS falha o task; Catch encaminha
        Catch = [
          {
            ErrorEquals = ["States.TaskFailed"]
            ResultPath  = "$.drift_error"
            Next        = "EvaluateDriftOutcome"
          }
        ]
      }

      # ──────────────────────────────────────────────────────────────────
      # Estado 2: Interpretação do exit code do drift
      # O campo $.drift_error.Cause é uma string JSON que contém o exitCode
      # do container. Exit code 20 = retrain solicitado pelo scheduler.
      # ──────────────────────────────────────────────────────────────────
      EvaluateDriftOutcome = {
        Type = "Choice"
        Choices = [
          {
            # Padrão de saída: "...\"exitCode\":20..."
            Variable      = "$.drift_error.Cause"
            StringMatches = "*\"exitCode\":20*"
            Next          = "TrainChallenger"
          }
        ]
        # Qualquer outro código de erro (infra, timeout) não deve bloquear inferência
        Default = "DriftSystemError"
      }

      # ──────────────────────────────────────────────────────────────────
      # Estado 3: Treinamento do Challenger
      # Usa aws_ecs_task_definition.training (compute ISOLADO da inferência):
      #   - CPU 2048 / RAM 4096 MiB → suficiente para LSTM 100 épocas
      #   - AUTO_PROMOTE_VALIDATED=false → Step Functions controla a promoção
      #   - O script registra o modelo com alias 'candidate' no MLflow
      # ──────────────────────────────────────────────────────────────────
      TrainChallenger = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.training.arn
          NetworkConfiguration = local.sfn_network_config
          Overrides = {
            ContainerOverrides = [
              {
                Name    = "training"
                Command = ["python", "-u", "src/train_model.py"]
                Environment = [
                  {
                    # Garante que o script NÃO promova automaticamente;
                    # a promoção é responsabilidade do próximo estado
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

      # ──────────────────────────────────────────────────────────────────
      # Estado 4: Avaliação Champion-Challenger e Promoção Condicional
      # Usa aws_ecs_task_definition.training (batch, sem tráfego de inferência).
      # Script: scripts/evaluate_champion_challenger.py
      #   Exit  0 → challenger promovido (melhoria ≥ CHAMPION_MIN_IMPROVEMENT)
      #   Exit 10 → challenger abaixo do threshold; champion mantido
      #   Exit  1 → erro de infraestrutura/API
      # ──────────────────────────────────────────────────────────────────
      EvaluateChampionChallenger = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          LaunchType     = "FARGATE"
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.training.arn
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
        # Exit 0 → task ECS bem-sucedida → promovido
        Next = "ChallengerPromoted"
        # Exit ≠ 0 → ECS falha → Catch para estado de não-promoção
        Catch = [
          {
            ErrorEquals = ["States.TaskFailed"]
            ResultPath  = "$.evaluation_error"
            Next        = "ChallengerNotPromoted"
          }
        ]
      }

      # ──────────────────────────────────────────────────────────────────
      # Estados terminais
      # ──────────────────────────────────────────────────────────────────

      ChallengerPromoted = {
        Type    = "Succeed"
        Comment = "Challenger promovido para alias champion no MLflow Model Registry."
      }

      ChallengerNotPromoted = {
        Type    = "Succeed"
        Comment = "Champion mantido. Challenger registrado como candidate (melhoria abaixo do threshold ou erro de API)."
      }

      TrainingFailed = {
        Type    = "Fail"
        Error   = "TrainingError"
        Cause   = "O treinamento do challenger falhou. Verifique os logs do ECS em /ecs/${local.name_prefix}/training."
      }

      DriftSystemError = {
        Type    = "Succeed"
        Comment = "Erro de infraestrutura na detecção de drift (não relacionado a threshold). Inferência não afetada."
      }

      NoRetrainingNeeded = {
        Type    = "Succeed"
        Comment = "PSI abaixo do threshold de retrain. Champion em produção permanece válido."
      }
    }
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-drift-retrain-sfn"
  })
}

