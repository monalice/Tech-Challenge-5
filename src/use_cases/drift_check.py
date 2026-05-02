from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger("stockcast.drift.automation")


# ─── Evidently: relatórios de data drift e prediction drift ──────────────────


def run_evidently_drift_report(
    reference_data: Any,
    current_data: Any,
    *,
    psi_bins: int = 10,
) -> dict[str, Any]:
    """Gera relatórios de data drift e prediction drift usando Evidently com PSI.

    Executa dois relatórios independentes:

    - **Data drift**: compara a distribuição do preço real (coluna ``price``)
      entre o período de referência (treino/validação) e o período atual.
    - **Prediction drift**: compara a distribuição dos erros de predição
      (``price_pred − price``) para detectar degradação do modelo.

    Usa ``DataDriftPreset`` com PSI (Population Stability Index) como
    estatística de teste. Fallback para cálculo manual de PSI por quantis
    quando Evidently não está disponível.

    Args:
        reference_data: ``pandas.DataFrame`` de referência com coluna ``price``
            (janela histórica do período de treino ou validação).
        current_data: ``pandas.DataFrame`` atual com coluna ``price`` (real) e,
            opcionalmente, ``price_pred`` (predito pelo modelo em produção).
        psi_bins: Número de bins para o cálculo manual de PSI (fallback).

    Returns:
        Dicionário com as chaves:

        - ``psi_data`` (float): PSI máximo para data drift.
        - ``psi_prediction`` (float): PSI máximo para prediction drift.
        - ``psi`` (float): ``max(psi_data, psi_prediction)`` — métrica
          consolidada usada nos thresholds de warning (> 0.1) e retrain (> 0.2).
        - ``drift_share`` (float): proporção de colunas com drift (Evidently).
        - ``used_evidently`` (bool): se Evidently foi utilizado com sucesso.
        - ``rows_compared`` (int): número de linhas em ``current_data``.
        - ``status`` (str): ``"ok"``.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    def _psi_manual(ref: Any, cur: Any) -> float:
        ref_s = pd.to_numeric(ref, errors="coerce").dropna()
        cur_s = pd.to_numeric(cur, errors="coerce").dropna()
        if ref_s.empty or cur_s.empty:
            return 0.0
        quantiles = [i / psi_bins for i in range(psi_bins + 1)]
        cut_points = ref_s.quantile(quantiles).drop_duplicates().to_numpy()
        if len(cut_points) < 3:
            return 0.0
        cut_points[0] = -np.inf
        cut_points[-1] = np.inf
        eps = 1e-6
        ref_pct = (
            pd.cut(ref_s, bins=cut_points, include_lowest=True)
            .value_counts(normalize=True, sort=False)
            .add(eps)
        )
        cur_pct = (
            pd.cut(cur_s, bins=cut_points, include_lowest=True)
            .value_counts(normalize=True, sort=False)
            .reindex(ref_pct.index, fill_value=0.0)
            .add(eps)
        )
        return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())

    ref_price: Any = (
        reference_data[["price"]]
        if "price" in reference_data.columns
        else pd.DataFrame({"price": pd.Series(dtype=float)})
    )
    cur_price: Any = (
        current_data[["price"]]
        if "price" in current_data.columns
        else pd.DataFrame({"price": pd.Series(dtype=float)})
    )

    if "price_pred" in current_data.columns and "price" in current_data.columns:
        error_series = (current_data["price_pred"] - current_data["price"]).rename("error")
        ref_error: Any = pd.DataFrame(
            {"error": pd.Series([0.0] * max(len(reference_data), 1), dtype=float)}
        )
        cur_error: Any = pd.DataFrame({"error": error_series.reset_index(drop=True)})
    else:
        ref_error = pd.DataFrame({"error": pd.Series([0.0], dtype=float)})
        cur_error = pd.DataFrame({"error": pd.Series([0.0], dtype=float)})

    psi_data = 0.0
    psi_prediction = 0.0
    drift_share = 0.0
    used_evidently = False

    try:
        try:
            from evidently.report import Report  # noqa: PLC0415
            from evidently.metric_preset import DataDriftPreset  # noqa: PLC0415
        except ImportError:
            from evidently import Report  # type: ignore[no-redef]  # noqa: PLC0415
            from evidently.presets import DataDriftPreset  # type: ignore[no-redef]  # noqa: PLC0415

        def _make_evidently_report() -> Any:
            try:
                from evidently.options import DataDriftOptions  # noqa: PLC0415

                return Report(
                    metrics=[DataDriftPreset()],
                    options=[DataDriftOptions(num_stattest="psi", cat_stattest="psi")],
                )
            except Exception:
                try:
                    return Report(metrics=[DataDriftPreset(stattest="psi")])
                except Exception:
                    return Report(metrics=[DataDriftPreset()])

        def _extract_psi_and_share(report_dict: dict[str, Any]) -> tuple[float, float]:
            max_psi_val = 0.0
            share_val = 0.0
            for metric in report_dict.get("metrics", []):
                result = metric.get("result", {})
                share_val = float(result.get("share_of_drifted_columns", share_val))
                for _, col_data in result.get("drift_by_columns", {}).items():
                    if "psi" in str(col_data.get("stattest_name", "")).lower():
                        score = col_data.get("drift_score")
                        if score is not None:
                            max_psi_val = max(max_psi_val, float(score))
            return max_psi_val, share_val

        if not ref_price.empty and not cur_price.empty:
            rpt_data = _make_evidently_report()
            rpt_data.run(reference_data=ref_price, current_data=cur_price)
            psi_data, drift_share = _extract_psi_and_share(rpt_data.as_dict())

        if not ref_error.empty and not cur_error.empty:
            rpt_pred = _make_evidently_report()
            rpt_pred.run(reference_data=ref_error, current_data=cur_error)
            psi_prediction, _ = _extract_psi_and_share(rpt_pred.as_dict())

        used_evidently = True

    except Exception as exc:
        logger.warning("Evidently indisponível, usando cálculo manual de PSI: %s", exc)
        psi_data = _psi_manual(
            ref_price["price"] if "price" in ref_price.columns else [],
            cur_price["price"] if "price" in cur_price.columns else [],
        )
        psi_prediction = _psi_manual(ref_error["error"], cur_error["error"])

    consolidated_psi = max(psi_data, psi_prediction)
    logger.info(
        json.dumps(
            {
                "event": "evidently_drift_report",
                "psi_data": round(psi_data, 6),
                "psi_prediction": round(psi_prediction, 6),
                "psi_consolidated": round(consolidated_psi, 6),
                "drift_share": round(drift_share, 4),
                "used_evidently": used_evidently,
                "rows_compared": len(current_data),
                "threshold_warning": 0.1,
                "threshold_retrain": 0.2,
            },
            ensure_ascii=False,
        )
    )

    return {
        "status": "ok",
        "psi_data": float(psi_data),
        "psi_prediction": float(psi_prediction),
        "psi": float(consolidated_psi),
        "drift_share": float(drift_share),
        "used_evidently": used_evidently,
        "rows_compared": len(current_data),
    }


@dataclass(frozen=True)
class DriftAutomationConfig:
    psi_warning_threshold: float = 0.1
    psi_retrain_threshold: float = 0.2
    check_interval_hours: int = 24
    alert_webhook_url: str | None = None
    retrain_enabled: bool = False
    retrain_command: str = "python -u src/train_model.py"
    retrain_timeout_seconds: int = 900
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "btc-hourly-serving"
    mlflow_retry_attempts: int = 3
    mlflow_retry_backoff_seconds: float = 1.0
    mlflow_error_log_path: str = "logs/operational/mlflow_drift_errors.jsonl"

    @staticmethod
    def _to_bool(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_sources(
        cls,
        config_path: str = "configs/monitoring_config.yaml",
    ) -> "DriftAutomationConfig":
        warning = 0.1
        retrain = 0.2
        interval = 24

        try:
            import yaml  # type: ignore[import-untyped]

            with open(config_path, encoding="utf-8") as file:
                cfg = yaml.safe_load(file) or {}
            drift_cfg = cfg.get("drift", {}) if isinstance(cfg, dict) else {}
            warning = float(drift_cfg.get("psi_warning_threshold", warning))
            retrain = float(drift_cfg.get("psi_retrain_threshold", retrain))
            interval = int(drift_cfg.get("check_interval_hours", interval))
        except Exception:
            logger.info("Arquivo de monitoramento indisponível. Usando defaults para automação de drift.")

        return cls(
            psi_warning_threshold=float(
                os.getenv("DRIFT_WARNING_THRESHOLD", str(warning))
            ),
            psi_retrain_threshold=float(
                os.getenv("DRIFT_RETRAIN_THRESHOLD", str(retrain))
            ),
            check_interval_hours=int(
                os.getenv("DRIFT_CHECK_INTERVAL_HOURS", str(interval))
            ),
            alert_webhook_url=os.getenv("DRIFT_ALERT_WEBHOOK_URL") or None,
            retrain_enabled=cls._to_bool(os.getenv("DRIFT_RETRAIN_ENABLED"), default=False),
            retrain_command=os.getenv(
                "DRIFT_RETRAIN_COMMAND",
                "python -u src/train_model.py",
            ),
            retrain_timeout_seconds=int(
                os.getenv("DRIFT_RETRAIN_TIMEOUT_SECONDS", "900")
            ),
            mlflow_tracking_uri=os.getenv(
                "SERVING_MLFLOW_TRACKING_URI",
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            ),
            mlflow_experiment_name=os.getenv(
                "SERVING_MLFLOW_EXPERIMENT_NAME",
                os.getenv("MLFLOW_EXPERIMENT_NAME", "btc-hourly-serving"),
            ),
            mlflow_retry_attempts=max(
                1,
                int(os.getenv("DRIFT_MLFLOW_RETRY_ATTEMPTS", "3")),
            ),
            mlflow_retry_backoff_seconds=max(
                0.0,
                float(os.getenv("DRIFT_MLFLOW_RETRY_BACKOFF_SECONDS", "1.0")),
            ),
            mlflow_error_log_path=os.getenv(
                "DRIFT_MLFLOW_ERROR_LOG_PATH",
                "logs/operational/mlflow_drift_errors.jsonl",
            ),
        )


def _persist_operational_error(error_log_path: str, error_payload: dict[str, Any]) -> None:
    """Persiste um payload de erro operacional em arquivo JSONL de forma apêndice.

    Args:
        error_log_path: Caminho absoluto ou relativo para o arquivo ``.jsonl``.
            O diretório pai é criado automaticamente se não existir.
        error_payload: Dicionário serializável em JSON com os detalhes do erro.
    """
    directory = os.path.dirname(error_log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(error_log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(error_payload, ensure_ascii=True) + "\n")


def _build_mlflow_operational_error(
    *,
    stage: str,
    attempt: int,
    max_attempts: int,
    exception: Exception,
    tracking_uri: str,
    experiment_name: str,
) -> dict[str, Any]:
    return {
        "event": "mlflow_drift_logging_failure",
        "stage": stage,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "error_type": type(exception).__name__,
        "error_message": str(exception),
    }


def _configure_mlflow_serving_context(config: DriftAutomationConfig, *, mlflow_module: Any) -> None:
    if not config.mlflow_tracking_uri.strip():
        raise RuntimeError("SERVING_MLFLOW_TRACKING_URI/MLFLOW_TRACKING_URI ausente para serving.")
    if not config.mlflow_experiment_name.strip():
        raise RuntimeError("SERVING_MLFLOW_EXPERIMENT_NAME/MLFLOW_EXPERIMENT_NAME ausente para serving.")

    mlflow_module.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow_module.set_experiment(config.mlflow_experiment_name)


def _log_drift_to_mlflow_with_policy(
    *,
    config: DriftAutomationConfig,
    mlflow_module: Any,
    psi: float,
    action: str,
    alert_sent: bool,
    retrain_result: dict[str, Any],
    drift_result: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    last_error_payload: dict[str, Any] | None = None

    for attempt in range(1, config.mlflow_retry_attempts + 1):
        try:
            _configure_mlflow_serving_context(config, mlflow_module=mlflow_module)
            with mlflow_module.start_run(run_name="drift_automation", nested=True):
                mlflow_module.log_metric("psi_btc_usd", psi)
                mlflow_module.log_metric("psi_data_drift", float(drift_result.get("psi_data", psi)))
                mlflow_module.log_metric("psi_prediction_drift", float(drift_result.get("psi_prediction", psi)))
                mlflow_module.log_metric("drift_share", float(drift_result.get("drift_share", 0.0)))
                mlflow_module.set_tag("used_evidently", str(drift_result.get("used_evidently", False)).lower())
                mlflow_module.set_tag("drift_action", action)
                mlflow_module.set_tag("alert_sent", str(alert_sent).lower())
                mlflow_module.set_tag(
                    "retrain_triggered", str(retrain_result.get("triggered", False)).lower()
                )
                mlflow_module.set_tag(
                    "retrain_success", str(retrain_result.get("success", False)).lower()
                )
                mlflow_module.set_tag(
                    "rows_compared",
                    str(drift_result.get("rows_compared", 0)),
                )
                mlflow_module.set_tag("serving_context", "drift_automation")
            return True, None
        except Exception as exc:
            last_error_payload = _build_mlflow_operational_error(
                stage="mlflow_log",
                attempt=attempt,
                max_attempts=config.mlflow_retry_attempts,
                exception=exc,
                tracking_uri=config.mlflow_tracking_uri,
                experiment_name=config.mlflow_experiment_name,
            )
            logger.error("Erro estruturado de MLflow no serving: %s", json.dumps(last_error_payload))

            if attempt < config.mlflow_retry_attempts:
                backoff_seconds = config.mlflow_retry_backoff_seconds * (2 ** (attempt - 1))
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds)

    if last_error_payload is not None:
        _persist_operational_error(config.mlflow_error_log_path, last_error_payload)
        logger.error(
            "Falha operacional persistida em '%s' apos retries de MLflow.",
            config.mlflow_error_log_path,
        )

    return False, last_error_payload


def _send_alert(
    webhook_url: str,
    payload: dict[str, Any],
) -> tuple[bool, str]:
    """Envia alerta de drift para um webhook HTTP.

    Args:
        webhook_url: URL do endpoint que recebe o payload JSON via POST.
        payload: Dicionário com dados do alerta a serializar como JSON.

    Returns:
        Tupla ``(sucesso, status)`` onde ``sucesso`` é ``True`` se o webhook
        respondeu com HTTP < 400, e ``status`` descreve o resultado ou o
        código de erro.
    """
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code >= 400:
            return False, f"webhook_http_{response.status_code}"
        return True, "sent"
    except Exception as exc:
        return False, f"error:{exc}"


def _trigger_retraining(
    command: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Dispara o pipeline de retreinamento como subprocesso bloqueante.

    Args:
        command: Comando shell a ser executado (ex.: ``"python -u src/train_model.py"``).
        timeout_seconds: Tempo máximo em segundos para aguardar a conclusão.

    Returns:
        Dicionário com as chaves ``triggered``, ``success``, ``exit_code``,
        ``stdout_tail`` (últimos 800 chars) e ``stderr_tail`` (idem).
    """
    args = shlex.split(command)
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "triggered": True,
            "success": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-800:],
            "stderr_tail": completed.stderr[-800:],
        }
    except Exception as exc:
        return {
            "triggered": True,
            "success": False,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


# --- Amazon CloudWatch Metrics ------------------------------------------------


def _publish_drift_metrics_to_cloudwatch(
    *,
    psi_data: float,
    psi_prediction: float,
    action: str,
    ticker: str,
    cloudwatch_namespace: str = "MLOps/DriftDetection",
    aws_region: str | None = None,
) -> bool:
    """Publica metricas de drift no Amazon CloudWatch Metrics via boto3.

    Metricas publicadas no namespace ``MLOps/DriftDetection``:

    - ``PSI_DataDrift``: PSI para distribuicao dos dados de entrada (preco real).
    - ``PSI_PredictionDrift``: PSI para distribuicao dos erros de predicao.
    - ``DriftActionCode``: 0 = monitor_only, 1 = send_alert, 2 = trigger_retrain.

    Todas as metricas usam a dimensao ``Ticker`` (ex.: ``BTC-USD``).

    Args:
        psi_data: PSI calculado para data drift.
        psi_prediction: PSI calculado para prediction drift.
        action: Acao determinada pela politica de drift.
        ticker: Simbolo do ativo usado como dimensao CloudWatch.
        cloudwatch_namespace: Namespace do CloudWatch Metrics.
        aws_region: Regiao AWS. Se ``None``, usa ``AWS_DEFAULT_REGION``.

    Returns:
        ``True`` se a publicacao foi bem-sucedida, ``False`` caso contrario.
    """
    _action_codes: dict[str, float] = {
        "monitor_only": 0.0,
        "send_alert": 1.0,
        "trigger_retrain": 2.0,
    }
    try:
        import boto3  # noqa: PLC0415

        cw_kwargs: dict[str, Any] = {}
        region = aws_region or os.getenv("AWS_DEFAULT_REGION")
        if region:
            cw_kwargs["region_name"] = region

        cw = boto3.client("cloudwatch", **cw_kwargs)
        cw.put_metric_data(
            Namespace=cloudwatch_namespace,
            MetricData=[
                {
                    "MetricName": "PSI_DataDrift",
                    "Dimensions": [{"Name": "Ticker", "Value": ticker}],
                    "Value": psi_data,
                    "Unit": "None",
                },
                {
                    "MetricName": "PSI_PredictionDrift",
                    "Dimensions": [{"Name": "Ticker", "Value": ticker}],
                    "Value": psi_prediction,
                    "Unit": "None",
                },
                {
                    "MetricName": "DriftActionCode",
                    "Dimensions": [{"Name": "Ticker", "Value": ticker}],
                    "Value": _action_codes.get(action, -1.0),
                    "Unit": "None",
                },
            ],
        )
        logger.info(
            "CloudWatch DriftMetrics publicadas: namespace=%s ticker=%s "
            "psi_data=%.4f psi_prediction=%.4f action=%s",
            cloudwatch_namespace,
            ticker,
            psi_data,
            psi_prediction,
            action,
        )
        return True
    except Exception as exc:
        logger.error("Falha ao publicar metricas de drift no CloudWatch: %s", exc)
        return False


def process_drift_result(
    drift_result: dict[str, Any],
    config: DriftAutomationConfig,
    *,
    mlflow_module: Any,
) -> dict[str, Any]:
    """Aplica politica operacional de drift: monitoramento, alerta e trigger de retraining."""
    if drift_result.get("status") != "ok" or "psi" not in drift_result:
        return {
            "action": "none",
            "reason": "drift_not_ok",
            "mlflow_logged": False,
            "alert_sent": False,
            "retrain": {"triggered": False},
        }

    psi = float(drift_result["psi"])
    action = "monitor_only"
    alert_sent = False
    alert_status = "not_required"
    retrain_result: dict[str, Any] = {"triggered": False}

    if psi > config.psi_retrain_threshold:
        action = "trigger_retrain"
        if config.retrain_enabled:
            retrain_result = _trigger_retraining(
                command=config.retrain_command,
                timeout_seconds=config.retrain_timeout_seconds,
            )
        else:
            retrain_result = {
                "triggered": False,
                "success": False,
                "exit_code": None,
                "stdout_tail": "",
                "stderr_tail": "retraining_disabled",
            }
    elif psi > config.psi_warning_threshold:
        action = "send_alert"

    if action in {"send_alert", "trigger_retrain"} and config.alert_webhook_url:
        alert_payload = {
            "event": "drift_threshold_crossed",
            "action": action,
            "psi": psi,
            "warning_threshold": config.psi_warning_threshold,
            "retrain_threshold": config.psi_retrain_threshold,
            "rows_compared": drift_result.get("rows_compared", 0),
            "data_source": drift_result.get("data_source", "unknown"),
        }
        alert_sent, alert_status = _send_alert(config.alert_webhook_url, alert_payload)

    mlflow_logged, mlflow_error = _log_drift_to_mlflow_with_policy(
        config=config,
        mlflow_module=mlflow_module,
        psi=psi,
        action=action,
        alert_sent=alert_sent,
        retrain_result=retrain_result,
        drift_result=drift_result,
    )

    cloudwatch_published = _publish_drift_metrics_to_cloudwatch(
        psi_data=float(drift_result.get("psi_data", psi)),
        psi_prediction=float(drift_result.get("psi_prediction", psi)),
        action=action,
        ticker=str(drift_result.get("ticker", drift_result.get("data_source", "BTC-USD"))),
    )

    return {
        "action": action,
        "psi": psi,
        "warning_threshold": config.psi_warning_threshold,
        "retrain_threshold": config.psi_retrain_threshold,
        "mlflow_logged": mlflow_logged,
        "alert_sent": alert_sent,
        "alert_status": alert_status,
        "mlflow_policy": "retry_with_backoff_then_persist_operational_error",
        "mlflow_error": mlflow_error,
        "cloudwatch_published": cloudwatch_published,
        "retrain": retrain_result,
    }
