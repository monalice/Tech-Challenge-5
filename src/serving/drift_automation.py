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


def process_drift_result(
    drift_result: dict[str, Any],
    config: DriftAutomationConfig,
    *,
    mlflow_module: Any,
) -> dict[str, Any]:
    """Aplica política operacional de drift: monitoramento, alerta e trigger de retraining."""
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
        "retrain": retrain_result,
    }
