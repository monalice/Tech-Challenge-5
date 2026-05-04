from __future__ import annotations

import json
from types import SimpleNamespace

from src.use_cases.drift_check import DriftAutomationConfig, process_drift_result


class _RunCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _MlflowFake:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float]] = []
        self.tags: list[tuple[str, str]] = []
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, run_name: str, nested: bool = False):  # noqa: ARG002
        return _RunCtx()

    def log_metric(self, key: str, value: float) -> None:
        self.metrics.append((key, value))

    def set_tag(self, key: str, value: str) -> None:
        self.tags.append((key, value))


def _base_result(psi: float) -> dict[str, object]:
    return {
        "status": "ok",
        "psi": psi,
        "rows_compared": 20,
        "data_source": "mock",
    }


def test_process_drift_result_monitor_only_logs_mlflow():
    mlflow_fake = _MlflowFake()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        retrain_enabled=False,
    )

    summary = process_drift_result(_base_result(0.05), config, mlflow_module=mlflow_fake)

    assert summary["action"] == "monitor_only"
    assert summary["mlflow_logged"] is True
    assert summary["mlflow_error"] is None
    assert summary["mlflow_policy"] == "retry_with_backoff_then_persist_operational_error"
    assert ("psi_btc_usd", 0.05) in mlflow_fake.metrics
    assert mlflow_fake.tracking_uri == config.mlflow_tracking_uri
    assert mlflow_fake.experiment_name == config.mlflow_experiment_name


def test_process_drift_result_warning_sends_alert(monkeypatch):
    mlflow_fake = _MlflowFake()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        alert_webhook_url="https://example.com/webhook",
        retrain_enabled=False,
    )

    monkeypatch.setattr(
        "src.use_cases.drift_check.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=200),
    )

    summary = process_drift_result(_base_result(0.15), config, mlflow_module=mlflow_fake)

    assert summary["action"] == "send_alert"
    assert summary["alert_sent"] is True


def test_process_drift_result_retrain_threshold_disabled():
    mlflow_fake = _MlflowFake()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        retrain_enabled=False,
    )

    summary = process_drift_result(_base_result(0.25), config, mlflow_module=mlflow_fake)

    assert summary["action"] == "trigger_retrain"
    retrain = summary["retrain"]
    assert retrain["triggered"] is False
    assert retrain["stderr_tail"] == "retraining_disabled"


def test_process_drift_result_retrain_enabled(monkeypatch):
    mlflow_fake = _MlflowFake()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        retrain_enabled=True,
        retrain_command="python -u training/train_model.py",
    )

    monkeypatch.setattr(
        "src.use_cases.drift_check.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="ok",
            stderr="",
        ),
    )

    summary = process_drift_result(_base_result(0.22), config, mlflow_module=mlflow_fake)

    assert summary["action"] == "trigger_retrain"
    assert summary["retrain"]["triggered"] is True
    assert summary["retrain"]["success"] is True


def test_process_drift_result_non_ok_no_action():
    mlflow_fake = _MlflowFake()
    config = DriftAutomationConfig()

    summary = process_drift_result(
        {"status": "skipped", "reason": "missing_data"},
        config,
        mlflow_module=mlflow_fake,
    )

    assert summary["action"] == "none"
    assert summary["mlflow_logged"] is False


def test_process_drift_result_retries_mlflow_and_succeeds(monkeypatch):
    class _MlflowFlaky(_MlflowFake):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def start_run(self, run_name: str, nested: bool = False):  # noqa: ARG002
            self.attempts += 1
            if self.attempts < 2:
                raise RuntimeError("mlflow temporary failure")
            return _RunCtx()

    mlflow_fake = _MlflowFlaky()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        mlflow_retry_attempts=3,
        mlflow_retry_backoff_seconds=0.0,
    )

    monkeypatch.setattr("src.use_cases.drift_check.time.sleep", lambda *_args, **_kwargs: None)

    summary = process_drift_result(_base_result(0.05), config, mlflow_module=mlflow_fake)

    assert summary["mlflow_logged"] is True
    assert summary["mlflow_error"] is None
    assert mlflow_fake.attempts == 2


def test_process_drift_result_persists_operational_error_when_mlflow_keeps_failing(
    monkeypatch,
    tmp_path,
):
    class _MlflowAlwaysFail(_MlflowFake):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def start_run(self, run_name: str, nested: bool = False):  # noqa: ARG002
            self.attempts += 1
            raise RuntimeError("mlflow down")

    error_log_path = tmp_path / "mlflow_drift_errors.jsonl"
    mlflow_fake = _MlflowAlwaysFail()
    config = DriftAutomationConfig(
        psi_warning_threshold=0.1,
        psi_retrain_threshold=0.2,
        mlflow_retry_attempts=2,
        mlflow_retry_backoff_seconds=0.0,
        mlflow_error_log_path=str(error_log_path),
    )

    monkeypatch.setattr("src.use_cases.drift_check.time.sleep", lambda *_args, **_kwargs: None)

    summary = process_drift_result(_base_result(0.05), config, mlflow_module=mlflow_fake)

    assert summary["mlflow_logged"] is False
    assert summary["mlflow_error"] is not None
    assert summary["mlflow_error"]["event"] == "mlflow_drift_logging_failure"
    assert summary["mlflow_error"]["max_attempts"] == 2
    assert mlflow_fake.attempts == 2
    assert error_log_path.exists()

    lines = error_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["event"] == "mlflow_drift_logging_failure"
    assert persisted["error_type"] == "RuntimeError"
