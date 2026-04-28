from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor


pytestmark = pytest.mark.mlflow_integration


def _is_enabled() -> bool:
    return os.getenv("RUN_MLFLOW_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}


def _is_docker_available() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_mlflow(tracking_uri: str, timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.MlflowClient()
            client.search_experiments(max_results=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def test_mlflow_container_registry_minimal_flow(tmp_path: Path) -> None:
    if not _is_enabled():
        pytest.skip("Defina RUN_MLFLOW_INTEGRATION=1 para executar este teste opcional.")

    if not _is_docker_available():
        pytest.skip("Docker indisponivel neste ambiente.")

    image = os.getenv("MLFLOW_TEST_DOCKER_IMAGE", "ghcr.io/mlflow/mlflow:v2.14.3")
    port = _find_free_port()
    container_name = f"mlflow-it-{uuid.uuid4().hex[:8]}"

    run_command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "-p",
        f"{port}:5000",
        image,
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "/tmp/mlruns",
    ]

    try:
        started = subprocess.run(run_command, check=False, capture_output=True, text=True)
        if started.returncode != 0:
            pytest.skip(
                "Nao foi possivel iniciar container do MLflow. "
                f"Imagem: {image}. Erro: {started.stderr.strip()}"
            )

        tracking_uri = f"http://127.0.0.1:{port}"
        if not _wait_for_mlflow(tracking_uri):
            logs = subprocess.run(
                ["docker", "logs", container_name],
                check=False,
                capture_output=True,
                text=True,
            )
            pytest.skip(
                "Servico MLflow local indisponivel dentro do timeout. "
                f"Logs: {logs.stderr[-400:] or logs.stdout[-400:]}"
            )

        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = f"it-mlflow-registry-{uuid.uuid4().hex[:8]}"
        mlflow.set_experiment(experiment_name)

        run_tags = {
            "model_name": "btc_hourly_forecaster_it",
            "model_version": "v-it",
            "owner": "integration-tests",
            "risk_level": "low",
            "fairness_checked": "false",
        }

        fairness_artifact = tmp_path / "fairness_report.json"
        fairness_artifact.write_text(
            json.dumps({"group_metrics": {"baseline": 0.0}, "status": "mock"}),
            encoding="utf-8",
        )

        with mlflow.start_run(run_name="integration_registry_run") as run:
            mlflow.set_tags(run_tags)
            mlflow.log_metric("mae_price", 1.23)
            mlflow.log_artifact(str(fairness_artifact), artifact_path="fairness")

            X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            model = DummyRegressor(strategy="mean")
            model.fit(X, y)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = run.info.run_id

        model_name = f"btc_hourly_forecaster_it_{uuid.uuid4().hex[:8]}"
        registered = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name,
            tags={
                "owner": "integration-tests",
                "risk_level": "low",
                "integration_test": "true",
            },
        )

        client = mlflow.MlflowClient()
        stored_run = client.get_run(run_id)
        assert stored_run.data.metrics["mae_price"] == pytest.approx(1.23)
        assert stored_run.data.tags["owner"] == "integration-tests"
        assert stored_run.data.tags["risk_level"] == "low"

        fairness_artifacts = client.list_artifacts(run_id, path="fairness")
        fairness_paths = [item.path for item in fairness_artifacts]
        assert "fairness/fairness_report.json" in fairness_paths

        model_version = client.get_model_version(model_name, str(registered.version))
        assert model_version.tags["owner"] == "integration-tests"
        assert model_version.tags["risk_level"] == "low"
        assert model_version.tags["integration_test"] == "true"

    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
