from __future__ import annotations

import json
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

pytestmark = pytest.mark.mlflow_integration


def test_mlflow_container_registry_minimal_flow(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    registry_dir = tmp_path / "mlregistry"
    tracking_uri = tracking_dir.resolve().as_uri()
    registry_uri = registry_dir.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
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
