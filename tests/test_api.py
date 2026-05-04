import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.delivery.api.dependencies as _dep_module
import src.delivery.api.lifespan as _lifespan_module
from src import app as app_module


class _DummyModel:
    def predict(self, values, verbose=0):
        return np.zeros((len(values), 1), dtype=float)


class _DummyScaler:
    def transform(self, values):
        return values


class _FakeS3Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeS3Client:
    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key.endswith("model_metadata_btc.json"):
            payload = b'{"target":"log_return","lookback":60,"ticker":"BTC-USD"}'
            return {"Body": _FakeS3Body(payload)}
        raise FileNotFoundError(Key)


class _FakeS3Manager:
    def __init__(self):
        self.s3_enabled = True
        self.s3_client = _FakeS3Client()
        self.model_calls: list[str] = []
        self.joblib_calls: list[str] = []

    def _s3_key(self, file_name: str) -> str:
        return f"models/{file_name}"

    def load_model(self, file_name: str):
        self.model_calls.append(file_name)
        return _DummyModel()

    def load_joblib(self, file_name: str):
        self.joblib_calls.append(file_name)
        return _DummyScaler()


def _mock_market_df() -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=72, freq="h", tz="UTC")
    close = np.linspace(100_000, 101_000, 72)
    high = close + 50
    low = close - 50
    volume = np.full(72, 150.0)
    df = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}, index=index)
    df.index.name = "Datetime"
    return df


def test_live_returns_200(monkeypatch):
    monkeypatch.setattr(_lifespan_module, "load_trained_model", lambda _: _DummyModel())
    monkeypatch.setattr(_lifespan_module.joblib, "load", lambda _: _DummyScaler())
    monkeypatch.setattr(
        _dep_module, "download_with_retry", lambda ticker: (_mock_market_df(), "test")
    )

    with TestClient(app_module.app) as client:
        response = client.get("/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alive"
    assert "artifacts_ready" in body


def test_health_returns_200(monkeypatch):
    monkeypatch.setattr(_lifespan_module, "load_trained_model", lambda _: _DummyModel())
    monkeypatch.setattr(_lifespan_module.joblib, "load", lambda _: _DummyScaler())
    monkeypatch.setattr(
        _dep_module, "download_with_retry", lambda ticker: (_mock_market_df(), "test")
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "artifacts_ready" in body
    assert "model_usable" in body
    assert "market_data_accessible" in body


def test_lifespan_loads_champion_artifacts_from_s3(monkeypatch):
    fake_s3_manager = _FakeS3Manager()
    monkeypatch.setattr(_lifespan_module, "S3_MODELS_BUCKET", "fake-bucket")
    monkeypatch.setattr(_lifespan_module, "s3_manager", fake_s3_manager)
    monkeypatch.setattr(
        _dep_module, "download_with_retry", lambda ticker: (_mock_market_df(), "test")
    )

    with TestClient(app_module.app) as client:
        response = client.get("/live")

    assert response.status_code == 200
    assert "champion/lstm_btc_hourly.keras" in fake_s3_manager.model_calls
    assert "champion/scaler_btc.gz" in fake_s3_manager.joblib_calls
    assert "champion/scaler_btc_return.gz" in fake_s3_manager.joblib_calls


def test_lifespan_starts_degraded_when_model_load_fails(monkeypatch):
    monkeypatch.setattr(_lifespan_module, "S3_MODELS_BUCKET", "")
    monkeypatch.setattr(_lifespan_module, "STRICT_ARTIFACT_STARTUP", False)
    monkeypatch.setattr(
        _lifespan_module,
        "load_trained_model",
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with TestClient(app_module.app) as client:
        response = client.get("/live")

    assert response.status_code == 200
    assert response.json()["artifacts_ready"] is False
