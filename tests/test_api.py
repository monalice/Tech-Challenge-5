import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import app as app_module


class _DummyModel:
    def predict(self, values, verbose=0):
        return np.zeros((len(values), 1), dtype=float)


class _DummyScaler:
    def transform(self, values):
        return values


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
    monkeypatch.setattr(app_module, "load_trained_model", lambda _: _DummyModel())
    monkeypatch.setattr(app_module.joblib, "load", lambda _: _DummyScaler())
    monkeypatch.setattr(
        app_module,
        "download_with_retry",
        lambda ticker: (_mock_market_df(), "test"),
    )

    with TestClient(app_module.app) as client:
        response = client.get("/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alive"
    assert "artifacts_ready" in body


def test_health_returns_200(monkeypatch):
    monkeypatch.setattr(app_module, "load_trained_model", lambda _: _DummyModel())
    monkeypatch.setattr(app_module.joblib, "load", lambda _: _DummyScaler())
    monkeypatch.setattr(
        app_module,
        "download_with_retry",
        lambda ticker: (_mock_market_df(), "test"),
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "artifacts_ready" in body
    assert "model_usable" in body
    assert "market_data_accessible" in body
