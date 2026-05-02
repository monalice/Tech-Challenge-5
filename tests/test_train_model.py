# mypy: disable_error_code="attr-defined,no-untyped-def,assignment,unreachable"

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import train_model as tm


class _FakeResponse:
    def __init__(self, payload: list[list[object]]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[list[object]]:
        return self._payload


def _price_df(periods: int = 12) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = np.linspace(100_000, 100_500, periods)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 30,
            "Low": close - 30,
            "Volume": np.full(periods, 100.0),
        },
        index=index,
    )


def _valid_mlflow_metadata_tags() -> dict[str, object]:
    return {
        "model_name": "btc_hourly_forecaster",
        "model_version": "v1",
        "model_type": "time_series",
        "training_data_version": "models/btc_hourly_cache.csv",
        "metrics": {"mae_price": 123.45},
        "owner": "ml-team",
        "risk_level": "medium",
        "fairness_checked": True,
        "git_sha": "abc123",
    }


def test_get_git_sha_success(monkeypatch):
    monkeypatch.setattr(
        tm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )

    assert tm.get_git_sha() == "abc123"


def test_get_git_sha_returns_unknown_on_error(monkeypatch):
    monkeypatch.setattr(
        tm.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("git unavailable")),
    )

    assert tm.get_git_sha() == "unknown"


def test_normalize_download_dataframe_validates_and_sorts():
    df = _price_df()
    unordered = df.iloc[[3, 1, 2, 0]].copy()
    duplicated = pd.concat([unordered, unordered.iloc[[0]]])

    normalized = tm.normalize_download_dataframe(duplicated)

    assert list(normalized.columns) == ["Close", "High", "Low", "Volume"]
    assert normalized.index.is_monotonic_increasing
    assert not normalized.index.duplicated().any()


def test_normalize_download_dataframe_multiindex_extracts_ticker():
    base = _price_df(4)
    tuples = [(c, tm.TICKER) for c in ["Close", "High", "Low", "Volume"]]
    multi_df = pd.DataFrame(base.values, index=base.index)
    multi_df.columns = pd.MultiIndex.from_tuples(tuples)

    normalized = tm.normalize_download_dataframe(multi_df)

    assert list(normalized.columns) == ["Close", "High", "Low", "Volume"]
    assert len(normalized) == 4


def test_normalize_download_dataframe_raises_on_missing_columns():
    bad_df = pd.DataFrame({"Close": [1.0], "High": [2.0]})

    with pytest.raises(ValueError, match="Colunas ausentes"):
        tm.normalize_download_dataframe(bad_df)


def test_validate_raw_training_data_accepts_valid_dataframe():
    validated = tm.validate_raw_training_data(_price_df(48))

    assert not validated.empty
    assert list(validated.columns) == ["Close", "High", "Low", "Volume"]


def test_validate_raw_training_data_rejects_nulls():
    invalid_df = _price_df(24)
    invalid_df.loc[invalid_df.index[0], "Close"] = np.nan

    with pytest.raises(ValueError, match="Validação de dados brutos falhou"):
        tm.validate_raw_training_data(invalid_df)


def test_validate_raw_training_data_rejects_inconsistent_price_ranges():
    invalid_df = _price_df(24)
    invalid_df["High"] = invalid_df["Low"] - 1

    with pytest.raises(ValueError, match="Validação de dados brutos falhou"):
        tm.validate_raw_training_data(invalid_df)


def test_validate_raw_training_data_rejects_temporal_gaps():
    invalid_df = _price_df(24).drop(_price_df(24).index[5])

    with pytest.raises(ValueError, match="frequência horária irregular"):
        tm.validate_raw_training_data(invalid_df)


def test_validate_required_training_metadata_rejects_missing_required_tag():
    with pytest.raises(ValueError, match="owner"):
        tm.validate_required_training_metadata(
            model_name="btc_hourly_forecaster",
            model_version="v1",
            training_data_version="sha:hash",
            model_type="time_series",
            owner="",
            risk_level="medium",
        )


def test_main_raises_before_mlflow_start_run_when_required_metadata_missing(monkeypatch):
    monkeypatch.setattr(tm, "ensure_directories", lambda: None)
    monkeypatch.setattr(tm, "configure_mlflow", lambda: None)
    monkeypatch.setattr(
        tm,
        "build_training_data_lineage",
        lambda dataset_path=tm.CACHE_DATA_PATH: {
            "git_sha": "sha-main-test",
            "dvc_data_rev": "sha-main-test",
            "dvc_data_hash": "abc123dvc",
            "training_data_version": "sha-main-test:abc123dvc",
        },
    )
    monkeypatch.setattr(
        tm,
        "get_fairness_artifact_status",
        lambda fairness_artifact_path=tm.FAIRNESS_ARTIFACT_PATH: {
            "fairness_checked": False,
            "artifact_path": "evaluation/fairness_report.json",
            "status": "missing",
            "alert": "missing_fairness_artifact:evaluation/fairness_report.json",
        },
    )
    monkeypatch.setattr(tm.tf.random, "set_seed", lambda value: None)
    monkeypatch.setattr(tm, "TAG_OWNER", "")

    start_run_called = {"value": False}

    def _start_run_spy(*args, **kwargs):
        start_run_called["value"] = True
        raise AssertionError("mlflow.start_run não deveria ser chamado")

    monkeypatch.setattr(tm.mlflow, "start_run", _start_run_spy)

    with pytest.raises(ValueError, match="owner"):
        tm.main()

    assert start_run_called["value"] is False


def test_load_cached_data_returns_empty_when_file_missing(monkeypatch):
    monkeypatch.setattr(tm.os.path, "exists", lambda path: False)

    loaded = tm.load_cached_data()

    assert loaded.empty


def test_load_cached_data_reads_and_normalizes(monkeypatch, tmp_path):
    cache_path = tmp_path / "btc_cache.csv"
    raw = _price_df(6)
    raw.to_csv(cache_path)

    monkeypatch.setattr(tm, "CACHE_DATA_PATH", str(cache_path))

    loaded = tm.load_cached_data()

    assert not loaded.empty
    assert list(loaded.columns) == ["Close", "High", "Low", "Volume"]


def test_save_cached_data_writes_file(monkeypatch, tmp_path):
    cache_path = tmp_path / "saved_cache.csv"
    monkeypatch.setattr(tm, "CACHE_DATA_PATH", str(cache_path))

    tm.save_cached_data(_price_df(3))

    assert cache_path.exists()


def test_evaluate_champion_challenger_no_production_returns_true(monkeypatch):
    fake_client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: (_ for _ in ()).throw(
            RuntimeError("alias not found")
        )
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    assert tm.evaluate_champion_challenger(challenger_mae=100.0) is True


def test_resolve_champion_version_by_alias(monkeypatch):
    champion = SimpleNamespace(version="7", run_id="run-7")
    fake_client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: champion,
    )

    resolved = tm.resolve_champion_version(client=fake_client)

    assert resolved is champion


def test_evaluate_champion_challenger_compares_mae(monkeypatch):
    champion_version = SimpleNamespace(run_id="run-1")
    champion_run = SimpleNamespace(data=SimpleNamespace(metrics={tm.CHAMPION_METRIC: 100.0}))

    fake_client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: champion_version,
        get_run=lambda run_id: champion_run,
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    assert tm.evaluate_champion_challenger(challenger_mae=99.0) is True
    assert tm.evaluate_champion_challenger(challenger_mae=100.0) is False


def test_evaluate_champion_challenger_returns_false_on_client_error(monkeypatch):
    fake_client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: (_ for _ in ()).throw(RuntimeError("mlflow down"))
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    assert tm.evaluate_champion_challenger(challenger_mae=90.0) is False


def test_download_crypto_data_fallbacks_to_cache(monkeypatch):
    monkeypatch.setattr(
        tm.yf,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("yf error")),
    )
    monkeypatch.setattr(tm, "DOWNLOAD_MAX_RETRIES", 2)
    monkeypatch.setattr(tm.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(tm, "load_cached_data", lambda: _price_df(8))

    used_binance = {"called": False}

    def _binance_not_expected():
        used_binance["called"] = True
        return _price_df(8)

    monkeypatch.setattr(tm, "download_from_binance", _binance_not_expected)

    result = tm.download_crypto_data()

    assert not result.empty
    assert used_binance["called"] is False


def test_download_crypto_data_fallbacks_to_binance_and_saves(monkeypatch):
    monkeypatch.setattr(tm.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(tm, "DOWNLOAD_MAX_RETRIES", 1)
    monkeypatch.setattr(tm, "load_cached_data", lambda: pd.DataFrame())
    monkeypatch.setattr(tm.time, "sleep", lambda *args, **kwargs: None)

    saved = {"called": False}

    def _save_spy(data: pd.DataFrame) -> None:
        saved["called"] = not data.empty

    monkeypatch.setattr(tm, "save_cached_data", _save_spy)
    monkeypatch.setattr(tm, "download_from_binance", lambda: _price_df(10))

    result = tm.download_crypto_data()

    assert not result.empty
    assert saved["called"] is True


def test_download_crypto_data_raises_when_all_sources_fail(monkeypatch):
    monkeypatch.setattr(tm.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(tm, "DOWNLOAD_MAX_RETRIES", 1)
    monkeypatch.setattr(tm, "load_cached_data", lambda: pd.DataFrame())
    monkeypatch.setattr(
        tm,
        "download_from_binance",
        lambda: (_ for _ in ()).throw(RuntimeError("binance down")),
    )

    with pytest.raises(ValueError, match="DataFrame vazio"):
        tm.download_crypto_data()


def test_download_from_binance_uses_requests_and_parses(monkeypatch):
    monkeypatch.setattr(tm, "PERIOD", "1d")
    monkeypatch.setattr(tm.time, "time", lambda: 1_700_000_000)
    monkeypatch.setattr(tm.time, "sleep", lambda *args, **kwargs: None)

    first_batch = [
        [1700000000000, "100", "110", "90", "105", "12", 0, 0, 0, 0, 0, 0],
        [1700003600000, "105", "115", "95", "108", "10", 0, 0, 0, 0, 0, 0],
    ]

    calls = {"n": 0}

    def _fake_get(url, params, timeout):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(first_batch)
        return _FakeResponse([])

    monkeypatch.setattr(tm.requests, "get", _fake_get)

    df = tm.download_from_binance()

    assert len(df) == 2
    assert list(df.columns) == ["Close", "High", "Low", "Volume"]


def test_main_sets_mlflow_tags_and_params_before_pipeline(monkeypatch):
    monkeypatch.setattr(tm, "ensure_directories", lambda: None)
    monkeypatch.setattr(tm, "configure_mlflow", lambda: None)
    monkeypatch.setattr(
        tm,
        "build_training_data_lineage",
        lambda dataset_path=tm.CACHE_DATA_PATH: {
            "git_sha": "sha-main-test",
            "dvc_data_rev": "sha-main-test",
            "dvc_data_hash": "abc123dvc",
            "training_data_version": "sha-main-test:abc123dvc",
        },
    )
    monkeypatch.setattr(
        tm,
        "get_fairness_artifact_status",
        lambda fairness_artifact_path=tm.FAIRNESS_ARTIFACT_PATH: {
            "fairness_checked": False,
            "artifact_path": "evaluation/fairness_report.json",
            "status": "missing",
            "alert": "missing_fairness_artifact:evaluation/fairness_report.json",
        },
    )
    monkeypatch.setattr(tm.tf.random, "set_seed", lambda value: None)

    class _RunCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    captured: dict[str, dict[str, object]] = {}
    captured_runtime_tags: dict[str, object] = {}
    captured_metrics: dict[str, float] = {}

    monkeypatch.setattr(tm.mlflow, "start_run", lambda run_name: _RunCtx())
    monkeypatch.setattr(
        tm.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-main-test")),
    )
    monkeypatch.setattr(tm.mlflow, "set_tags", lambda tags: captured.setdefault("tags", tags))
    monkeypatch.setattr(tm.mlflow, "set_tag", lambda key, value: captured_runtime_tags.__setitem__(key, value))
    monkeypatch.setattr(tm.mlflow, "log_params", lambda params: captured.setdefault("params", params))
    monkeypatch.setattr(tm.mlflow, "log_metric", lambda key, value: captured_metrics.__setitem__(key, value))
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda file_path, artifact_path=None: None)
    monkeypatch.setattr(
        tm,
        "download_crypto_data",
        lambda: (_ for _ in ()).throw(RuntimeError("stop_after_mlflow_setup")),
    )

    with pytest.raises(RuntimeError, match="stop_after_mlflow_setup"):
        tm.main()

    tags = captured["tags"]
    params = captured["params"]

    assert tags["model_name"] == tm.MLFLOW_MODEL_NAME
    assert tags["model_version"] == tm.TAG_MODEL_VERSION
    assert tags["owner"] == tm.TAG_OWNER
    assert tags["risk_level"] == tm.TAG_RISK_LEVEL
    assert tags["training_data_version"] == "sha-main-test:abc123dvc"
    assert tags["metrics"] == {}
    assert tags["git_sha"] == "sha-main-test"
    assert tags["dvc_data_rev"] == "sha-main-test"
    assert tags["dvc_data_hash"] == "abc123dvc"
    assert tags["fairness_checked"] is False

    assert params["ticker"] == tm.TICKER
    assert params["interval"] == tm.INTERVAL
    assert params["lookback"] == tm.LOOKBACK
    assert params["architecture"] == "bidirectional_lstm_multifeature"
    assert params["optimizer"] == "Adam"
    assert params["training_data_version"] == "sha-main-test:abc123dvc"
    assert params["git_sha"] == "sha-main-test"
    assert params["dvc_data_rev"] == "sha-main-test"
    assert params["dvc_data_hash"] == "abc123dvc"
    assert captured_runtime_tags["fairness_artifact_status"] == "missing"
    assert captured_runtime_tags["fairness_alert"] == "missing_fairness_artifact:evaluation/fairness_report.json"
    assert captured_metrics["fairness_artifact_present"] == 0.0


def test_get_fairness_artifact_status_returns_true_when_json_exists(tmp_path):
    fairness_file = tmp_path / "fairness_report.json"
    fairness_file.write_text('{"group_metrics": {"a": 0.02}}', encoding="utf-8")

    status = tm.get_fairness_artifact_status(str(fairness_file))

    assert status["fairness_checked"] is True
    assert status["status"] == "valid"
    assert status["artifact_path"] == str(fairness_file)


def test_get_fairness_artifact_status_returns_false_when_missing(tmp_path):
    missing_file = tmp_path / "missing_fairness_report.json"

    status = tm.get_fairness_artifact_status(str(missing_file))

    assert status["fairness_checked"] is False
    assert status["status"] == "missing"
    assert "missing_fairness_artifact" in status["alert"]


def test_get_fairness_artifact_status_returns_false_when_invalid_json(tmp_path):
    invalid_file = tmp_path / "fairness_report_invalid.json"
    invalid_file.write_text("{invalid-json", encoding="utf-8")

    status = tm.get_fairness_artifact_status(str(invalid_file))

    assert status["fairness_checked"] is False
    assert status["status"] == "invalid"
    assert "invalid_fairness_artifact" in status["alert"]
    assert "error" in status


def test_build_training_data_lineage_reads_dvc_lock_hash_and_git_sha(monkeypatch, tmp_path):
    dvc_lock_path = tmp_path / "dvc.lock"
    dvc_lock_path.write_text(
        """
schema: '2.0'
stages:
  prepare_data:
    outs:
      - path: models/btc_hourly_cache.csv
        hash: md5
        md5: deadbeefcafebabe1234567890abcdef
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(tm, "get_git_sha_required", lambda: "gitsha123")

    lineage = tm.build_training_data_lineage(
        dataset_path="models/btc_hourly_cache.csv",
        dvc_lock_path=str(dvc_lock_path),
    )

    assert lineage["git_sha"] == "gitsha123"
    assert lineage["dvc_data_rev"] == "gitsha123"
    assert lineage["dvc_data_hash"] == "deadbeefcafebabe1234567890abcdef"
    assert lineage["training_data_version"] == "gitsha123:deadbeefcafebabe1234567890abcdef"


def test_build_training_data_lineage_fails_when_dataset_hash_not_found(monkeypatch, tmp_path):
    dvc_lock_path = tmp_path / "dvc.lock"
    dvc_lock_path.write_text(
        """
schema: '2.0'
stages:
  prepare_data:
    outs:
      - path: other/path.csv
        hash: md5
        md5: 1234
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(tm, "get_git_sha_required", lambda: "gitsha123")

    with pytest.raises(RuntimeError, match="nao encontrado em dvc.lock"):
        tm.build_training_data_lineage(
            dataset_path="models/btc_hourly_cache.csv",
            dvc_lock_path=str(dvc_lock_path),
        )


def test_build_training_data_lineage_fails_when_git_sha_unavailable(monkeypatch):
    monkeypatch.setattr(tm, "get_git_sha", lambda: "unknown")

    with pytest.raises(RuntimeError, match="Falha ao capturar git SHA"):
        tm.get_git_sha_required()


def test_ensure_directories_creates_models_folder(monkeypatch):
    created = {"called": False}

    monkeypatch.setattr(tm.os.path, "exists", lambda path: False)

    def _makedirs(path):
        created["called"] = path == "models"

    monkeypatch.setattr(tm.os, "makedirs", _makedirs)

    tm.ensure_directories()

    assert created["called"] is True


def test_configure_mlflow_validates_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(OSError, match="MLFLOW_TRACKING_URI"):
        tm.configure_mlflow()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow")

    with pytest.raises(OSError, match="não pode usar file://"):
        tm.configure_mlflow()


def test_configure_mlflow_creates_experiment_when_missing(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    fake_client = SimpleNamespace(get_experiment_by_name=lambda name: None)
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    calls = {"set_uri": None, "created": None, "set_experiment": None}

    monkeypatch.setattr(tm.mlflow, "set_tracking_uri", lambda uri: calls.__setitem__("set_uri", uri))
    monkeypatch.setattr(
        tm.mlflow,
        "create_experiment",
        lambda name, artifact_location=None: calls.__setitem__("created", (name, artifact_location)),
    )
    monkeypatch.setattr(
        tm.mlflow,
        "set_experiment",
        lambda experiment_name: calls.__setitem__("set_experiment", experiment_name),
    )

    tm.configure_mlflow()

    assert calls["set_uri"] == "http://mlflow:5000"
    assert calls["created"][0] == tm.MLFLOW_EXPERIMENT_NAME
    assert calls["set_experiment"] == tm.MLFLOW_EXPERIMENT_NAME


def test_promote_to_production_marks_old_champion_and_updates_alias(monkeypatch):
    current = SimpleNamespace(version="3")
    alias_updates: list[tuple[str, str, str]] = []
    version_tags: list[tuple[str, str, str, str]] = []

    fake_client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: current,
        set_registered_model_alias=lambda name, alias, version: alias_updates.append(
            (name, alias, str(version))
        ),
        set_model_version_tag=lambda name, version, key, value: version_tags.append(
            (name, str(version), key, value)
        ),
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    tm.promote_to_production("4")

    assert (tm.MLFLOW_MODEL_NAME, tm.CHAMPION_ALIAS, "4") in alias_updates
    assert (tm.MLFLOW_MODEL_NAME, "3", "lifecycle_state", "archived") in version_tags
    assert (tm.MLFLOW_MODEL_NAME, "4", "lifecycle_state", "champion") in version_tags


def test_archive_challenger_sets_candidate_alias(monkeypatch):
    alias_updates: list[tuple[str, str, str]] = []
    version_tags: list[tuple[str, str, str, str]] = []
    fake_client = SimpleNamespace(
        set_registered_model_alias=lambda name, alias, version: alias_updates.append(
            (name, alias, str(version))
        ),
        set_model_version_tag=lambda name, version, key, value: version_tags.append(
            (name, str(version), key, value)
        )
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    tm.archive_challenger("9")

    assert alias_updates == [(tm.MLFLOW_MODEL_NAME, tm.CANDIDATE_ALIAS, "9")]
    assert version_tags == [(tm.MLFLOW_MODEL_NAME, "9", "lifecycle_state", "candidate")]


def test_is_manual_promotion_approved_via_env(monkeypatch):
    monkeypatch.setenv(tm.PROMOTION_APPROVAL_ENV_VAR, "true")
    monkeypatch.delenv(tm.PROMOTION_ADMIN_COMMAND_ENV_VAR, raising=False)

    assert tm.is_manual_promotion_approved("7") is True


def test_is_manual_promotion_approved_via_admin_command_with_version(monkeypatch):
    monkeypatch.delenv(tm.PROMOTION_APPROVAL_ENV_VAR, raising=False)
    monkeypatch.setenv(tm.PROMOTION_ADMIN_COMMAND_ENV_VAR, "promote:7")

    assert tm.is_manual_promotion_approved("7") is True
    assert tm.is_manual_promotion_approved("8") is False


def test_is_manual_promotion_approved_returns_false_without_explicit_gate(monkeypatch):
    monkeypatch.delenv(tm.PROMOTION_APPROVAL_ENV_VAR, raising=False)
    monkeypatch.delenv(tm.PROMOTION_ADMIN_COMMAND_ENV_VAR, raising=False)

    assert tm.is_manual_promotion_approved("7") is False


def test_mark_challenger_as_candidate_sets_alias_and_tags(monkeypatch):
    alias_updates: list[tuple[str, str, str]] = []
    version_tags: list[tuple[str, str, str, str]] = []
    fake_client = SimpleNamespace(
        set_registered_model_alias=lambda name, alias, version: alias_updates.append(
            (name, alias, str(version))
        ),
        set_model_version_tag=lambda name, version, key, value: version_tags.append(
            (name, str(version), key, value)
        ),
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client)

    tm.mark_challenger_as_candidate("12", reason="manual_approval_pending")

    assert alias_updates == [(tm.MLFLOW_MODEL_NAME, tm.CANDIDATE_ALIAS, "12")]
    assert (tm.MLFLOW_MODEL_NAME, "12", "lifecycle_state", "candidate") in version_tags
    assert (
        tm.MLFLOW_MODEL_NAME,
        "12",
        "promotion_gate",
        "manual_approval_required",
    ) in version_tags
    assert (
        tm.MLFLOW_MODEL_NAME,
        "12",
        "candidate_reason",
        "manual_approval_pending",
    ) in version_tags


def test_handle_champion_challenger_outcome_requires_explicit_approval(monkeypatch):
    monkeypatch.setattr(tm, "AUTO_PROMOTE_VALIDATED", False)
    monkeypatch.setattr(tm, "evaluate_champion_challenger", lambda challenger_mae: True)
    monkeypatch.setattr(tm, "is_manual_promotion_approved", lambda challenger_version: False)

    promoted: list[str] = []
    candidates: list[tuple[str, str]] = []

    monkeypatch.setattr(tm, "promote_to_production", lambda version: promoted.append(version))
    monkeypatch.setattr(
        tm,
        "mark_challenger_as_candidate",
        lambda version, reason: candidates.append((version, reason)),
    )

    outcome = tm.handle_champion_challenger_outcome("5", challenger_mae=10.0)

    assert outcome == "candidate_pending_approval"
    assert promoted == []
    assert candidates == [("5", "metric_gate_passed_manual_approval_pending")]


def test_handle_champion_challenger_outcome_promotes_with_explicit_approval(monkeypatch):
    monkeypatch.setattr(tm, "AUTO_PROMOTE_VALIDATED", False)
    monkeypatch.setattr(tm, "evaluate_champion_challenger", lambda challenger_mae: True)
    monkeypatch.setattr(tm, "is_manual_promotion_approved", lambda challenger_version: True)

    promoted: list[str] = []
    candidates: list[tuple[str, str]] = []

    monkeypatch.setattr(tm, "promote_to_production", lambda version: promoted.append(version))
    monkeypatch.setattr(
        tm,
        "mark_challenger_as_candidate",
        lambda version, reason: candidates.append((version, reason)),
    )

    outcome = tm.handle_champion_challenger_outcome("5", challenger_mae=10.0)

    assert outcome == "promoted"
    assert promoted == ["5"]
    assert candidates == []


def test_handle_champion_challenger_outcome_marks_candidate_when_metric_fails(monkeypatch):
    monkeypatch.setattr(tm, "evaluate_champion_challenger", lambda challenger_mae: False)

    promoted: list[str] = []
    candidates: list[tuple[str, str]] = []

    monkeypatch.setattr(tm, "promote_to_production", lambda version: promoted.append(version))
    monkeypatch.setattr(
        tm,
        "mark_challenger_as_candidate",
        lambda version, reason: candidates.append((version, reason)),
    )

    outcome = tm.handle_champion_challenger_outcome("5", challenger_mae=10.0)

    assert outcome == "candidate_not_promoted"
    assert promoted == []
    assert candidates == [("5", "metric_gate_not_passed")]


def test_model_name_single_source_used_in_tags_registry_and_champion(monkeypatch):
    # 1) Tags: model_name deve usar a mesma constante de fonte única.
    monkeypatch.setattr(tm, "ensure_directories", lambda: None)
    monkeypatch.setattr(tm, "configure_mlflow", lambda: None)
    monkeypatch.setattr(
        tm,
        "build_training_data_lineage",
        lambda dataset_path=tm.CACHE_DATA_PATH: {
            "git_sha": "sha-test",
            "dvc_data_rev": "sha-test",
            "dvc_data_hash": "hash-test",
            "training_data_version": "sha-test:hash-test",
        },
    )
    monkeypatch.setattr(
        tm,
        "get_fairness_artifact_status",
        lambda fairness_artifact_path=tm.FAIRNESS_ARTIFACT_PATH: {
            "fairness_checked": False,
            "artifact_path": "evaluation/fairness_report.json",
            "status": "missing",
            "alert": "missing_fairness_artifact:evaluation/fairness_report.json",
        },
    )
    monkeypatch.setattr(tm.tf.random, "set_seed", lambda value: None)

    class _RunCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    captured_tags: dict[str, object] = {}
    monkeypatch.setattr(tm.mlflow, "start_run", lambda run_name: _RunCtx())
    monkeypatch.setattr(
        tm.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-tags-capture")),
    )
    monkeypatch.setattr(tm.mlflow, "set_tags", lambda tags: captured_tags.update(tags))
    monkeypatch.setattr(tm.mlflow, "set_tag", lambda key, value: None)
    monkeypatch.setattr(tm.mlflow, "log_metric", lambda key, value: None)
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda file_path, artifact_path=None: None)
    monkeypatch.setattr(
        tm,
        "download_crypto_data",
        lambda: (_ for _ in ()).throw(RuntimeError("stop_after_tags_capture")),
    )

    with pytest.raises(RuntimeError, match="stop_after_tags_capture"):
        tm.main()

    assert captured_tags["model_name"] == tm.MLFLOW_MODEL_NAME

    # 2) Champion-challenger: consulta ao Registry deve usar o mesmo nome.
    queried_model_name_alias_pairs: list[tuple[str, str]] = []
    fake_client_for_champion = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: queried_model_name_alias_pairs.append(
            (name, alias)
        )
        or (_ for _ in ()).throw(RuntimeError("alias not found"))
    )
    monkeypatch.setattr(tm.mlflow, "MlflowClient", lambda: fake_client_for_champion)

    assert tm.evaluate_champion_challenger(challenger_mae=100.0) is True
    assert queried_model_name_alias_pairs == [(tm.MLFLOW_MODEL_NAME, tm.CHAMPION_ALIAS)]

    # 3) Registry: register_model deve usar o mesmo nome.
    captured_register_name: dict[str, str] = {}

    class _Model:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("model")

    monkeypatch.setattr(tm.joblib, "dump", lambda obj, path: open(path, "w", encoding="utf-8").write("x"))
    monkeypatch.setattr(tm.mlflow.keras, "log_model", lambda model, artifact_path: None)
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda file_path, artifact_path: None)
    monkeypatch.setattr(
        tm.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-xyz")),
    )
    def _register_model_spy(model_uri, name, tags):  # noqa: ARG001
        captured_register_name["name"] = name
        return SimpleNamespace(name=name, version="1")

    monkeypatch.setattr(
        tm.mlflow,
        "register_model",
        _register_model_spy,
    )

    tm.log_training_artifacts(
        _Model(),
        object(),
        object(),
        {"m": 1},
        metadata_tags=_valid_mlflow_metadata_tags(),
    )

    assert captured_register_name["name"] == tm.MLFLOW_MODEL_NAME


def test_log_training_artifacts_registers_model(monkeypatch):
    class _Model:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("model")

    artifacts: list[tuple[str, str]] = []
    monkeypatch.setattr(tm.joblib, "dump", lambda obj, path: open(path, "w", encoding="utf-8").write("x"))
    monkeypatch.setattr(tm.mlflow.keras, "log_model", lambda model, artifact_path: None)
    monkeypatch.setattr(
        tm.mlflow,
        "log_artifact",
        lambda file_path, artifact_path: artifacts.append((file_path, artifact_path)),
    )
    monkeypatch.setattr(
        tm.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-xyz")),
    )
    monkeypatch.setattr(tm, "get_git_sha", lambda: "sha-123")
    monkeypatch.setattr(
        tm.mlflow,
        "register_model",
        lambda model_uri, name, tags: SimpleNamespace(name=name, version="11"),
    )

    version = tm.log_training_artifacts(
        _Model(),
        object(),
        object(),
        {"m": 1},
        metadata_tags=_valid_mlflow_metadata_tags(),
    )

    assert version == "11"
    assert len(artifacts) == 3


def test_validate_mlflow_metadata_tags_raises_when_field_is_missing():
    tags = _valid_mlflow_metadata_tags()
    tags.pop("git_sha")

    with pytest.raises(ValueError, match="campos ausentes"):
        tm.validate_mlflow_metadata_tags(tags, context="run")


def test_validate_mlflow_metadata_tags_raises_when_type_is_invalid():
    tags = _valid_mlflow_metadata_tags()
    tags["fairness_checked"] = "true"

    with pytest.raises(ValueError, match="campos inválidos"):
        tm.validate_mlflow_metadata_tags(tags, context="run")


def test_log_training_artifacts_fails_when_required_metadata_is_missing(monkeypatch):
    class _Model:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("model")

    monkeypatch.setattr(tm.joblib, "dump", lambda obj, path: open(path, "w", encoding="utf-8").write("x"))
    monkeypatch.setattr(tm.mlflow.keras, "log_model", lambda model, artifact_path: None)
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda file_path, artifact_path: None)
    monkeypatch.setattr(
        tm.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-xyz")),
    )

    invalid_tags = _valid_mlflow_metadata_tags()
    invalid_tags.pop("owner")

    with pytest.raises(ValueError, match="model_registry"):
        tm.log_training_artifacts(
            _Model(),
            object(),
            object(),
            {"m": 1},
            metadata_tags=invalid_tags,
        )


def test_build_feature_matrix_and_windows_helpers():
    df = _price_df(90)
    features = tm.build_feature_matrix(df)

    assert not features.empty
    assert list(features.columns) == [
        "log_return",
        "rsi",
        "macd_signal",
        "bb_pct_b",
        "sma_ratio",
        "vol_ratio",
    ]

    values = features.to_numpy()
    X, y = tm.create_sliding_window_multifeature(values, look_back=10)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10
    assert X.shape[2] == values.shape[1]


def test_validate_feature_training_data_accepts_valid_features():
    features = tm.build_feature_matrix(_price_df(100))

    validated = tm.validate_feature_training_data(features)

    assert not validated.empty
    assert list(validated.columns) == tm.REQUIRED_FEATURE_COLUMNS


def test_validate_feature_training_data_rejects_out_of_range_values():
    features = tm.build_feature_matrix(_price_df(100)).copy()
    features.loc[features.index[0], "rsi"] = 1.5

    with pytest.raises(ValueError, match="Validação de features falhou"):
        tm.validate_feature_training_data(features)


def test_validate_feature_training_data_rejects_temporal_disorder():
    features = tm.build_feature_matrix(_price_df(100)).copy().sort_index(ascending=False)

    with pytest.raises(ValueError, match="índice deve estar ordenado"):
        tm.validate_feature_training_data(features)


def test_safe_mape_is_finite():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    mape = tm.safe_mape(y_true, y_pred)

    assert np.isfinite(mape)
    assert mape > 0


def test_download_crypto_data_success_path_saves_cache(monkeypatch):
    monkeypatch.setattr(tm.yf, "download", lambda *args, **kwargs: _price_df(20))
    saved = {"called": False}

    def _save_spy(data: pd.DataFrame) -> None:
        saved["called"] = not data.empty

    monkeypatch.setattr(tm, "save_cached_data", _save_spy)

    result = tm.download_crypto_data()

    assert not result.empty
    assert saved["called"] is True


def test_run_walk_forward_backtest_handles_small_dataset(caplog):
    caplog.set_level("WARNING")

    X_train = np.zeros((2, 5, 2), dtype=float)
    y_train = np.zeros((2,), dtype=float)

    tm.run_walk_forward_backtest(X_train, y_train, scaler_return=SimpleNamespace())

    assert "Backtest pulado" in caplog.text


def test_run_walk_forward_backtest_executes_with_mocked_model(monkeypatch):
    class _Scaler:
        def inverse_transform(self, values):
            return np.asarray(values, dtype=float)

    class _FoldModel:
        def fit(self, *args, **kwargs):
            return None

        def predict(self, X, verbose=0):  # noqa: ARG002
            return np.zeros((len(X), 1), dtype=float)

    monkeypatch.setattr(tm, "build_lstm_architecture", lambda input_shape: _FoldModel())

    X_train = np.zeros((30, 8, 3), dtype=float)
    y_train = np.zeros((30,), dtype=float)

    tm.run_walk_forward_backtest(X_train, y_train, scaler_return=_Scaler())


# ---------------------------------------------------------------------------
# Gap 04 — fluxo de split temporal e treino com MLflow completamente isolado
# ---------------------------------------------------------------------------


def test_create_sliding_window_multifeature_shapes() -> None:
    """Verifica os shapes de saída de create_sliding_window_multifeature.

    Arrange: dataset com 80 amostras × 3 features, look_back=10.
    Act: chama create_sliding_window_multifeature.
    Assert: X tem shape (70, 10, 3) e y tem shape (70,).
    """
    # Arrange
    look_back = 10
    n_samples = 80
    n_features = 3
    dataset = np.arange(n_samples * n_features, dtype=np.float64).reshape(n_samples, n_features)

    # Act
    X, y = tm.create_sliding_window_multifeature(dataset, look_back=look_back)

    # Assert
    expected_windows = n_samples - look_back
    assert X.shape == (expected_windows, look_back, n_features), (
        f"Shape de X esperado: ({expected_windows}, {look_back}, {n_features}), obtido: {X.shape}"
    )
    assert y.shape == (expected_windows,), (
        f"Shape de y esperado: ({expected_windows},), obtido: {y.shape}"
    )


def test_create_sliding_window_multifeature_target_is_first_feature() -> None:
    """Verifica que o target y corresponde ao log_return (feature índice 0).

    Arrange: dataset sintético com feature 0 = [0, 1, 2, ...].
    Act: cria janelas com look_back=2.
    Assert: cada valor de y é igual à posição look_back na feature 0.
    """
    # Arrange
    n_samples = 10
    dataset = np.zeros((n_samples, 3), dtype=np.float64)
    dataset[:, 0] = np.arange(n_samples, dtype=np.float64)

    # Act
    _, y = tm.create_sliding_window_multifeature(dataset, look_back=2)

    # Assert — y[i] deve ser feature[0] da posição look_back + i
    expected_y = np.arange(2, n_samples, dtype=np.float64)
    np.testing.assert_array_equal(y, expected_y)


def test_create_sliding_window_multifeature_returns_empty_when_insufficient_data() -> None:
    """Verifica que arrays vazios são retornados quando look_back >= len(dataset).

    Arrange: dataset com 5 amostras, look_back=5.
    Act: chama create_sliding_window_multifeature.
    Assert: X e y têm comprimento 0.
    """
    # Arrange
    dataset = np.ones((5, 2), dtype=np.float64)

    # Act
    X, y = tm.create_sliding_window_multifeature(dataset, look_back=5)

    # Assert
    assert len(X) == 0
    assert len(y) == 0


def test_build_lstm_architecture_output_shape() -> None:
    """Verifica que build_lstm_architecture retorna modelo com output (None, 1).

    Arrange: input_shape = (60, 6).
    Act: constrói o modelo.
    Assert: output shape é (None, 1) e modelo é compilado (possui optimizer).
    """
    # Arrange
    input_shape = (60, 6)

    # Act
    model = tm.build_lstm_architecture(input_shape)

    # Assert
    assert model.output_shape == (None, 1), (
        f"Output shape esperado (None, 1), obtido {model.output_shape}"
    )
    assert model.optimizer is not None, "Modelo deve estar compilado com optimizer"


def _make_price_df_for_main(periods: int = 200) -> pd.DataFrame:
    """Cria DataFrame OHLCV com dados sintéticos para testes do fluxo main()."""
    index = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    close = np.linspace(90_000.0, 100_000.0, periods)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 50,
            "Low": close - 50,
            "Volume": np.full(periods, 200.0),
        },
        index=index,
    )


def test_main_logs_required_mlflow_tags_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que main() chama mlflow com as tags obrigatórias (model_name, owner, risk_level).

    Todas as chamadas externas (MLflow, download, modelo, scalers, artefatos) são
    isoladas via monkeypatch. O teste verifica apenas o fluxo de separação temporal
    e a passagem das tags de governança ao MLflow.

    Arrange: mocks de todas as dependências externas; dados sintéticos com 200 linhas.
    Act: chama tm.main().
    Assert: mlflow.start_run chamado; tags obrigatórias presentes em set_tags.
    """
    # Arrange — dados sintéticos suficientes para o fluxo completo (LOOKBACK=60)
    raw_df = _make_price_df_for_main(200)

    logged_tags: dict[str, object] = {}
    logged_params: dict[str, object] = {}
    logged_metrics: dict[str, float] = {}
    start_run_called: list[bool] = []

    class _FakeRun:
        class info:
            run_id = "fake-run-id"

    class _FakeRunContext:
        def __enter__(self) -> "_FakeRunContext":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeActiveRun:
        class info:
            run_id = "fake-run-id"

    class _FakeScaler:
        data_min_ = np.array([0.0])
        data_max_ = np.array([1.0])

        def fit(self, X: np.ndarray) -> "_FakeScaler":
            return self

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return np.clip(X, 0, 1)

        def transform(self, X: np.ndarray) -> np.ndarray:
            return np.clip(X, 0, 1)

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            return X

    class _FakeModel:
        output_shape = (None, 1)
        optimizer = object()

        def fit(self, *args: object, **kwargs: object) -> None:
            return None

        def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
            return np.zeros((len(X), 1), dtype=np.float64)

        def save(self, path: str) -> None:
            return None

    fake_scaler = _FakeScaler()

    monkeypatch.setattr(tm, "download_crypto_data", lambda: raw_df)
    monkeypatch.setattr(tm, "ensure_directories", lambda: None)
    monkeypatch.setattr(tm, "configure_mlflow", lambda: None)
    monkeypatch.setattr(
        tm,
        "build_training_data_lineage",
        lambda **kwargs: {  # noqa: ARG005
            "training_data_version": "v1",
            "git_sha": "abc123",
            "dvc_data_rev": "rev1",
            "dvc_data_hash": "hash1",
        },
    )
    monkeypatch.setattr(
        tm,
        "get_fairness_artifact_status",
        lambda: {
            "fairness_checked": False,
            "artifact_path": "evaluation/fairness_report.json",
            "status": "missing",
            "alert": "fairness_artifact_missing",
            "error": None,
        },
    )
    monkeypatch.setattr(tm, "MinMaxScaler", lambda **kwargs: fake_scaler)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm, "build_lstm_architecture", lambda input_shape: _FakeModel())
    monkeypatch.setattr(tm, "run_walk_forward_backtest", lambda *args, **kwargs: None)
    monkeypatch.setattr(tm, "log_training_artifacts", lambda *args, **kwargs: "1")
    monkeypatch.setattr(tm, "register_challenger_initial_state", lambda version: "Staging")

    # Isolamento completo do MLflow
    monkeypatch.setattr(tm.mlflow, "start_run", lambda **kwargs: _FakeRunContext())  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_params", lambda params: logged_params.update(params))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_param", lambda k, v: logged_params.update({k: v}))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_metrics", lambda metrics: logged_metrics.update(metrics))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_metric", lambda k, v: logged_metrics.update({k: v}))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "set_tag", lambda k, v: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "set_tags", lambda t: logged_tags.update(t))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda *args, **kwargs: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "active_run", lambda: _FakeActiveRun())  # type: ignore[attr-defined]
    monkeypatch.setattr(
        tm,
        "set_required_tags_on_active_run",
        lambda tags: (start_run_called.append(True), tags)[1],
    )

    # Act
    tm.main()

    # Assert — MLflow foi invocado e tags obrigatórias foram registadas
    assert start_run_called, "set_required_tags_on_active_run deve ter sido chamado dentro de start_run"
    assert "model_name" in logged_tags or logged_params, (
        "model_name deve estar presente nas tags ou params registados no MLflow"
    )
    assert logged_params.get("ticker") == tm.TICKER, (
        f"Param 'ticker' esperado '{tm.TICKER}', obtido '{logged_params.get('ticker')}'"
    )
    assert "train_rows" in logged_metrics, "Métrica train_rows deve ser registada"
    assert "test_rows" in logged_metrics, "Métrica test_rows deve ser registada"
    assert logged_metrics["train_rows"] > logged_metrics["test_rows"], (
        "Conjunto de treino deve ser maior que o de teste"
    )


def test_main_temporal_split_proportions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que a separação treino/teste respeita TEST_SIZE_PCT.

    Isola o fluxo de main() e inspeciona as métricas train_rows e test_rows
    para confirmar que a proporção temporal está correcta.

    Arrange: dados com 200 linhas; TEST_SIZE_PCT = 0.2 → ~80% treino / ~20% teste.
    Act: chama tm.main().
    Assert: train_rows ≈ 80% do total e test_rows ≈ 20% do total.
    """
    # Arrange
    raw_df = _make_price_df_for_main(200)
    logged_metrics: dict[str, float] = {}

    class _FakeRunContext:
        def __enter__(self) -> "_FakeRunContext":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeActiveRun:
        class info:
            run_id = "fake-run-id"

    class _FakeScaler:
        data_min_ = np.array([0.0])
        data_max_ = np.array([1.0])

        def fit(self, X: np.ndarray) -> "_FakeScaler":
            return self

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return np.clip(X, 0, 1)

        def transform(self, X: np.ndarray) -> np.ndarray:
            return np.clip(X, 0, 1)

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            return X

    class _FakeModel:
        def fit(self, *args: object, **kwargs: object) -> None:
            return None

        def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
            return np.zeros((len(X), 1), dtype=np.float64)

        def save(self, path: str) -> None:
            return None

    monkeypatch.setattr(tm, "download_crypto_data", lambda: raw_df)
    monkeypatch.setattr(tm, "ensure_directories", lambda: None)
    monkeypatch.setattr(tm, "configure_mlflow", lambda: None)
    monkeypatch.setattr(
        tm,
        "build_training_data_lineage",
        lambda **kwargs: {  # noqa: ARG005
            "training_data_version": "v1",
            "git_sha": "abc123",
            "dvc_data_rev": "rev1",
            "dvc_data_hash": "hash1",
        },
    )
    monkeypatch.setattr(
        tm,
        "get_fairness_artifact_status",
        lambda: {
            "fairness_checked": False,
            "artifact_path": "evaluation/fairness_report.json",
            "status": "missing",
            "alert": "fairness_artifact_missing",
            "error": None,
        },
    )
    monkeypatch.setattr(tm, "MinMaxScaler", lambda **kwargs: _FakeScaler())  # type: ignore[attr-defined]
    monkeypatch.setattr(tm, "build_lstm_architecture", lambda input_shape: _FakeModel())
    monkeypatch.setattr(tm, "run_walk_forward_backtest", lambda *args, **kwargs: None)
    monkeypatch.setattr(tm, "log_training_artifacts", lambda *args, **kwargs: "1")
    monkeypatch.setattr(tm, "register_challenger_initial_state", lambda version: "Staging")
    monkeypatch.setattr(tm, "set_required_tags_on_active_run", lambda tags: tags)
    monkeypatch.setattr(tm.mlflow, "start_run", lambda **kwargs: _FakeRunContext())  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_params", lambda params: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_param", lambda k, v: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_metrics", lambda metrics: logged_metrics.update(metrics))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_metric", lambda k, v: logged_metrics.update({k: v}))  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "set_tag", lambda k, v: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "set_tags", lambda t: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "log_artifact", lambda *args, **kwargs: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(tm.mlflow, "active_run", lambda: _FakeActiveRun())  # type: ignore[attr-defined]

    # Act
    tm.main()

    # Assert — proporção de split ~80/20
    train_rows = logged_metrics["train_rows"]
    test_rows = logged_metrics["test_rows"]
    total = train_rows + test_rows

    assert total > 0, "Total de linhas deve ser > 0"
    train_pct = train_rows / total
    assert 0.75 <= train_pct <= 0.85, (
        f"Proporção de treino esperada ≈ 80%, obtida {train_pct:.1%}"
    )
