from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import ragas_eval as reval


def test_extract_metrics_ignores_nan_and_keeps_finite_values() -> None:
    class _Scores:
        @staticmethod
        def to_pandas() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "faithfulness": [0.8, np.nan, 0.6],
                    "answer_relevancy": [0.9, 0.7, np.nan],
                    "context_precision": [0.5, 0.4, 0.3],
                    "context_recall": [0.6, np.nan, 0.2],
                }
            )

    metrics, diagnostics = reval._extract_metrics(_Scores())

    assert metrics["faithfulness"] == pytest.approx(0.7)
    assert metrics["answer_relevancy"] == pytest.approx(0.8)
    assert metrics["context_precision"] == pytest.approx(0.4)
    assert metrics["context_recall"] == pytest.approx(0.4)
    assert diagnostics["faithfulness"] == {"total": 3, "valid": 2, "invalid": 1}


def test_extract_metrics_raises_when_metric_has_only_invalid_values() -> None:
    class _Scores:
        @staticmethod
        def to_pandas() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "faithfulness": [np.nan, np.nan],
                    "answer_relevancy": [0.9, 0.7],
                    "context_precision": [0.5, 0.4],
                    "context_recall": [0.6, 0.2],
                }
            )

    with pytest.raises(ValueError, match="faithfulness"):
        reval._extract_metrics(_Scores())


def test_evaluate_golden_set_enforces_exact_sample_count(tmp_path: Path) -> None:
    sample = [
        {
            "query": "Q1",
            "expected_answer": "A1",
            "answer": "A1",
            "contexts": ["ctx"],
        }
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ValueError, match="esperado exatamente 21"):
        reval.evaluate_golden_set(golden_path)


def test_write_json_atomic_rejects_nan(tmp_path: Path) -> None:
    output = tmp_path / "ragas_results.json"

    with pytest.raises(ValueError):
        reval._write_json_atomic(output, {"metrics": {"faithfulness": float("nan")}})


def test_load_dotenv_file_sets_variables_without_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "GOOGLE_API_KEY=from_file\nOTHER_KEY='abc'\n# COMMENT\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_KEY", "from_env")

    reval._load_dotenv_file(dotenv_path)

    assert reval.os.getenv("GOOGLE_API_KEY") == "from_file"
    assert reval.os.getenv("OTHER_KEY") == "from_env"


def test_evaluate_golden_set_uses_offline_fallback_by_default_even_with_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden_set = [
        {
            "query": f"Q{i}",
            "expected_answer": f"A{i}",
            "answer": f"A{i}",
            "contexts": [f"ctx {i}"],
        }
        for i in range(21)
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_set), encoding="utf-8")

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    def _unexpected_live_clients() -> tuple[object, object]:
        raise AssertionError("live Gemini clients should not be created without explicit opt-in")

    monkeypatch.setattr(reval, "_build_gemini_clients", _unexpected_live_clients)

    result = reval.evaluate_golden_set(golden_path)

    assert result["sample_count"] == 21
    assert result["reproducibility"]["metric_backend"] == "deterministic_offline_fallback"
