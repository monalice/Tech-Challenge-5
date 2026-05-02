import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check

from src import app
from src.features.technical_features import build_feature_matrix


def _raw_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "Open": pa.Column(float, nullable=False),
            "High": pa.Column(float, nullable=False),
            "Low": pa.Column(float, nullable=False),
            "Close": pa.Column(float, nullable=False),
            "Volume": pa.Column(float, nullable=False, checks=Check.ge(0)),
        },
        strict=True,
    )


def _features_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "log_return": pa.Column(float, nullable=False),
            "rsi": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)]),
            "macd_signal": pa.Column(float, nullable=False),
            "bb_pct_b": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)]),
            "sma_ratio": pa.Column(float, nullable=False),
            "vol_ratio": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(10)]),
        },
        strict=True,
    )


def test_validate_raw_market_dataframe_schema(synthetic_yfinance_df: pd.DataFrame):
    validated = _raw_schema().validate(synthetic_yfinance_df)
    assert not validated.empty
    assert list(validated.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_validate_indicator_dataframe_schema(synthetic_yfinance_df: pd.DataFrame):
    raw_df = _raw_schema().validate(synthetic_yfinance_df)

    close_col = raw_df["Close"]
    log_return = np.log(close_col.astype(float)).diff()

    features_df = pd.DataFrame(
        {
            "log_return": log_return,
            "rsi": app._compute_rsi(close_col, 14) / 100.0,
            "macd_signal": app._compute_macd_signal(close_col),
            "bb_pct_b": app._compute_bollinger_pct_b(close_col),
            "sma_ratio": app._compute_sma_ratio(close_col),
            "vol_ratio": app._compute_volume_ratio(raw_df["Volume"]),
        },
        index=raw_df.index,
    )

    features_df = features_df.dropna()

    validated = _features_schema().validate(features_df)
    assert not validated.empty
    assert list(validated.columns) == [
        "log_return",
        "rsi",
        "macd_signal",
        "bb_pct_b",
        "sma_ratio",
        "vol_ratio",
    ]


def test_build_feature_matrix_pandera_no_nulls(synthetic_yfinance_df: pd.DataFrame) -> None:
    """build_feature_matrix deve retornar DataFrame sem nulls validado pelo schema pandera.

    Garante que os seis indicadores técnicos estejam presentes, dentro dos intervalos
    esperados e completamente livres de NaN após a transformação.
    """
    features = build_feature_matrix(synthetic_yfinance_df)
    validated = _features_schema().validate(features)

    assert not validated.empty, "O DataFrame de features não deve ser vazio"
    assert not validated.isna().any().any(), "Não devem existir NaN após build_feature_matrix"
    assert list(validated.columns) == [
        "log_return",
        "rsi",
        "macd_signal",
        "bb_pct_b",
        "sma_ratio",
        "vol_ratio",
    ]


def test_build_feature_matrix_preserves_record_count(synthetic_yfinance_df: pd.DataFrame) -> None:
    """build_feature_matrix deve preservar ao máximo os registros de entrada.

    Apenas os períodos de aquecimento dos indicadores (máx. ~27 linhas pela
    janela MACD lenta de 26 + 1 diff de log_return) são descartados via
    dropna. O restante deve ser mantido integralmente.
    """
    n_input = len(synthetic_yfinance_df)
    features = build_feature_matrix(synthetic_yfinance_df)

    # MACD slow=26 + 1 diff => no máximo 27 linhas de warm-up descartadas
    max_warmup_rows = 30
    assert len(features) >= n_input - max_warmup_rows, (
        f"Esperado >= {n_input - max_warmup_rows} linhas, obtido {len(features)}"
    )
