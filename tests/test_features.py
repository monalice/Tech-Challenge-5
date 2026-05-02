import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src import app
from src.domain.features.technical_features import build_feature_matrix


def _raw_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "Open": pa.Column(float, nullable=False),
            "High": pa.Column(float, nullable=False),
            "Low": pa.Column(float, nullable=False),
            "Close": pa.Column(float, nullable=False),
            "Volume": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        },
        strict=True,
    )


def _features_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "log_return": pa.Column(float, nullable=False),
            "rsi": pa.Column(float, nullable=False, checks=[pa.Check.ge(0), pa.Check.le(1)]),
            "macd_signal": pa.Column(float, nullable=False),
            "bb_pct_b": pa.Column(float, nullable=False, checks=[pa.Check.ge(0), pa.Check.le(1)]),
            "sma_ratio": pa.Column(float, nullable=False),
            "vol_ratio": pa.Column(float, nullable=False, checks=[pa.Check.ge(0), pa.Check.le(10)]),
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


# ---------------------------------------------------------------------------
# Gap 04 — testes explícitos com fixture de dados sintéticos
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture com dados OHLCV sintéticos determinísticos para testes de data quality.

    Gera 200 períodos horários com preços e volumes simulados usando seeds fixas,
    sem dependência de arquivos externos ou APIs.

    Returns:
        DataFrame com colunas ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
        e índice DatetimeIndex UTC, adequado para alimentar ``build_feature_matrix``.
    """
    periods = 200
    index = pd.date_range(
        end=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        periods=periods,
        freq="h",
    )
    rng = np.random.default_rng(seed=42)
    close = 50_000.0 + np.cumsum(rng.normal(0, 200, periods))
    high = close + np.abs(rng.normal(100, 30, periods))
    low = close - np.abs(rng.normal(100, 30, periods))
    open_ = close + rng.normal(0, 50, periods)
    volume = np.abs(rng.normal(300, 60, periods))

    df = pd.DataFrame(
        {
            "Open": open_.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Close": close.astype(float),
            "Volume": volume.astype(float),
        },
        index=index,
    )
    df.index.name = "Datetime"
    return df


def test_schema_contract(sample_data: pd.DataFrame) -> None:
    """Valida que a saída de build_feature_matrix respeita integralmente o schema pandera.

    Garante que todas as colunas esperadas estejam presentes, com os tipos corretos
    e dentro dos intervalos definidos no contrato de dados (schema estrito).

    Args:
        sample_data: Fixture com DataFrame OHLCV sintético determinístico.
    """
    result = build_feature_matrix(sample_data)
    validated = _features_schema().validate(result)

    assert list(validated.columns) == [
        "log_return",
        "rsi",
        "macd_signal",
        "bb_pct_b",
        "sma_ratio",
        "vol_ratio",
    ], "As colunas do DataFrame de features devem corresponder ao contrato definido"
    assert not validated.empty, "O DataFrame validado não deve ser vazio"


def test_no_nulls(sample_data: pd.DataFrame) -> None:
    """Assegura que o DataFrame resultante de build_feature_matrix não contém valores nulos.

    Verifica que nenhuma célula do output é NaN após o processo de transformação,
    incluindo as colunas com indicadores que dependem de janelas de aquecimento.

    Args:
        sample_data: Fixture com DataFrame OHLCV sintético determinístico.
    """
    result = build_feature_matrix(sample_data)

    null_counts = result.isna().sum()
    assert not result.isna().any().any(), (
        f"Colunas com NaN: {null_counts[null_counts > 0].to_dict()}"
    )


def test_row_count_preserved(sample_data: pd.DataFrame) -> None:
    """Confirma que o número de linhas do DataFrame de entrada é estritamente preservado.

    Apenas a primeira linha é descartada pelo ``.diff()`` do ``log_return``; todos os
    demais indicadores preenchem NaN internamente antes do ``dropna()`` final.
    Portanto, ``len(output) == len(input) - 1`` é o comportamento esperado.

    Args:
        sample_data: Fixture com DataFrame OHLCV sintético determinístico.
    """
    n_input = len(sample_data)
    result = build_feature_matrix(sample_data)

    # Apenas 1 linha descartada: o NaN inicial de log_return = np.log(close).diff()
    assert len(result) >= n_input - 1, (
        f"Esperado >= {n_input - 1} linhas, obtido {len(result)}"
    )
