"""Constantes de domínio compartilhadas entre todos os módulos do StockCast.

Fonte única de verdade para valores que aparecem em múltiplos módulos.
Importar daqui elimina a duplicação identificada na auditoria (M1, O1).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Modelo / janela temporal
# ---------------------------------------------------------------------------

#: Número de candles horários usados como janela de entrada do LSTM.
LOOKBACK: int = 60

# ---------------------------------------------------------------------------
# Ativo suportado
# ---------------------------------------------------------------------------

#: Único ticker aceito pela API e pelo agente.
SUPPORTED_TICKER: str = "BTC-USD"

#: Alias de compatibilidade para módulos legados que usam TICKER.
TICKER: str = SUPPORTED_TICKER

# ---------------------------------------------------------------------------
# Fuso horário
# ---------------------------------------------------------------------------

#: Fuso horário de Brasília, usado nas respostas formatadas.
BRASILIA_TZ: str = "America/Sao_Paulo"

# ---------------------------------------------------------------------------
# Binance REST API
# ---------------------------------------------------------------------------

#: Símbolo do par no formato Binance.
BINANCE_SYMBOL: str = "BTCUSDT"

#: URL base do endpoint de klines da Binance.
BINANCE_API_URL: str = "https://api.binance.com/api/v3/klines"

#: Timeout (segundos) para chamadas HTTP à Binance.
BINANCE_TIMEOUT_SECONDS: int = 10

# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

#: Timeout (segundos) para chamadas ao yfinance.
YFINANCE_TIMEOUT_SECONDS: int = 10

# ---------------------------------------------------------------------------
# Paths de artefatos do modelo
# ---------------------------------------------------------------------------

#: Caminho do modelo principal serializado (Keras).
#: Pode ser local (models/...) ou S3 (s3://bucket/...).
MODEL_PATH: str = "models/lstm_btc_hourly.keras"

#: Caminho do scaler principal das features.
SCALER_PATH: str = "models/scaler_btc.gz"

#: Caminho do scaler específico do alvo (log_return).
SCALER_RETURN_PATH: str = "models/scaler_btc_return.gz"

#: Caminho do arquivo JSON com metadados do modelo.
MODEL_META_PATH: str = "models/model_metadata_btc.json"

# ---------------------------------------------------------------------------
# S3 para artefatos de modelo
# ---------------------------------------------------------------------------

#: Bucket S3 para armazenar modelos treinados.
#: Se vazio/None, o sistema usa fallback local (models/).
#: Define via variável de ambiente S3_MODELS_BUCKET.
S3_MODELS_BUCKET: str | None = None  # Será carregado de env() em lifespan

#: Prefixo de caminho dentro do bucket S3.
S3_MODELS_PREFIX: str = "models"

# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------

#: Z-score para intervalo de confiança de 95 % (IC = predicted ± Z * RMSE).
Z_SCORE_95_CONFIDENCE: float = 1.96
