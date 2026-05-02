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
# Estatística
# ---------------------------------------------------------------------------

#: Z-score para intervalo de confiança de 95 % (IC = predicted ± Z * RMSE).
Z_SCORE_95_CONFIDENCE: float = 1.96
