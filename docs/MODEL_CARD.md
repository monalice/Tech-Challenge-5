# Model Card

## Resumo

Este projeto utiliza um modelo LSTM para prever o próximo fechamento horário do Bitcoin (`BTC-USD`). O modelo é consumido tanto pelo endpoint `POST /predict` quanto pelo Agente ReAct, que combina previsão numérica, preço atual e contexto de mercado.

## Objetivo do Modelo

- Prever o preço de fechamento da próxima hora do Bitcoin em USD.
- Apoiar respostas do agente com um componente quantitativo de curto prazo.
- Fornecer uma estimativa operacional para monitoramento, auditoria e comparação com valores reais.

## Dataset

- Ativo: `BTC-USD`
- Frequência: `1h`
- Janela histórica: `730d`
- Lookback: `60` candles
- Fonte primária: Yahoo Finance (`yfinance`)
- Fonte de fallback: Binance REST API pública
- Cache local de apoio: `models/btc_hourly_cache.csv`

O pipeline de treino normaliza os dados e remove valores ausentes/duplicados. Na inferência, a aplicação também remove a vela horária em formação por padrão, usando apenas candles fechados.

## Features

O modelo atual usa 6 features técnicas, descritas em [models/model_metadata_btc.json](models/model_metadata_btc.json):

- `log_return`
- `rsi`
- `macd_signal`
- `bb_pct_b`
- `sma_ratio`
- `vol_ratio`

Essas features são calculadas a partir de OHLCV horário do BTC.

## Arquitetura

- Tipo declarado: `bidirectional_lstm_multifeature`
- Camada principal: LSTM bidirecional
- Camadas adicionais: duas camadas LSTM adicionais no pipeline de treino
- Otimizador: Adam
- Regularização operacional: early stopping e `ReduceLROnPlateau`
- Validação temporal: walk-forward backtest com `3` folds

O alvo do modelo é `log_return`, que depois é convertido novamente para preço.

## Métricas

Métricas registradas em [models/model_metadata_btc.json](models/model_metadata_btc.json):

- MAE em preço: `285.46` USD
- RMSE em preço: `411.72` USD
- MAPE em preço: `0.3679%`
- MAE em retorno: `0.00368`
- Acurácia direcional: `50.11%`

Baseline registrado no metadata:

- MAE baseline: `261.81` USD
- RMSE baseline: `395.20` USD
- MAPE baseline: `0.3390%`
- `beats_baseline`: `false`

## Uso Pretendido

- Suporte a previsão horária de BTC em contexto analítico e demonstrativo.
- Apoio ao endpoint de inferência e ao agente conversacional.
- Monitoramento de drift e comparação histórica via `GET /predictions/history`.

## Limitações

- O modelo cobre apenas `BTC-USD`.
- O horizonte é curto: previsão do próximo fechamento horário, não projeções de médio ou longo prazo.
- A performance registrada não supera o baseline armazenado no metadata atual.
- A acurácia direcional próxima de 50% limita o uso para decisões automáticas de trading.
- O preço do BTC é altamente sensível a eventos macro, fluxo institucional, liquidez e notícias não observadas diretamente pelo modelo.
- Mudanças bruscas de regime de mercado podem degradar a qualidade da previsão.
- O histórico de previsões é mantido em memória enquanto o processo da API estiver ativo.

## Riscos

- Uso inadequado para recomendação financeira determinística.
- Interpretação excessiva de pequenas diferenças percentuais.
- Dependência de provedores externos de dados de mercado.

## Salvaguardas Operacionais

- Fallback automático de dados: Yahoo Finance para Binance.
- Intervalo de confiança estimado no endpoint `/predict`.
- Erro percentual estimado exposto na API.
- Monitoramento via Prometheus e histórico recente de previsões.

## Artefatos Relacionados

- [models/lstm_btc_hourly.keras](models/lstm_btc_hourly.keras)
- [models/model_metadata_btc.json](models/model_metadata_btc.json)
- [src/train_model.py](src/train_model.py)
- [src/app.py](src/app.py)
