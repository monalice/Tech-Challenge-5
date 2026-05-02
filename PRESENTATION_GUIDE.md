# Guia de Apresentação para a Banca — Datathon Fase 05

> **Sistema:** StockCast — Previsão Horária de Bitcoin com Agente ReAct  
> **Branch:** `develop` | **Repositório:** `monalice/Tech-Challenge-5`  
> **Estrutura da apresentação:** ≤ 10 min pitch + Q&A

---

## Roteiro de Apresentação (10 min)

| Bloco | Tempo | Conteúdo |
|---|---|---|
| Problema | ~1 min | Por que prever o preço do BTC? O que o sistema entrega? |
| Arquitetura | ~2 min | Visão geral dos componentes e fluxo de dados |
| Demo ao vivo | ~3 min | `/chat`, `/predict`, `/metrics` |
| Resultados | ~2 min | Métricas do modelo, avaliação RAG, LLM-judge |
| Governança | ~1 min | Guardrails, LGPD, OWASP, Model Card |
| Encerramento | ~1 min | Próximos passos e perguntas |

---

## 1. Problema e Proposta de Valor

**O que o sistema faz:**
Prevê o preço de fechamento da **próxima hora** do Bitcoin (`BTC-USD`) e expõe essa previsão via:
- endpoint REST (`POST /predict`) para integração direta
- agente conversacional em linguagem natural (`POST /chat`)

**Por que importa:**
- Bitcoin é o ativo cripto de maior liquidez e referência de mercado
- Previsão horária tem uso prático em arbitragem, gestão de risco e monitoramento de posições
- O sistema integra previsão quantitativa (LSTM) + contexto qualitativo (RAG) + LLM

**Diferencial arquitetural:**
- Fallback automático de dados (Yahoo Finance → Binance) garante disponibilidade
- Agente ReAct orquestra 3 ferramentas de forma transparente e auditável
- Guardrails de entrada e saída protegem contra prompt injection e vazamento de PII

---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (src/app.py)                    │
│                                                               │
│  POST /chat ──► InputGuardrail ──► Agente ReAct ──► OutputGuardrail
│                                         │                     │
│                              ┌──────────┼──────────┐         │
│                              ▼          ▼          ▼         │
│                         LSTM Tool  Cotação RAG Tool          │
│                         (Previsão) (yfinance) (Bedrock)      │
│                                                               │
│  POST /predict ──► InferenceService ──► LSTM ──► resposta    │
│  GET  /metrics ──► Prometheus metrics                         │
│  GET  /predictions/history ──► histórico auditável           │
└─────────────────────────────────────────────────────────────┘
               │ dados                      │ artefatos
               ▼                            ▼
   YFinance / Binance API          models/lstm_btc_hourly.keras
   (fallback automático)           models/scaler_btc.gz
                                   models/model_metadata_btc.json
```

### Componentes principais

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| API FastAPI | `src/app.py` | Orquestração de todos os endpoints |
| Treinamento | `training/train_model.py` | Pipeline LSTM com MLflow tracking |
| Agente ReAct | `src/agent/react_agent.py` | LangChain + Bedrock (Claude Haiku) |
| RAG pipeline | `src/agent/rag_pipeline.py` | Vector store local + Bedrock embeddings |
| Guardrails | `src/security/guardrails.py` | Segurança de input/output |
| Drift detection | `src/domain/drift/detection.py` | Comparação previsão vs. real |
| Observabilidade | `src/adapters/observability/prometheus.py` | Métricas Prometheus |
| Infraestrutura | `infra/terraform/` | AWS ECS + RDS + S3 + Secrets Manager |

---

## 3. Etapa 1 — Dados + Baseline (Fases 01–02)

### Dataset

| Atributo | Valor |
|---|---|
| Ativo | `BTC-USD` |
| Frequência | `1h` (candles OHLCV) |
| Janela histórica de treino | `730 dias` |
| Lookback para inferência | `60 candles` |
| Fonte primária | Yahoo Finance (`yfinance`) |
| Fonte de fallback | Binance REST API pública |
| Cache local | `models/btc_hourly_cache.csv` |

### Features técnicas (6 features)

```
log_return     → retorno logarítmico da vela
rsi            → RSI de 14 períodos
macd_signal    → linha de sinal do MACD
bb_pct_b       → posição relativa nas Bandas de Bollinger
sma_ratio      → razão SMA(7) / SMA(21)
vol_ratio      → razão de volume relativo
```

Implementação: `src/domain/features/technical_features.py`  
Validação de schema: `pandera` com contrato declarado em `train_model.py`

### Modelo LSTM

```
Tipo:          bidirectional_lstm_multifeature
Alvo:          log_return (convertido de volta para preço)
Otimizador:    Adam
Callbacks:     EarlyStopping + ReduceLROnPlateau
Validação:     Walk-forward backtest com 3 folds (TimeSeriesSplit)
```

### Métricas do modelo

> Registradas em `models/model_metadata_btc.json` e logadas no **MLflow**

| Métrica | Valor |
|---|---|
| MAE em preço | **285,46 USD** |
| RMSE em preço | **411,72 USD** |
| MAPE em preço | **0,3679 %** |
| Acurácia direcional | **50,11 %** |
| Beats baseline? | `false` (registrado com transparência) |

> **Nota para a banca:** o baseline (naive last-value) tem MAE de 261 USD. O LSTM atual não supera o baseline nas métricas absolutas — essa limitação está documentada com transparência no Model Card. O valor demonstrado está na **arquitetura completa** e não apenas no modelo isolado.

### MLflow tracking

- Experimento: `btc_lstm`
- Cada run loga: métricas, parâmetros, artefatos, features e metadata
- Versionamento via `Model Registry` com tags de `git_sha`, `training_data_version`, `risk_level`
- Reproduzível via `make train` ou DVC pipeline

### Gestão de dados e reprodutibilidade

```bash
make train          # treina e loga no MLflow
dvc repro           # reproduz pipeline completo via DVC
docker-compose up   # sobe API com artefatos versionados
```

---

## 4. Etapa 2 — LLM + Agente ReAct (Fases 03–05)

### Agente ReAct

Implementado em `src/agent/react_agent.py` com **LangChain + AWS Bedrock (Claude Haiku)**

**3 ferramentas (@tool):**

| Ferramenta | Descrição |
|---|---|
| `PrevisaoBitcoinTool` | Executa o pipeline LSTM via InferenceService |
| `CotacaoAtualTool` | Consulta preço atual via YFinance/Binance |
| `CryptoKnowledgeRAG` | Recupera contexto de notícias cripto do vector store |

**Configuração do LLM:**
- Modelo padrão: `anthropic.claude-haiku-4-5-20251001-v1:0` (via Bedrock)
- Temperatura: `0.0` (máximo determinismo)
- Parâmetros configuráveis via `.env`: `AGENT_LLM_TEMPERATURE`, `AGENT_LLM_TOP_P`, `AGENT_LLM_TOP_K`

### RAG Pipeline

Implementado em `src/agent/rag_pipeline.py`

- **Embeddings:** Amazon Bedrock (`amazon.titan-embed-text-v2:0`, 3072 dimensões)
- **Vector store:** ChromaDB local em `data/processed/crypto_news_chroma`
- **Corpus simulado:** 5 notícias curadas sobre ETFs, Fed, mineradores, volatilidade e dominância BTC
- **Busca:** similaridade semântica com `similarity_search`

### Endpoint `/chat`

```json
POST /chat
{ "message": "Qual a previsão do BTC para a próxima hora e como está o preço agora?" }
```

Fluxo interno:
1. `InputGuardrail` valida a mensagem (injection, tamanho)
2. Agente ReAct orquestra as 3 ferramentas em sequência (Thought → Action → Observation)
3. Observações intermediárias são sanitizadas pelo `OutputGuardrail`
4. Resposta final é sanitizada antes de retornar ao cliente

### CI/CD — GitHub Actions

Pipeline em `.github/workflows/ci.yml`:
```
lint (ruff) → type-check (mypy) → security (bandit) → tests (pytest)
```

Gates de qualidade:
- `--cov-fail-under=60` — mínimo de cobertura exigido
- `bandit -r src/` — varredura de vulnerabilidades de segurança
- `mypy --strict` — verificação de tipos estrita

---

## 5. Etapa 3 — Avaliação + Observabilidade (Fases 03–05)

### Golden Set

- **Localização:** `data/golden_set/btc_rag_golden_set.json`
- **Tamanho:** **21 pares** question/answer/contexts (supera mínimo de 20)
- **Categorias:** model_scope, inference, api_features, resilience, observability, security, evaluation

### RAGAS — 4 métricas obrigatórias

> Executar: `python evaluation/ragas_eval.py`  
> Resultados em: `evaluation/ragas_results.json`

| Métrica | Valor | Interpretação |
|---|---|---|
| `faithfulness` | **0.44** | Respostas fundamentadas nas observações das ferramentas |
| `answer_relevancy` | **0.65** | Alinhamento da resposta com a pergunta |
| `context_precision` | **0.38** | Precisão dos contextos recuperados pelo RAG |
| `context_recall` | **0.39** | Cobertura dos contextos relevantes |

- `sample_count`: 21 questões avaliadas
- `seed`: 42 (reprodutível)
- `metric_backend`: deterministic_offline_fallback (sem dependência de API em avaliação)

### LLM-as-Judge — 3 critérios

> Executar: `make llm-judge` (offline) ou `make llm-judge-live` (com API)  
> Resultados em: `evaluation/llm_judge_results.json`

| Critério | Média (1–5) |
|---|---|
| `precisao_financeira` | **3,52** |
| `clareza` | **4,19** |
| `ausencia_alucinacoes` | **4,24** |
| **nota_final** | **7,67 / 10** |

- Juiz: `gemini-2.5-flash` (5 avaliações LLM + 16 fallback determinístico)
- Schema de resposta: `JudgeVerdict` com `CriterionScore` (score + rationale)
- Resultado versionado em: `evaluation/results/llm_judge/`

### Drift Detection

Implementado em `src/domain/drift/detection.py`:

- Compara previsões históricas (`/predictions/history`) com preços reais de mercado
- Calcula MAE acumulado e desvio percentual médio
- Threshold configurável em `configs/monitoring_config.yaml`

Endpoint de trigger manual:
```bash
make drift-check
# POST /admin/check-drift {"ticker": "BTC-USD"}
```

Scheduler automático: `scripts/run_drift_scheduler.py` + `src/serving/drift_automation.py`

### Observabilidade Operacional

**Prometheus (`GET /metrics`):**

| Métrica | Tipo | Descrição |
|---|---|---|
| `stockcast_predict_requests_total` | Counter | Chamadas a `/predict` por status |
| `stockcast_predict_latency_seconds` | Histogram | Latência de inferência |
| `stockcast_market_data_source_total` | Counter | Uso por fonte (yfinance/binance) |
| `stockcast_market_data_errors_total` | Counter | Falhas por fonte |
| `stockcast_cpu_usage_percent` | Gauge | CPU do processo |
| `stockcast_memory_usage_percent` | Gauge | Memória do processo |

**Stack completo:**
```bash
docker-compose up -d  # API + Prometheus + Grafana
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

**Telemetria de qualidade LLM — Langfuse:**
- Traces por interação: faithfulness, relevância, latência, tokens
- Sequência de ferramentas invocadas por chamada
- Opcional e não-bloqueante (configura via `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`)

---

## 6. Etapa 4 — Segurança + Governança (Fases 04–05)

### OWASP Top 10 LLM — Mapeamento (≥ 5 ameaças)

> Documento completo: `docs/OWASP_MAPPING.md`

| # | Ameaça | Mitigação Implementada |
|---|---|---|
| 1 | **Prompt Injection** | `InputGuardrail` com regex — bloqueia antes do agente |
| 2 | **Sensitive Info Disclosure** | `OutputGuardrail` com Presidio + regex fallback |
| 3 | **Training/Retrieval Data Poisoning** | Corpus RAG local e controlado pelo projeto |
| 4 | **Denial of Service (Context Stuffing)** | Limite de 4096 chars no input |
| 5 | **Excessive Agency** | Toolset mínimo e específico; temperatura 0 |
| 6 | **Supply Chain Risk** | Versões pinadas; fallback de dados de mercado |
| 7 | **Secrets Exposure** | Secrets Manager + hooks de pre-commit + `detect-secrets` |

### Guardrails — Implementação

`src/security/guardrails.py`

**InputGuardrail:**
- Bloqueia padrões de prompt injection por regex (`ignore previous instructions`, etc.)
- Rejeita context stuffing (> 4096 caracteres)
- Rejeita input vazio

**OutputGuardrail:**
- Presidio `AnalyzerEngine` + `AnonymizerEngine` (quando instalado)
- Fallback regex para: email, CPF, CNPJ, cartão, telefone
- Marcadores de saída: `<EMAIL_MASKED>`, `<CPF_MASKED>`, `<CNPJ_MASKED>`, `<CARD_MASKED>`, `<PHONE_MASKED>`

**Onde é aplicado em `src/app.py`:**
1. Mensagem de entrada → `InputGuardrail.validate()`
2. Observações intermediárias → `OutputGuardrail.sanitize()`
3. Resposta final → `OutputGuardrail.sanitize()`

### Red Team — 5 cenários adversariais

> Documento completo: `docs/RED_TEAM_REPORT.md`

| Cenário | Ataque | Resultado |
|---|---|---|
| 1 | Prompt injection direto | Bloqueado pelo InputGuardrail |
| 2 | Context stuffing (> 4096 chars) | Bloqueado por tamanho |
| 3 | Vazamento de PII na resposta | Mascarado pelo OutputGuardrail |
| 4 | Falha do motor Presidio | Fallback regex mantém proteção |
| 5 | Inputs adversariais de borda | Tratados com graceful degradation |

### LGPD

> Documento: `docs/LGPD_PLAN.md`

- Minimização de dados: sistema não coleta PII como dado de negócio
- Sanitização dupla: Presidio + regex aplicados em toda resposta conversacional
- Secrets: armazenados em AWS Secrets Manager (Terraform: `infra/terraform/secrets.tf`)
- Retenção: histórico de previsões não armazena PII; textos brutos não são persistidos
- Limitação: `analyzer` configurado para `language="en"` — cobertura parcial em PT

### Testes de segurança

```bash
make security   # bandit -r src/ (OWASP-aligned)
pytest tests/test_guardrails.py -v  # 10+ cenários de guardrails
```

---

## 7. Model Card e System Card

> Documentos: `docs/MODEL_CARD.md`, `docs/SYSTEM_CARD.md`

### Model Card — resumo

- Tipo: `bidirectional_lstm_multifeature`
- Dataset: `BTC-USD`, `1h`, `730d`
- 6 features técnicas padronizadas
- Métricas registradas e baseline documentado
- Riscos e limitações explícitos (acurácia direcional ~50%)
- Artefatos versionados: `.keras`, `.gz`, `.json`

### System Card — resumo

- Componentes: FastAPI + LSTM + ReAct + RAG + Guardrails + Prometheus
- Arquitetura cloud: ECS Fargate + ECR + RDS PostgreSQL + S3 + Secrets Manager
- Fluxo de dados documentado passo a passo
- Telemetria LLM com Langfuse (opcional)
- Observabilidade operacional com Prometheus

---

## 8. Infraestrutura Cloud (AWS + Terraform)

> Código: `infra/terraform/`

| Serviço | Arquivo Terraform | Uso |
|---|---|---|
| ECS Fargate | `ecs.tf` | Execução da API em container |
| ECR | `ecs.tf` | Armazenamento da imagem Docker |
| RDS PostgreSQL | `rds.tf` | Backend do MLflow |
| S3 | `storage.tf` | Artefatos MLflow, DVC, estado Terraform |
| Secrets Manager | `secrets.tf` | API keys e senha do banco |
| ALB | `alb.tf` | Load balancer da API |
| CloudWatch | `observability.tf` | Logs do ECS |
| EventBridge | `eventbridge.tf` | Agendamento de jobs |
| Step Functions | `step_functions.tf` | Orquestração de retraining |

Backend Terraform: S3 (`tech-challenge-5-terraform-state`, `us-east-1`)

---

## 9. Testes e Qualidade de Código

### Cobertura e gate

```bash
make test   # pytest tests/ --cov=src --cov-fail-under=60 -v
```

### Arquivos de teste

| Arquivo | Cobertura |
|---|---|
| `tests/test_api.py` | Endpoints FastAPI (TestClient) |
| `tests/test_guardrails.py` | Segurança: injection, PII, context stuffing |
| `tests/test_features.py` | Schema de features, nulls, shapes |
| `tests/test_drift_detection.py` | Drift detection e thresholds |
| `tests/test_drift_automation.py` | Automação de drift |
| `tests/test_react_agent.py` | Agente ReAct e ferramentas |
| `tests/test_train_model.py` | Pipeline de treinamento |
| `tests/test_ragas_eval.py` | Avaliação RAGAS |
| `tests/test_technical_features.py` | Features técnicas individuais |
| `tests/test_mlflow_registry_integration.py` | Integração MLflow |

### Qualidade de código

```bash
make quality   # lint + type-check + security + tests
```

- **Lint:** `ruff` (E, F, I, B, UP, SIM, C4, N)
- **Type check:** `mypy --strict` (Python 3.12)
- **Security:** `bandit -r src/`
- **Pre-commit:** `detect-private-key`, `detect-secrets`, bloqueio de `.env` e artefatos

---

## 10. Respostas para Perguntas Frequentes da Banca

### "Por que o modelo não bate o baseline?"

O modelo LSTM (MAE 285 USD) não supera o baseline naive (MAE 261 USD) porque prever retornos de BTC em escala horária é inerentemente próximo de ruído. Isso está documentado com transparência no Model Card. O valor do projeto está na **plataforma completa**: pipeline reprodutível, agente conversacional, guardrails, observabilidade e governança — não em superação de baseline em ativo altamente volátil.

### "O agente está respondendo de forma determinística?"

Sim. Temperatura `0.0` no LLM garante máximo determinismo operacional. O agente ReAct usa a sequência Thought → Action → Observation de forma transparente e auditável.

### "Como é feita a resiliência de dados?"

Fallback automático: Yahoo Finance → Binance REST API pública. O campo `data_source` em toda resposta de `/predict` indica qual fonte está ativa. Em caso de falha de ambas, o cache local (`btc_hourly_cache.csv`) é consultado.

### "Como você detecta drift em produção?"

O endpoint `POST /admin/check-drift` compara o histórico de previsões (`/predictions/history`) com os preços reais via `detect_data_drift()`. O scheduler (`scripts/run_drift_scheduler.py`) executa isso periodicamente e pode disparar retraining via Step Functions/EventBridge (configurados no Terraform).

### "Como o sistema garante LGPD?"

Sanitização dupla em toda resposta do agente: Presidio (NER) + regex fallback. PII nunca é armazenado em texto bruto. Segredos ficam em AWS Secrets Manager. O LGPD Plan (`docs/LGPD_PLAN.md`) documenta a estratégia completa.

### "Qual o papel do RAG?"

O `CryptoKnowledgeRAG` fornece contexto qualitativo de mercado (notícias, sentimento macro, fluxo institucional) que o modelo LSTM não captura. O vetor store ChromaDB é local e controlado, reduzindo risco de poisoning por fontes externas.

### "Quantas ferramentas o agente tem?"

3 ferramentas (`@tool` do LangChain):
1. `PrevisaoBitcoinTool` — previsão LSTM
2. `CotacaoAtualTool` — preço atual de mercado
3. `CryptoKnowledgeRAG` — contexto cripto via RAG

### "Como subir o sistema para demo?"

```bash
# Opção 1: Docker Compose completo (API + Prometheus + Grafana)
docker-compose up --build

# Opção 2: Local
.venv\Scripts\activate
uvicorn src.app:app --host 127.0.0.1 --port 8000

# Swagger: http://127.0.0.1:8000/docs
```

Variáveis de ambiente necessárias (copiar de `.env.example`):
- `BEDROCK_AWS_REGION` — região AWS para Bedrock
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — credenciais AWS
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — telemetria (opcional)

---

## 11. Checklist Final por Critério da Rubrica

### Etapa 1 — Dados + Baseline (Fases 01–02)

- [x] Dataset `BTC-USD`, `1h`, `730d` — `training/train_model.py`
- [x] 6 features técnicas documentadas — `src/domain/features/technical_features.py`
- [x] LSTM bidirecional com walk-forward validation (3 folds)
- [x] MLflow tracking com métricas, parâmetros e artefatos
- [x] Baseline documentado no metadata
- [x] Pipeline reprodutível via `make train` + DVC
- [x] Docker e `docker-compose.yml`
- [x] `pyproject.toml` com dependências gerenciadas

### Etapa 2 — LLM + Agente (Fases 03–05)

- [x] LLM via AWS Bedrock (Claude Haiku)
- [x] Agente ReAct com **3 tools** (LSTM, cotação, RAG)
- [x] RAG com Bedrock embeddings + ChromaDB local
- [x] CI/CD com GitHub Actions (lint → type-check → security → tests)
- [x] `make quality` como gate unificado
- [x] Configuração de modelo/temperatura via variáveis de ambiente

### Etapa 3 — Avaliação + Observabilidade (Fases 03–05)

- [x] Golden set com **21 pares** (`data/golden_set/btc_rag_golden_set.json`)
- [x] RAGAS com **4 métricas** (`evaluation/ragas_results.json`)
- [x] LLM-as-judge com **3 critérios** (`evaluation/llm_judge_results.json`)
- [x] Drift detection implementado (`src/domain/drift/detection.py`)
- [x] Prometheus com **6 métricas** (`GET /metrics`)
- [x] Stack de observabilidade completa (Prometheus + Grafana via Docker Compose)
- [x] Telemetria LLM com Langfuse (opcional, documentado no System Card)
- [x] Histórico de previsões auditável (`GET /predictions/history`)

### Etapa 4 — Segurança + Governança (Fases 04–05)

- [x] OWASP mapping com **7 ameaças** (`docs/OWASP_MAPPING.md`)
- [x] Guardrails de input e output funcionais (`src/security/guardrails.py`)
- [x] **5 cenários adversariais** testados (`docs/RED_TEAM_REPORT.md`)
- [x] Plano LGPD com estratégia de sanitização (`docs/LGPD_PLAN.md`)
- [x] Model Card completo (`docs/MODEL_CARD.md`)
- [x] System Card completo (`docs/SYSTEM_CARD.md`)
- [x] Secrets em AWS Secrets Manager (Terraform)
- [x] Pre-commit hooks com `detect-secrets`
- [x] Testes de guardrails (`tests/test_guardrails.py`)

---

## 12. Arquivos-Chave para Demonstrar na Banca

| O que mostrar | Onde |
|---|---|
| API rodando | `http://127.0.0.1:8000/docs` |
| Swagger do `/chat` | Executar com pergunta sobre BTC |
| Swagger do `/predict` | Mostrar intervalo de confiança |
| Métricas Prometheus | `http://127.0.0.1:8000/metrics` |
| Histórico de previsões | `http://127.0.0.1:8000/predictions/history` |
| MLflow runs | `mlflow ui` → experimento `btc_lstm` |
| Resultados LLM-judge | `evaluation/llm_judge_results.json` |
| Resultados RAGAS | `evaluation/ragas_results.json` |
| Guardrails em ação | `pytest tests/test_guardrails.py -v` |
| Golden set | `data/golden_set/btc_rag_golden_set.json` |
| Infraestrutura Terraform | `infra/terraform/` (mostrar ECS + RDS + S3) |
| Grafana dashboard | `http://localhost:3000` (com Docker Compose) |
