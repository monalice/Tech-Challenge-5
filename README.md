# Tech-Challenge-5

API e pipeline de treino para previsão do próximo fechamento horário do Bitcoin (`BTC-USD`) com LSTM.

## Requisitos

- Docker (recomendado para execução da API)
- Python 3.11+
- Ambiente virtual (`.venv`)

## Executar API com Docker (recomendado)

### Com Docker Compose

```bash
docker-compose up --build
```

### Com Docker direto

```bash
docker build -t stockcast-api:latest .
docker run --rm -p 8000:8000 stockcast-api:latest
```

## Execução local (alternativa)

### Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Subir API localmente

```bash
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

## Documentação da API

Após subir a aplicação, acesse:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Endpoints

- `GET /live`
  - Endpoint leve para liveness (usado no healthcheck do Docker Compose).
  - Não consulta mercado nem aquece cache da previsão.
- `GET /health`
  - Endpoint de diagnóstico completo.
  - Retorna estado efetivo da API com validações de:
    - artefatos carregados
    - inferência real do modelo
    - acesso a dados de mercado (e qual fonte está ativa: `yfinance` ou `binance`)
  - Inclui timestamp do último candle válido em:
    - `last_market_timestamp_utc`
    - `last_market_timestamp_brt`
- `GET /metrics`
  - Expõe métricas operacionais no formato **Prometheus/OpenMetrics**.
  - Compatível com Prometheus, Grafana e qualquer ferramenta de observabilidade padrão.
  - Métricas disponíveis:
    - `stockcast_predict_requests_total` – contagem de chamadas a `/predict` por status
    - `stockcast_predict_latency_seconds` – histograma de latência de inferência
    - `stockcast_market_data_source_total` – contagem por fonte de dados (yfinance/binance)
    - `stockcast_market_data_errors_total` – total de falhas por fonte
    - `stockcast_cpu_usage_percent` – uso de CPU do processo
    - `stockcast_memory_usage_percent` – uso de memória do processo
- `GET /predictions/history`
  - Retorna as últimas 100 previsões realizadas pela API em ordem decrescente.
  - Inclui: timestamp, ticker, modo de entrada, candle usado, previsão, fonte de dados e latência.
  - Útil para **auditoria**, **detecção de drift** e comparação posterior com valores reais.
- `POST /chat`
=  - Endpoint do **Agente LLM ReAct** (LangChain + Gemini).
  - Recebe uma mensagem em linguagem natural e orquestra 3 ferramentas:
    - `PrevisaoBitcoinTool` — executa o pipeline de inferência LSTM e retorna a previsão.
    - `CotacaoAtualTool` — consulta o preço atual do BTC via yfinance / Binance.
    - `CryptoKnowledgeRAG` — fornece contexto financeiro e notícias cripto via vector store local.
  - Requer a variável de ambiente `GOOGLE_API_KEY`.
  - Exemplo de body:

```json
{ "message": "Qual a previsão do BTC para a próxima hora e como está o preço agora?" }
```

- `POST /predict`
  - Aceita apenas `BTC-USD`.
  - O body é opcional (`{}` usa os valores padrão).
  - Por padrão usa apenas velas fechadas. Para incluir a vela em formação, envie `use_partial_candle: true`.
  - Retorna, além do preço previsto:
    - `input_mode` (modo de entrada usado no modelo)
    - `last_input_candle_utc` / `last_input_candle_brt` (último candle na entrada)
    - `forecast_for_utc` / `forecast_for_brt` (início da hora prevista)
    - `forecast_close_utc` / `forecast_close_brt` (fechamento da hora prevista)
    - `confidence_interval_95_usd` (intervalo de confiança estimado)
    - `estimated_error_pct` (erro percentual estimado)
    - `data_source` (fonte de dados utilizada: `yfinance` ou `binance`)
  - Exemplo de body:

```json
{
  "ticker": "BTC-USD",
  "use_partial_candle": false
}
```

Exemplo com vela parcial:

```json
{
  "ticker": "BTC-USD",
  "use_partial_candle": true
}
```

No Swagger (`/docs`), o parâmetro aparece no body do `POST /predict` e também no exemplo **com_vela_parcial**.

## Resiliência de dados de mercado

A API utiliza **duas fontes de dados com fallback automático**:

1. **Yahoo Finance** (fonte primária) — `yfinance`, com cache e até 3 tentativas.
2. **Binance REST API pública** (fallback) — ativado automaticamente quando o Yahoo Finance estiver indisponível ou com limite de requisições atingido.

O campo `data_source` nas respostas de `/predict` e `/health` indica qual fonte está em uso.

## Monitoramento com Prometheus + Grafana (MLOps)

O projeto inclui monitoramento operacional com Docker Compose para API + Prometheus + Grafana.

### Subir stack de observabilidade

```bash
docker-compose up -d --build
```

Serviços esperados:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (user `admin`, senha `admin`)

O scrape do Prometheus está em [monitoring/prometheus.yml](monitoring/prometheus.yml):

```yaml
scrape_configs:
  - job_name: stockcast
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

### Configurar datasource no Grafana

1. Acesse `http://localhost:3000`.
2. Vá em `Connections` > `Data sources` > `Add data source`.
3. Selecione `Prometheus`.
4. Em `URL`, informe `http://prometheus:9090`.
5. Clique em `Save & test`.

### Importar dashboard FastAPI + Prometheus

1. No Grafana, vá em `Dashboards` > `Import`.
2. Informe o ID `14282` (Grafana.com).
3. Selecione o datasource Prometheus criado.
4. Conclua em `Import`.

Depois disso, use as métricas `stockcast_*` para acompanhar latência, erros, uso de CPU/memória e volume de predições.

## Treinamento do modelo (com MLflow)

Para testar a API, não é necessário treinar o modelo localmente: os artefatos já estão versionados no repositório. Nesse caso, basta rodar a API com Docker.

Treine localmente apenas se quiser gerar novos artefatos com o modelo melhorado.

### 1. Configurar variáveis de ambiente

Copie [.env.example](.env.example) para `.env` e preencha com os valores reais. Consulte a [seção de variáveis de ambiente](#variáveis-de-ambiente) para detalhes de cada chave.

### 2. Executar o treinamento

```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^[^#]') { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) } }; .venv\Scripts\python.exe -u src/train_model.py
```

Saídas geradas em `models/`:

- `lstm_btc_hourly.keras`
- `scaler_btc.gz`
- `scaler_btc_return.gz`
- `model_metadata_btc.json`

### Melhorias no pipeline de treino

O modelo treinado com `train_model.py` incorpora:

- **Features técnicas**: além do `log_return`, usa RSI(14), MACD Signal, Bollinger %B, razão de SMA(7/21) e razão de volume — totalizando 6 features de entrada.
- **Arquitetura bidirecional**: LSTM Bidirecional + 2 camadas LSTM adicionais para melhor captura de padrões temporais.
- **Learning rate scheduling**: `ReduceLROnPlateau` reduz a taxa de aprendizado automaticamente quando o progresso estagna.
- **Walk-forward backtest**: validação temporal em 3 folds para estimar performance fora da amostra.

## Data Management com DVC

O projeto inclui um pipeline DVC em [dvc.yaml](dvc.yaml) e um script de setup em [scripts/setup_dvc.sh](scripts/setup_dvc.sh).

Fluxo recomendado:

1. Instale as dependências na `.venv` com `pip install -r requirements.txt`.
2. Execute o setup inicial do DVC com `bash scripts/setup_dvc.sh`.
3. Reproduza o pipeline com `dvc repro`.

O setup realiza:

- `dvc init`
- configuração do remote padrão `s3remote`
- `dvc add models/btc_hourly_cache.csv`

O pipeline DVC orquestra:

- `prepare_data`: geração/atualização de `models/btc_hourly_cache.csv`
- `train_model`: treino do modelo em [src/train_model.py](src/train_model.py)

## Deploy de infraestrutura AWS (Terraform)

Os arquivos IaC estão em [infra/terraform](infra/terraform), incluindo:

- S3 para artefatos DVC/MLflow
- RDS PostgreSQL para metadados do MLflow
- ECR para imagem Docker
- ECS/Fargate para execução da API
- Secrets Manager para `GOOGLE_API_KEY` e senha do banco

### Preparação

1. Copie [infra/terraform/terraform.tfvars.example](infra/terraform/terraform.tfvars.example) para [infra/terraform/terraform.tfvars](infra/terraform/terraform.tfvars).
2. Preencha os valores reais de `db_password` e `google_api_key`.

### Execução (PowerShell)

```powershell
./scripts/deploy_terraform.ps1 -Action init
./scripts/deploy_terraform.ps1 -Action validate
./scripts/deploy_terraform.ps1 -Action plan
./scripts/deploy_terraform.ps1 -Action apply
```

Execução em fluxo único:

```powershell
./scripts/deploy_terraform.ps1 -Action all
```

Para aplicar sem confirmação interativa:

```powershell
./scripts/deploy_terraform.ps1 -Action apply -AutoApprove
```

## Documentação do Projeto

Além da documentação operacional da API, o projeto inclui artefatos de governança, risco e arquitetura em [docs](docs):

- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — dataset, arquitetura LSTM, métricas e limitações do modelo.
- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md) — arquitetura cloud, fluxo de dados e guardrails do sistema.
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md) — uso de Presidio para proteção de PII e diretrizes LGPD.
- [docs/OWASP_MAPPING.md](docs/OWASP_MAPPING.md) — mapeamento de ameaças OWASP Top 10 para LLMs e mitigações no código.
- [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) — cenários adversariais testados contra o Agente ReAct.

## Desenvolvimento

### Makefile — atalhos principais

O projeto inclui um `Makefile` com os comandos mais comuns:

```bash
make install       # instala dependências de desenvolvimento
make lint          # ruff check em src/, monitoring/, tests/
make type-check    # mypy com --explicit-package-bases
make security      # bandit -r src/
make test          # pytest com cobertura mínima de 60 %
make quality       # lint + type-check + security + test
make train         # executa src/train_model.py
make serve         # sobe a API localmente com reload
make docker-build  # docker build -t btc-predictor:latest .
```

### Pre-commit hooks

O arquivo [.pre-commit-config.yaml](.pre-commit-config.yaml) configura hooks automáticos de qualidade:

```bash
pip install pre-commit
pre-commit install        # instala os hooks no repositório
pre-commit run --all-files  # executa manualmente em todos os arquivos
```

Hooks ativos: `ruff` (lint + format), `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-merge-conflict`.

### Variáveis de ambiente

Copie [.env.example](.env.example) e preencha com os valores reais:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `GOOGLE_API_KEY` | Chave da Google AI (obrigatória para o agente ReAct) |
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow (ex.: PostgreSQL RDS) |
| `MLFLOW_EXPERIMENT_NAME` | Nome do experimento MLflow |
| `AWS_REGION` | Região AWS para ECR/ECS |
| `ECR_REPOSITORY` | Nome do repositório ECR |
| `ECS_CLUSTER` | Cluster ECS para deploy |
| `LANGFUSE_PUBLIC_KEY` | Chave pública Langfuse (opcional) |
| `LANGFUSE_SECRET_KEY` | Chave secreta Langfuse (opcional) |
| `APP_PORT` | Porta da aplicação (padrão: `8000`) |

## Telemetria de Qualidade LLM (Langfuse)

O agente ReAct é instrumentado com **[Langfuse](https://cloud.langfuse.com)** para rastreamento de qualidade das chamadas LLM. A integração é **opcional e não bloqueante**: o agente opera normalmente se as credenciais não estiverem definidas.

Para ativar, defina as variáveis no `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Métricas rastreadas por trace: faithfulness, relevância da resposta, latência por chamada LLM, contagem de tokens e sequência de ferramentas invocadas.

## Avaliação de Qualidade RAG (RAGAS)

O golden set com **21 pares Q&A** está em [`data/golden_set/btc_rag_golden_set.json`](data/golden_set/btc_rag_golden_set.json), cobrindo 10 categorias de perguntas:

| Categoria | Exemplos de perguntas |
|---|---|
| `model_scope` | O que o modelo prevê? Qual o horizonte temporal? |
| `technical_analysis` | Quais features técnicas usa o LSTM? |
| `uncertainty` | O modelo indica incerteza da previsão? |
| `agent_tools` | Quais ferramentas o agente ReAct usa? |
| `trend_analysis` | Qual a dominância do BTC em 2024? |
| `market_events` | O que foi o halving de 2024? Por que ETFs spot importam? |
| `market_context` | Como macro influencia o BTC? |
| `data_sources` | Qual fonte de dados foi usada? E se o Yahoo falhar? |
| `api_usage` | Endpoint de histórico de previsões? Diferença /live vs /health? |
| `monitoring` | Que métricas operacionais a API expõe? |

Cada par tem a estrutura:

```json
{
  "query": "...",
  "expected_answer": "...",
  "answer": "...",
  "contexts": ["contexto 1", "contexto 2"],
  "metadata": { "category": "trend_analysis", "difficulty": "medium" }
}
```

Para avaliar com as 4 métricas RAGAS (faithfulness, answer_relevancy, context_precision, context_recall):

```bash
# Usando respostas pré-computadas no golden set (sem chamada de API)
python evaluation/ragas_eval.py \
  --golden-set data/golden_set/btc_rag_golden_set.json

# Gerando respostas ao vivo via /chat (API rodando)
python evaluation/ragas_eval.py \
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --api-url http://localhost:8000
```

Requer `GOOGLE_API_KEY` no ambiente. Resultados salvos em `evaluation/ragas_results.json`.

## Configuração de Monitoramento

O arquivo [`configs/monitoring_config.yaml`](configs/monitoring_config.yaml) centraliza os parâmetros de detecção de drift e nomenclatura do Model Registry:

```yaml
drift:
  psi_warning_threshold: 0.1   # PSI ≥ 0.1 → alerta
  psi_retrain_threshold: 0.2   # PSI ≥ 0.2 → re-treino recomendado
  check_interval_hours: 24
  min_rows_for_comparison: 50

model:
  name: "btc-lstm-hourly"
  registry_stage_production: "Production"
  registry_stage_challenger: "Staging"
```

## Observações

- A API utiliza cache curto e retry para chamadas ao Yahoo Finance.
- Com fallback para a Binance, a API mantém disponibilidade mesmo durante instabilidades da fonte primária.
- O healthcheck do Docker Compose usa `GET /live` para evitar impacto em cache de mercado.
- O pipeline treina em `log-return` e converte previsão para preço final.
- O histórico de previsões (`/predictions/history`) persiste em memória enquanto o container estiver ativo.
