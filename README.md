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

Para monitorar a API em produção, adicione o scrape do endpoint `/metrics` no seu `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: stockcast
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```

Exemplo de dashboard Grafana: importe um dashboard genérico de FastAPI/Python e aponte para as métricas `stockcast_*`.

## Treinamento do modelo (com MLflow)

Para testar a API, não é necessário treinar o modelo localmente: os artefatos já estão versionados no repositório. Nesse caso, basta rodar a API com Docker.

Treine localmente apenas se quiser gerar novos artefatos com o modelo melhorado.

### 1. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha os valores reais antes de executar:

```bash
cp .env .env.local   # edite .env.local com suas credenciais reais
```

O arquivo `.env` contém as seguintes variáveis (todas com valores de exemplo):

| Variável | Descrição |
| --- | --- |
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow (ex.: PostgreSQL RDS) |
| `MLFLOW_ARTIFACT_URI` | URI do bucket S3 para artefatos |
| `MLFLOW_EXPERIMENT_NAME` | Nome do experimento MLflow |
| `MLFLOW_MODEL_NAME` | Tag `model_name` registrada no run |
| `MLFLOW_MODEL_VERSION` | Tag `model_version` registrada no run |
| `MLFLOW_OWNER` | Tag `owner` registrada no run |
| `MLFLOW_RISK_LEVEL` | Tag `risk_level` registrada no run |
| `AWS_ACCESS_KEY_ID` | Credencial AWS para escrita no S3 |
| `AWS_SECRET_ACCESS_KEY` | Credencial AWS para escrita no S3 |
| `AWS_DEFAULT_REGION` | Região AWS do bucket |

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

## Observações

- A API utiliza cache curto e retry para chamadas ao Yahoo Finance.
- Com fallback para a Binance, a API mantém disponibilidade mesmo durante instabilidades da fonte primária.
- O healthcheck do Docker Compose usa `GET /live` para evitar impacto em cache de mercado.
- O pipeline treina em `log-return` e converte previsão para preço final.
- O histórico de previsões (`/predictions/history`) persiste em memória enquanto o container estiver ativo.
