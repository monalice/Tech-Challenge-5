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

### Gerenciamento de dependências com pip-tools

Arquivos fonte:

- [requirements.in](requirements.in)
- [requirements.eval.in](requirements.eval.in)
- [requirements.dataops.in](requirements.dataops.in)

Arquivos lock gerados:

- [requirements.txt](requirements.txt)
- [requirements.eval.txt](requirements.eval.txt)
- [requirements.dataops.txt](requirements.dataops.txt)

Comandos principais:

```bash
make deps-compile   # recompila locks
make deps-upgrade   # recompila locks com upgrade
make deps-sync      # sincroniza ambiente exatamente com os locks
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
Get-Content .env | ForEach-Object { if ($_ -match '^[^#]') { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) } }; .venv\Scripts\python.exe -u training/train_model.py
```

Saídas geradas em `models/`:

- `lstm_btc_hourly.keras`
- `scaler_btc.gz`
- `scaler_btc_return.gz`
- `model_metadata_btc.json`

### Melhorias no pipeline de treino

O modelo treinado com `training/train_model.py` incorpora:

- **Features técnicas**: além do `log_return`, usa RSI(14), MACD Signal, Bollinger %B, razão de SMA(7/21) e razão de volume — totalizando 6 features de entrada.
- **Arquitetura bidirecional**: LSTM Bidirecional + 2 camadas LSTM adicionais para melhor captura de padrões temporais.
- **Learning rate scheduling**: `ReduceLROnPlateau` reduz a taxa de aprendizado automaticamente quando o progresso estagna.
- **Walk-forward backtest**: validação temporal em 3 folds para estimar performance fora da amostra.

## Data Management com DVC

O projeto inclui um pipeline DVC em [dvc.yaml](dvc.yaml) e um script de setup em [scripts/setup_dvc.sh](scripts/setup_dvc.sh).

Fluxo recomendado (reprodutibilidade local):

1. Instale as dependências na `.venv` com `pip install -r requirements.txt`.
2. Execute o setup inicial do DVC com `bash scripts/setup_dvc.sh`.
3. Configure credenciais do remote DVC (S3) no ambiente local.
4. Execute `dvc pull` para baixar artefatos/dados versionados.
5. Reproduza o pipeline com `dvc repro`.

### Reprodução local passo a passo (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install dvc[s3]

# Credenciais de exemplo (use valores reais no seu ambiente)
$env:AWS_ACCESS_KEY_ID="<seu_access_key>"
$env:AWS_SECRET_ACCESS_KEY="<seu_secret_key>"
$env:AWS_DEFAULT_REGION="us-east-1"

dvc pull
dvc repro
```

### Comportamento em CI/CD sem credenciais

Os workflows do GitHub Actions tentam executar `dvc pull` **antes dos testes** somente quando as credenciais opcionais estão presentes:

- `DVC_AWS_ACCESS_KEY_ID`
- `DVC_AWS_SECRET_ACCESS_KEY`
- `DVC_AWS_SESSION_TOKEN` (opcional)
- `DVC_AWS_DEFAULT_REGION` (opcional)

Se as credenciais não estiverem disponíveis (por exemplo, PR de fork), o pipeline **não quebra**: os testes seguem com fixtures locais versionadas e fallback determinístico.

O setup realiza:

- `dvc init`
- configuração do remote padrão `s3remote`
- `dvc add models/btc_hourly_cache.csv`

O pipeline DVC orquestra:

- `prepare_data`: geração/atualização de `models/btc_hourly_cache.csv`
- `train_model`: treino do modelo em [training/train_model.py](training/train_model.py)

### Estratégia segura para dados mínimos de teste

Para garantir reprodutibilidade de testes sem depender de rede/credenciais:

- Dataset mínimo versionado em [tests/fixtures/btc_hourly_minimal.csv](tests/fixtures/btc_hourly_minimal.csv).
- Fixture principal em [tests/conftest.py](tests/conftest.py) tenta carregar esse CSV primeiro.
- Se o arquivo não estiver disponível, a fixture usa fallback sintético **determinístico** (seed fixa).

Isso evita dados sensíveis e mantém os testes estáveis em ambientes locais e CI.

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
make train         # executa training/train_model.py
make serve         # sobe a API localmente com reload
make docker-build  # docker build -t btc-predictor:latest .
make llm-judge     # avalia golden set com 3 criterios e gera saidas latest + versionada
make llm-judge-live # gera respostas via /chat e avalia com 3 criterios
```

### Pre-commit hooks

O arquivo [.pre-commit-config.yaml](.pre-commit-config.yaml) configura hooks automáticos de qualidade:

```bash
pip install pre-commit
pre-commit install        # instala os hooks no repositório
pre-commit run --all-files  # executa manualmente em todos os arquivos
```

Hooks ativos incluem:

- qualidade de código: `ruff`, `ruff-format`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-merge-conflict`
- segurança de segredos: `detect-private-key`, `detect-secrets`
- checklist de segurança no commit: bloqueio de `.env`, `.env.*` (exceto `.env.example`), logs (`*.log`, `train_out.txt`, `train_err.txt`), `mlruns/` e artefatos binários em `models/`

Checklist rápido antes de commitar:

- confirme que nenhum segredo real foi adicionado em arquivos de código, docs ou YAML
- mantenha `.env` fora do controle de versão
- mantenha `.env.example` apenas com placeholders
- não versione logs operacionais nem artefatos binários de treino

Detalhes operacionais da política: [docs/OWASP_MAPPING.md](docs/OWASP_MAPPING.md).

### Variáveis de ambiente

Copie [.env.example](.env.example) e preencha com os valores reais:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `GOOGLE_API_KEY` | Chave da Google AI (obrigatória para o agente ReAct) |
| `APP_ENV` | Ambiente da aplicação (`development`, `staging`, `production`). Em `production`, a API faz fail-fast se `GOOGLE_API_KEY` estiver ausente, placeholder ou formato inválido |
| `GEMINI_LLM_MODEL` | Modelo LLM Gemini base (fallback para componentes de avaliação) |
| `GEMINI_EMBEDDING_MODEL` | Modelo de embeddings Gemini base |
| `GEMINI_TEMPERATURE` | Temperatura padrão global para componentes Gemini |
| `GEMINI_TOP_P` | Top-p padrão global para componentes Gemini (opcional) |
| `GEMINI_TOP_K` | Top-k padrão global para componentes Gemini (opcional) |
| `AGENT_LLM_MODEL` | Modelo Gemini do agente ReAct |
| `AGENT_LLM_TEMPERATURE` | Override de temperatura do agente ReAct |
| `AGENT_LLM_TOP_P` | Override de top-p do agente ReAct (opcional) |
| `AGENT_LLM_TOP_K` | Override de top-k do agente ReAct (opcional) |
| `RAGAS_LLM_MODEL` | Modelo Gemini usado no `evaluation/ragas_eval.py` |
| `RAGAS_EMBEDDING_MODEL` | Modelo de embeddings usado no `evaluation/ragas_eval.py` |
| `RAGAS_LLM_TEMPERATURE` | Override de temperatura no `evaluation/ragas_eval.py` |
| `RAGAS_LLM_TOP_P` | Override de top-p no `evaluation/ragas_eval.py` (opcional) |
| `RAGAS_LLM_TOP_K` | Override de top-k no `evaluation/ragas_eval.py` (opcional) |
| `LLM_JUDGE_MODEL` | Modelo Gemini usado no `evaluation/llm_judge.py` |
| `LLM_JUDGE_TEMPERATURE` | Override de temperatura no `evaluation/llm_judge.py` |
| `LLM_JUDGE_TOP_P` | Override de top-p no `evaluation/llm_judge.py` (opcional) |
| `LLM_JUDGE_TOP_K` | Override de top-k no `evaluation/llm_judge.py` (opcional) |
| `RAG_EMBEDDING_MODEL` | Modelo de embeddings usado no pipeline RAG do agente |
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow (ex.: PostgreSQL RDS) |
| `MLFLOW_EXPERIMENT_NAME` | Nome do experimento MLflow |
| `MLFLOW_ARTIFACT_URI` | (Opcional) URI de artefatos do experimento MLflow quando ele é criado pela primeira vez |
| `MLFLOW_MODEL_NAME` | Nome lógico único do modelo no Registry; controla agrupamento de versões, aliases (`champion`/`candidate`) e trilha de auditoria |
| `MLFLOW_MODEL_VERSION` | Tag semântica de versão registrada no run/modelo (ex.: `v1`, `v2`); facilita governança de release e rollback |
| `MLFLOW_OWNER` | Responsável pelo modelo (squad/pessoa); usado para accountability e gestão de incidentes |
| `MLFLOW_RISK_LEVEL` | Nível de risco do modelo (`low`, `medium`, `high`); suporta controles de aprovação e priorização de monitoramento |
| `MLFLOW_TRAINING_DATA_VERSION` | Referência documental para lineage de dados de treino. No pipeline atual, o valor efetivo é calculado automaticamente como `git_sha:dvc_data_hash` (imutável) |
| `AWS_REGION` | Região AWS para ECR/ECS |
| `ECR_REPOSITORY` | Nome do repositório ECR |
| `ECS_CLUSTER` | Cluster ECS para deploy |
| `LANGFUSE_PUBLIC_KEY` | Chave pública Langfuse (opcional) |
| `LANGFUSE_SECRET_KEY` | Chave secreta Langfuse (opcional) |
| `APP_PORT` | Porta da aplicação (padrão: `8000`) |

### Governança de metadata do treino (MLflow)

Variáveis recomendadas no `.env` para rastreabilidade e governança do treino:

```bash
MLFLOW_MODEL_NAME=btc_hourly_forecaster
MLFLOW_MODEL_VERSION=v1
MLFLOW_OWNER=ml-team
MLFLOW_RISK_LEVEL=medium
# Valor efetivo no treino: derivado automaticamente de lineage imutável
MLFLOW_TRAINING_DATA_VERSION=gitsha123:deadbeefcafebabe1234567890abcdef
```

Impacto de governança:

- `MLFLOW_MODEL_NAME`: define o namespace de versões e aliases do Registry (`champion`/`candidate`), evitando mistura entre famílias de modelo.
- `MLFLOW_MODEL_VERSION`: cria trilha de release semântica para auditoria de mudanças de arquitetura/hiperparâmetros.
- `MLFLOW_OWNER`: explicita ownership operacional para resposta a incidentes, aprovações e handoff.
- `MLFLOW_RISK_LEVEL`: sinaliza criticidade para políticas de revisão, observabilidade e cadência de revalidação.
- `training_data_version` (tag efetiva): vincula o modelo ao dataset exato usado no treino, garantindo reprodutibilidade e investigação forense.

## Telemetria de Qualidade LLM (Langfuse)

O agente ReAct é instrumentado com **[Langfuse](https://cloud.langfuse.com)** para rastreamento de qualidade das chamadas LLM. A integração é **opcional e não bloqueante**: o agente opera normalmente se as credenciais não estiverem definidas.

Para ativar, defina as variáveis no `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Métricas rastreadas por trace: faithfulness, relevância da resposta, latência por chamada LLM, contagem de tokens e sequência de ferramentas invocadas.

## Hardening de Segredos no Startup

Para reduzir risco de credenciais fracas em produção, o startup da API valida `GOOGLE_API_KEY` com política fail-fast quando `APP_ENV=production`.

Bloqueios em produção:

- chave ausente
- placeholder/insegura (ex.: `your-google-api-key`, `mock_key_para_testes`)
- formato incompatível com chave esperada

Exemplo de configuração local segura:

```bash
APP_ENV=development
GOOGLE_API_KEY=your-google-api-key
```

Exemplo de produção:

```bash
APP_ENV=production
GOOGLE_API_KEY=AIzaSy<valor-real>
```

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
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --expected-questions 21 \
  --seed 42

# Gerando respostas ao vivo via /chat (API rodando)
python evaluation/ragas_eval.py \
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --api-url http://localhost:8000

# Executando métricas RAGAS online com Gemini (consome cota)
python evaluation/ragas_eval.py \
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --expected-questions 21 \
  --seed 42 \
  --enable-live-ragas \
  --strict-ragas
```

Observações importantes para reprodutibilidade e validade:

- O script carrega automaticamente o arquivo `.env` na raiz do projeto.
- O golden set é validado com contagem exata de 21 casos (`--expected-questions 21`).
- A saída é salva de forma atômica em `evaluation/ragas_results.json`.
- O JSON de saída não aceita `NaN`/`inf` (validação estrita antes de salvar).

Comportamento do backend de métricas:

- Por padrão, o script usa fallback determinístico (`deterministic_offline_fallback`) mesmo que `GOOGLE_API_KEY` esteja definida, para evitar consumo acidental de cota.
- Se `--enable-live-ragas` estiver ativo e o backend executar normalmente, o resultado terá `metric_backend = ragas`.
- Se o backend online falhar e `--strict-ragas` não estiver ativo, o script volta para o fallback determinístico para evitar métricas inválidas.
- Para exigir RAGAS estrito, use `--enable-live-ragas --strict-ragas` em conjunto.

Modelo LLM usado na avaliação:

- Padrão: lê `RAGAS_LLM_MODEL`; se ausente, usa fallback `GEMINI_LLM_MODEL`; se ambos ausentes, usa `models/gemini-2.5-flash`.
- Embeddings: lê `RAGAS_EMBEDDING_MODEL`; se ausente, usa fallback `GEMINI_EMBEDDING_MODEL`; se ambos ausentes, usa `models/gemini-embedding-001` (compatível com `embedContent` no free tier atual).
- Sampling: lê `RAGAS_LLM_TEMPERATURE`, `RAGAS_LLM_TOP_P`, `RAGAS_LLM_TOP_K`; se ausentes, usa fallback global `GEMINI_TEMPERATURE`, `GEMINI_TOP_P`, `GEMINI_TOP_K`.
- Evite modelos que suportam apenas Interactions API, pois podem gerar erro `400 This model only supports Interactions API` no executor do RAGAS.

Interpretação das 4 métricas (escala 0 a 1, quanto maior melhor):

- `faithfulness`: quanto da resposta está suportado pelos contextos recuperados (evita alucinação).
- `answer_relevancy`: quanto a resposta é relevante para a pergunta e para a referência esperada.
- `context_precision`: proporção do contexto recuperado que é realmente útil para sustentar a resposta.
- `context_recall`: quanto da informação necessária (referência) foi coberta pelos contextos recuperados.

Leitura prática rápida:

- `faithfulness` baixa: resposta pode estar inventando ou extrapolando além dos contextos.
- `answer_relevancy` baixa: resposta tangencia o tema, mas não responde bem a pergunta.
- `context_precision` baixa: recuperação trouxe muito ruído.
- `context_recall` baixa: recuperação perdeu fatos importantes.

## Avaliação LLM-as-judge (3 critérios)

O avaliador em [evaluation/llm_judge.py](evaluation/llm_judge.py) usa LLM-as-judge com 3 critérios fixos para o golden set existente:

- `precisao_financeira` (1-5)
- `clareza` (1-5)
- `ausencia_alucinacoes` (1-5)

A nota final (`nota_final`) é calculada no veredito estruturado em escala 0-10, usando pesos 40/30/30 para os 3 critérios.

Execução com respostas já presentes no golden set:

```bash
python evaluation/llm_judge.py \
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --min-questions 21 \
  --output evaluation/llm_judge_results.json
```

Execução gerando respostas ao vivo via endpoint `/chat` (API rodando):

```bash
python evaluation/llm_judge.py \
  --golden-set data/golden_set/btc_rag_golden_set.json \
  --api-url http://127.0.0.1:8000 \
  --min-questions 21 \
  --output evaluation/llm_judge_results.json
```

Atalhos no Makefile:

```bash
make llm-judge
make llm-judge-live
```

### Saída consistente e versionada

Cada execução gera:

- saída estável (latest): `evaluation/llm_judge_results.json`
- saída versionada por execução: `evaluation/results/llm_judge/llm_judge_results_<timestamp>_<model>.json`

Campos estáveis no JSON de saída:

- `schema_version`
- `evaluation_type` (`llm_judge_3_criteria`)
- `criteria`
- `generated_at_utc`
- `run_config`
- `judge_backend_counts`
- `summary`
- `records`

Observações:

- O script carrega automaticamente o `.env` da raiz do projeto.
- O modelo juiz segue fallback: `LLM_JUDGE_MODEL` → `GEMINI_LLM_MODEL` → default interno.
- Se ocorrer erro de quota/429 ou indisponibilidade do Gemini, o script entra em fallback determinístico para manter a execução e gerar saída válida.
- Para forçar falha sem fallback, use `--strict-judge`.
- Para desativar a cópia versionada, use `--skip-versioned-output`.

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

## Automação Operacional de Drift

O endpoint `POST /admin/check-drift` agora atua como ponto de integração operacional para MLOps:

- executa detecção de drift (PSI)
- decide ação por threshold
- envia alerta quando `psi > 0.1`
- dispara trigger de retraining quando `psi > 0.2`
- registra evento operacional no MLflow (`drift_action`, `alert_sent`, `retrain_*`)

### Arquitetura mínima (local e testável)

1. Scheduler chama periodicamente `POST /admin/check-drift`
2. API calcula PSI e aplica política operacional
3. API registra no MLflow o resultado da automação
4. Se necessário:
   - alerta por webhook
   - trigger de retreino por comando local

Componentes implementados:

- Lógica operacional: [src/serving/drift_automation.py](src/serving/drift_automation.py)
- Scheduler local APScheduler: [scripts/run_drift_scheduler.py](scripts/run_drift_scheduler.py)
- Endpoint integrado: [src/app.py](src/app.py)

### Variáveis de ambiente da automação

| Variável | Default | Descrição |
|---|---|---|
| `DRIFT_WARNING_THRESHOLD` | `0.1` | Limite para alerta |
| `DRIFT_RETRAIN_THRESHOLD` | `0.2` | Limite para trigger de retreino |
| `DRIFT_CHECK_INTERVAL_HOURS` | `24` | Intervalo do scheduler |
| `DRIFT_ALERT_WEBHOOK_URL` | vazio | Webhook de alerta (Slack/Teams/etc.) |
| `DRIFT_RETRAIN_ENABLED` | `false` | Habilita execução real do retreino |
| `DRIFT_RETRAIN_COMMAND` | `python -u training/train_model.py` | Comando de retreino |
| `DRIFT_RETRAIN_TIMEOUT_SECONDS` | `900` | Timeout do comando de retreino |
| `DRIFT_AUTOMATION_API_URL` | `http://127.0.0.1:8000` | URL base da API para o scheduler |
| `DRIFT_AUTOMATION_TICKER` | `BTC-USD` | Ticker usado no scheduler |

### Como executar localmente

1. Suba a API:

```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

2. Rode um check manual:

```bash
make drift-check
```

3. Inicie agendamento periódico:

```bash
make drift-scheduler
```

### Observação de segurança operacional

Por padrão, `DRIFT_RETRAIN_ENABLED=false`, então a ação de retraining é registrada, mas não executa comando local até habilitação explícita.

## Observações

- A API utiliza cache curto e retry para chamadas ao Yahoo Finance.
- Com fallback para a Binance, a API mantém disponibilidade mesmo durante instabilidades da fonte primária.
- O healthcheck do Docker Compose usa `GET /live` para evitar impacto em cache de mercado.
- O pipeline treina em `log-return` e converte previsão para preço final.
- O histórico de previsões (`/predictions/history`) persiste em memória enquanto o container estiver ativo.
