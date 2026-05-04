# System Card

## Resumo

O sistema combina uma API FastAPI, um modelo LSTM de previsão horária de Bitcoin e um Agente ReAct com Gemini. A solução expõe previsão numérica, contexto de mercado, guardrails e observabilidade operacional.

## Componentes Principais

- API FastAPI em [src/app.py](src/app.py)
- Pipeline de treino em [training/train_model.py](training/train_model.py)
- Agente ReAct em [src/agent/react_agent.py](src/agent/react_agent.py)
- Guardrails de entrada e saída em [src/security/guardrails.py](src/security/guardrails.py)
- Infraestrutura como código em [infra/terraform/main.tf](infra/terraform/main.tf)

## Arquitetura Cloud

Com base no Terraform atual, a arquitetura AWS inclui:

- Amazon ECS/Fargate para execução da API
- Amazon ECR para armazenamento da imagem Docker
- Amazon RDS PostgreSQL para metadados do MLflow
- Amazon S3 para artefatos do modelo, DVC e MLflow
- AWS Secrets Manager para `GOOGLE_API_KEY` e senha do banco
- CloudWatch Logs para logs do serviço ECS
- Security Groups separados para ECS e RDS

## Fluxo de Dados

1. O treino baixa dados históricos de BTC por `yfinance`, com fallback para Binance.
2. O pipeline gera features técnicas, treina o modelo LSTM e salva artefatos em `models/`.
3. Em produção, a API carrega o modelo e scalers durante o `lifespan`.
4. O endpoint `/predict` busca dados de mercado, prepara a janela temporal e calcula a previsão do próximo fechamento horário.
5. O endpoint `/chat` usa um Agente ReAct com três ferramentas:
   - previsão LSTM
   - cotação atual
   - recuperação de contexto cripto via `CryptoKnowledgeRAG`
6. A saída final do agente passa por sanitização de PII antes de ser retornada.

## Agente ReAct

O agente usa `ChatGoogleGenerativeAI` com modelo configurável via `.env` (default: **`gemini-2.5-flash`** para free tier) e temperatura `0` (determinístico). A resolução de modelo segue cadeia: `AGENT_LLM_MODEL` (env) → `GEMINI_LLM_MODEL` (env) → hardcoded `gemini-2.5-flash`. Embeddings usam `gemini-embedding-001` (dimensão 3072, free tier). O agente orquestra ferramentas para responder perguntas sobre previsão, preço atual e contexto de mercado.

Ferramentas relevantes:

- `PrevisaoBitcoinTool`
- `CotacaoAtualTool`
- `CryptoKnowledgeRAG` (com `GoogleGenerativeAIEmbeddings`)
- Temperatura e parâmetros de sampling via `.env` (overrides: `AGENT_LLM_TEMPERATURE`, `AGENT_LLM_TOP_P`, `AGENT_LLM_TOP_K`)

## Guardrails

Implementados em [src/security/guardrails.py](src/security/guardrails.py) e aplicados em [src/app.py](src/app.py):

- `InputGuardrail`
  - bloqueia prompt injection por regex
  - bloqueia context stuffing acima de 4096 caracteres
  - rejeita input vazio
- `OutputGuardrail`
  - detecta e mascara PII com Presidio
  - usa fallback por regex para email, CPF, CNPJ, cartão e telefone

No fluxo de chat:

- a mensagem do usuário é validada antes da execução do agente
- observações intermediárias das ferramentas são sanitizadas
- a resposta final do agente também é sanitizada

## Observabilidade

### Métricas operacionais (`/metrics`)

Métricas expostas por `/metrics` (Prometheus):

- total de chamadas ao `/predict`
- latência de inferência
- fonte de dados utilizada
- falhas por fonte de mercado
- uso de CPU
- uso de memória

### Telemetria de Qualidade LLM — Langfuse

O agente ReAct é instrumentado com **[Langfuse](https://langfuse.com)** para rastreamento de qualidade das interações LLM. Quando as variáveis `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` estão configuradas, cada chamada ao agente produz um trace com:

| Métrica | Descrição |
|---|---|
| **Faithfulness** | Grau em que a resposta é fundamentada nas observações das ferramentas |
| **Relevância** | Alinhamento da resposta com a pergunta do usuário |
| **Latência** | Tempo total de execução do agente e tempo por LLM call |
| **Tokens** | Contagem de tokens de entrada e saída por chamada |
| **Ferramentas invocadas** | Sequência de tools utilizadas (LSTM, cotação, RAG) |

A telemetria é **opcional e não bloqueante**: se as credenciais não estiverem definidas ou o serviço estiver indisponível, o agente continua operando normalmente.

Configuração: ver `.env.example` para as variáveis `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` e `LANGFUSE_HOST`.

### Logs e histórico

- logs estruturados por componente
- histórico recente de previsões via `/predictions/history`

## Limitações de Sistema

- O serviço atende apenas `BTC-USD`.
- O RAG atual usa corpus simulado/local, não um feed contínuo de notícias em produção.
- O histórico de previsões é armazenado apenas em memória do processo.
- A qualidade final do agente depende tanto do modelo LSTM quanto da qualidade do contexto recuperado.

## Dependências Externas Críticas

- Gemini API
- Yahoo Finance
- Binance REST API
- Serviços AWS provisionados por Terraform
