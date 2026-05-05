# Armazenamento de Modelos em S3

## Visão Geral

Os modelos treinados agora são persistidos automaticamente em um bucket AWS S3 (`tech-challenge-5-dev-artifacts-db6c23fb`), além do armazenamento local de fallback e do registro no MLflow.

**Fluxo:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ training/train_model.py                                             │
│ - Treina modelo LSTM                                                │
│ - Registra no MLflow Registry (artefatos MLflow)                    │
│ - NOVO: Salva em S3 também (se S3_MODELS_BUCKET configurado)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
         ┌──────────────────┐  ┌──────────────────┐
         │ MLflow Registry  │  │ AWS S3 Bucket    │
         │ (mlflow-artifacts│  │ (tech-challenge- │
         │  ou local)       │  │ 5-trained-models)│
         └──────────────────┘  └──────────────────┘
                    ↑                   ↑
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ src/delivery/api/lifespan.py (API startup)                          │
│ - Carrega modelo de S3 (ou fallback local se indisponível)         │
│ - Carrega scalers de S3                                             │
│ - Carrega metadata de S3                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuração

### Variáveis de Ambiente

Defina em `.env` ou secretos do ambiente:

```bash
# Bucket S3 para modelos treinados (obrigatório para usar S3)
S3_MODELS_BUCKET=tech-challenge-5-dev-artifacts-db6c23fb

# Credenciais AWS (use uma das opções abaixo)
# Opção 1: Credenciais explícitas (dev/teste)
AWS_ACCESS_KEY_ID=sua-chave
AWS_SECRET_ACCESS_KEY=sua-chave-secreta
AWS_DEFAULT_REGION=us-east-1

# Opção 2: IAM Role (produção, recomendado)
# Não defina as credenciais acima; use IAM role da instância/container
```

### Exemplo de Deploy

**Docker (ECS/Fargate):**

```dockerfile
# Dockerfile já instala boto3 via requirements.txt
ENV S3_MODELS_BUCKET=tech-challenge-5-dev-artifacts-db6c23fb
# IAM role configurada no ECS task definition
```

**Docker Compose (local/dev):**

```yaml
services:
  api:
    environment:
      S3_MODELS_BUCKET: tech-challenge-5-dev-artifacts-db6c23fb
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      AWS_DEFAULT_REGION: us-east-1
```

## Comportamento

### Com S3 Habilitado

Se `S3_MODELS_BUCKET` é definido e as credenciais AWS estão disponíveis:

1. **Treino (`train_model.py`):**
   - Salva modelo Keras em: `s3://{S3_MODELS_BUCKET}/models/lstm_btc_hourly.keras`
   - Salva scaler em: `s3://{S3_MODELS_BUCKET}/models/scaler_btc.gz`
   - Salva scaler_return em: `s3://{S3_MODELS_BUCKET}/models/scaler_btc_return.gz`
   - Salva metadata em: `s3://{S3_MODELS_BUCKET}/models/model_metadata_btc.json`
   - **Também** registra em MLflow (não é substituído)

2. **API Startup (`lifespan.py`):**
   - Tenta carregar de S3
   - Se falhar (S3 indisponível), usa fallback local (`models/`)

### Sem S3 (Fallback Local)

Se `S3_MODELS_BUCKET` está vazio ou indefinido:

- Modelos são salvos localmente em `models/` (comportamento anterior)
- Nenhuma dependência de AWS
- Útil para development local

### Tratamento de Erros

**Erro ao salvar em S3:**
```
[ERROR] training/train_model.py: Erro ao salvar artefatos em S3 (continuando com MLflow): [details]
```
- Não bloqueia o treino; continua com MLflow
- Modelos ainda disponíveis via MLflow

**Erro ao carregar de S3:**
```
[ERROR] src/adapters/ml/model_loader.py: Erro ao carregar modelo de S3: ...
```
- Tenta fallback local
- Se local também falhar, API startup falha

## Estrutura do Bucket S3

```
tech-challenge-5-dev-artifacts-db6c23fb/
├── models/
│   ├── lstm_btc_hourly.keras           # Modelo Keras
│   ├── scaler_btc.gz                   # Scaler features
│   ├── scaler_btc_return.gz            # Scaler target
│   └── model_metadata_btc.json         # Metadata
├── [versões anteriores podem ficar aqui via DVC ou manual cleanup]
```

## Permissões de IAM

Para produção (ECS task role), adicione policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::tech-challenge-5-dev-artifacts-db6c23fb/models/*"
    }
  ]
}
```

## Arquivos Modificados

- `src/adapters/ml/s3_model_manager.py` — **Novo**: Gerenciador de S3
- `src/adapters/ml/model_loader.py` — Atualizado para suportar S3
- `src/delivery/api/lifespan.py` — Carrega do S3 no startup
- `src/domain/constants.py` — Constantes de S3
- `training/train_model.py` — Salva em S3 após treino
- `.env.example` — Documenta variáveis

## Próximas Melhorias

1. **Versionamento:** Usar timestamps ou hashes para múltiplas versões
2. **Cleanup:** Política de retenção (manter N últimas versões)
3. **Sincronização:** Sincronizar modelo local com S3 periodicamente
4. **Observabilidade:** Logs estruturados + métricas CloudWatch
