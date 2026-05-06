# LGPD Plan

## Objetivo

Este plano descreve como o projeto reduz risco de exposição de dados pessoais no fluxo do Agente ReAct e da API, com foco no uso de `presidio_analyzer` e `presidio_anonymizer` na camada de guardrails.

## Escopo de Proteção

O sistema não foi desenhado para coletar PII como dado de negócio principal. Mesmo assim, entradas de usuários e saídas do agente podem conter dados pessoais acidentalmente. Por isso, o projeto aplica uma camada de sanitização na resposta do agente.

## Componentes Relevantes

- [src/security/guardrails.py](../src/security/guardrails.py)
- [src/delivery/api/routers/chat.py](../src/delivery/api/routers/chat.py)
- [tests/test_guardrails.py](../tests/test_guardrails.py)

## Como o Presidio Protege PII

`OutputGuardrail` encapsula dois motores:

- `AnalyzerEngine` do Presidio para detectar entidades sensíveis
- `AnonymizerEngine` do Presidio para anonimizar o texto detectado

O método `sanitize` executa a seguinte estratégia:

1. Inicializa os motores do Presidio sob demanda.
2. Analisa o texto com `score_threshold` configurável.
3. Anonimiza o texto detectado.
4. Aplica uma segunda camada de mascaramento por regex.

## Fallback por Regex

Mesmo quando o Presidio falha, o sistema ainda aplica mascaramento por regex para:

- email
- CPF com e sem máscara
- CNPJ com e sem máscara
- números de cartão
- telefone

Marcadores usados no output:

- `<EMAIL_MASKED>`
- `<CPF_MASKED>`
- `<CNPJ_MASKED>`
- `<CARD_MASKED>`
- `<PHONE_MASKED>`

## Onde a Sanitização é Aplicada

No endpoint `/chat` em [src/delivery/api/routers/chat.py](../src/delivery/api/routers/chat.py):

- observações intermediárias das ferramentas são sanitizadas
- resposta final do agente é sanitizada antes de retornar ao cliente

Isso reduz risco de vazamento acidental de PII tanto no raciocínio intermediário exposto quanto na resposta final.

## Evidências de Teste

Os testes em [tests/test_guardrails.py](../tests/test_guardrails.py) cobrem:

- bloqueio de prompt injection
- bloqueio de context stuffing
- fallback regex quando Presidio falha
- anonimização com analyzer/anonymizer simulados
- mascaramento mesmo sem entidades detectadas pelo analyzer

## Medidas LGPD Recomendadas

- Minimização de dados: evitar solicitar PII em prompts e formulários.
- Limitação de finalidade: usar dados apenas para operação da API e suporte à previsão de BTC.
- Segurança: manter segredos em AWS Secrets Manager.
- Transparência: documentar que o sistema mascara PII em respostas conversacionais.
- Retenção: evitar persistência de textos brutos contendo PII.

## Limitações

- A sanitização atual é aplicada na saída, não como pipeline formal de classificação e retenção de dados em toda a plataforma.
- O analyzer usa `language="en"`, o que pode reduzir cobertura para alguns padrões textuais em português.
- O histórico de previsões não foi desenhado para armazenar dados pessoais, mas políticas de retenção explícitas ainda devem ser formalizadas.
