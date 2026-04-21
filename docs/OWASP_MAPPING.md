# OWASP Mapping

## Resumo

Este documento mapeia ameaças relevantes do OWASP Top 10 para aplicações com LLM ao código atual do projeto e às mitigações já implementadas.

## 1. Prompt Injection

Risco:
Usuários tentam sobrescrever instruções do sistema ou induzir o agente a ignorar políticas.

Exemplos:

- `ignore previous instructions`
- `ignore as instruções anteriores`
- pedidos para revelar system prompt

Mitigação no código:

- [src/security/guardrails.py](src/security/guardrails.py): `InputGuardrail` usa regex para padrões de prompt injection
- [src/app.py](src/app.py): valida a mensagem antes de executar o agente
- [tests/test_guardrails.py](tests/test_guardrails.py): cobre bloqueio explícito de prompt injection

## 2. Sensitive Information Disclosure

Risco:
O agente ou as ferramentas podem retornar PII presente na entrada, em observações intermediárias ou em respostas geradas.

Mitigação no código:

- [src/security/guardrails.py](src/security/guardrails.py): `OutputGuardrail` com Presidio + regex fallback
- [src/app.py](src/app.py): sanitiza observações intermediárias e resposta final
- mascaramento de email, CPF, CNPJ, telefone e cartão

## 3. Training Data / Retrieval Data Poisoning

Risco:
Documentos de recuperação ou contexto externo podem introduzir informação enganosa no agente.

Mitigação parcial no código:

- [src/agent/rag_pipeline.py](src/agent/rag_pipeline.py): corpus local e simulado, reduzindo dependência de fontes arbitrárias em runtime
- [src/agent/react_agent.py](src/agent/react_agent.py): `CryptoKnowledgeRAG` usa vector store local controlado pelo projeto

Lacuna:

- ainda não há pipeline formal de curadoria, versionamento e aprovação de documentos RAG em produção

## 4. Denial of Service via Context Stuffing

Risco:
Entradas grandes podem degradar performance, custo e estabilidade do agente.

Mitigação no código:

- [src/security/guardrails.py](src/security/guardrails.py): limite de 4096 caracteres por input
- [tests/test_guardrails.py](tests/test_guardrails.py): valida bloqueio por excesso de tamanho
- [src/app.py](src/app.py): rejeita a entrada antes da execução do agente

## 5. Excessive Agency / Unsafe Tool Use

Risco:
O agente pode usar ferramentas de forma inadequada, confiar demais em contexto fraco ou responder com excesso de certeza.

Mitigação parcial no código:

- [src/agent/react_agent.py](src/agent/react_agent.py): conjunto pequeno e específico de ferramentas
- ferramentas limitadas a previsão, cotação e RAG local
- temperatura `0` no LLM para maior determinismo operacional
- resposta final pode ser auditada com passos intermediários no `/chat`

Lacuna:

- ainda não há política formal de autorização por ferramenta nem validação semântica de outputs das ferramentas

## 6. Supply Chain / External Dependency Risk

Risco:
Dependências de provedores externos e bibliotecas podem falhar ou alterar comportamento do sistema.

Mitigação parcial no código:

- [requirements.txt](requirements.txt): versões pinadas ou faixas controladas em partes críticas
- [src/app.py](src/app.py): fallback de mercado entre Yahoo Finance e Binance
- [infra/terraform/main.tf](infra/terraform/main.tf): segredos armazenados em Secrets Manager

## Resumo de Cobertura

Mitigações já implementadas cobrem melhor:

- prompt injection
- divulgação de PII
- context stuffing
- redução de superfície de ferramentas

Mitigações ainda maduras apenas parcialmente:

- governança do corpus RAG
- hardening de supply chain
- autorização detalhada de ferramentas
- validação robusta de outputs de ferramentas
