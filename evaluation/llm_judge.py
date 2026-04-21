"""Avalia respostas com a abordagem LLM-as-judge usando LangChain.

O script pontua cada resposta em 3 criterios:
1. precisao_financeira
2. clareza
3. ausencia_alucinacoes

Ele aceita um golden set em JSON com respostas prontas ou, se --api-url for informado,
gera as respostas chamando POST /chat no servico da aplicacao.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

import requests
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("evaluation.llm_judge")
CONTEXT_SPLIT_PATTERN = re.compile(r"\[Contexto\s+\d+\]\s*", re.IGNORECASE)
DEFAULT_OUTPUT_PATH = Path("evaluation/llm_judge_results.json")


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    precisao_financeira: CriterionScore
    clareza: CriterionScore
    ausencia_alucinacoes: CriterionScore
    nota_final: float = Field(ge=0, le=10)
    resumo: str = Field(min_length=1)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _load_golden_set(golden_set_path: Path) -> list[dict[str, Any]]:
    with golden_set_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("O golden set deve ser uma lista JSON de objetos.")
    return data


def _normalize_contexts(raw_contexts: Any) -> list[str]:
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, list):
        return [str(item).strip() for item in raw_contexts if str(item).strip()]
    if isinstance(raw_contexts, str):
        cleaned = raw_contexts.strip()
        if not cleaned:
            return []
        if "[Contexto" in cleaned:
            return [chunk.strip() for chunk in CONTEXT_SPLIT_PATTERN.split(cleaned) if chunk.strip()]
        return [cleaned]
    raise TypeError("contexts deve ser string, lista de strings ou null.")


def _extract_question(item: dict[str, Any], index: int) -> str:
    question = item.get("question") or item.get("query")
    if not question:
        raise ValueError(f"Item {index} sem 'question' ou 'query'.")
    return str(question).strip()


def _extract_reference_answer(item: dict[str, Any], index: int) -> str:
    reference_answer = item.get("expected_answer") or item.get("ground_truth") or item.get("reference_answer")
    if not reference_answer:
        raise ValueError(f"Item {index} sem resposta de referencia ('expected_answer' ou 'ground_truth').")
    return str(reference_answer).strip()


def _extract_candidate_answer(item: dict[str, Any]) -> str:
    answer = item.get("answer") or item.get("generated_answer") or item.get("response")
    return str(answer).strip() if answer else ""


def _extract_contexts_from_item(item: dict[str, Any]) -> list[str]:
    for key in ("contexts", "retrieved_contexts", "reference_contexts", "expected_contexts"):
        contexts = _normalize_contexts(item.get(key))
        if contexts:
            return contexts
    return []


def _resolve_chat_url(api_url: str) -> str:
    normalized = api_url.rstrip("/")
    if normalized.endswith("/chat"):
        return normalized
    return f"{normalized}/chat"


def _extract_contexts_from_steps(steps: Any) -> list[str]:
    if not isinstance(steps, list):
        return []

    contexts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool", "")).lower()
        if "crypto_rag" not in tool_name and "rag" not in tool_name:
            continue
        contexts.extend(_normalize_contexts(step.get("observation")))

    unique_contexts: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        if context not in seen:
            unique_contexts.append(context)
            seen.add(context)
    return unique_contexts


def _fetch_chat_response(api_url: str, question: str, timeout_seconds: int) -> tuple[str, list[str], dict[str, Any]]:
    response = requests.post(
        _resolve_chat_url(api_url),
        json={"message": question},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    answer = str(payload.get("response", "")).strip()
    contexts = _extract_contexts_from_steps(payload.get("steps", []))
    return answer, contexts, payload


def _materialize_records(
    golden_set: list[dict[str, Any]],
    api_url: str | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for index, item in enumerate(golden_set, start=1):
        question = _extract_question(item, index)
        reference_answer = _extract_reference_answer(item, index)
        answer = _extract_candidate_answer(item)
        contexts = _extract_contexts_from_item(item)
        raw_payload: dict[str, Any] = {}

        if not answer and api_url:
            LOGGER.info("Gerando resposta do item %d/%d via endpoint de chat...", index, len(golden_set))
            answer, fetched_contexts, raw_payload = _fetch_chat_response(api_url, question, timeout_seconds)
            if fetched_contexts and not contexts:
                contexts = fetched_contexts

        if not answer:
            raise ValueError(
                f"Item {index} sem resposta candidata. Inclua 'answer' no JSON ou use --api-url."
            )

        records.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "candidate_answer": answer,
                "contexts": contexts,
                "raw_response": raw_payload,
            }
        )

    return records


def _build_judge_chain(model_name: str, temperature: float) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Voce e um avaliador senior de sistemas de IA para analise e previsao do Bitcoin.

Avalie a resposta candidata usando apenas estes criterios:
1. precisao_financeira: fidelidade tecnica sobre preco do BTC, direcao esperada, magnitude, horizonte temporal, riscos, incerteza, volatilidade e implicacoes financeiras.
2. clareza: objetividade, organizacao, legibilidade e capacidade de responder diretamente a pergunta sobre a previsao do BTC.
3. ausencia_alucinacoes: ausencia de afirmacoes nao suportadas pela resposta de referencia ou pelos contextos fornecidos, especialmente numeros, eventos de mercado, causalidades ou certezas indevidas.

Regras obrigatorias:
- Dê score inteiro de 1 a 5 para cada criterio.
- Use a resposta de referencia como baseline principal.
- Use os contextos apenas como evidencia adicional.
- Considere negativamente previsoes sem horizonte temporal claro, sem sinalizacao de risco ou com excesso de certeza para um ativo volatil como BTC.
- Considere positivamente respostas que explicam premissas, limitacoes do modelo e cenarios alternativos quando isso estiver alinhado com a referencia.
- Penalize afirmacoes categoricas sem suporte.
- A nota_final deve ficar entre 0 e 10 e refletir a media ponderada: 40% precisao_financeira, 30% clareza, 30% ausencia_alucinacoes.
- Responda no schema estruturado solicitado.
                """.strip(),
            ),
            (
                "human",
                """
Pergunta do usuario:
{question}

Contexto da tarefa:
Julgue uma resposta de um agente que responde perguntas sobre previsoes do Bitcoin.
Verifique se a resposta candidata permanece fiel a referencia e aos contextos ao falar de preco, tendencia, horizonte, risco e justificativas.

Resposta de referencia:
{reference_answer}

Contextos recuperados:
{contexts}

Resposta candidata:
{candidate_answer}
                """.strip(),
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return prompt | llm.with_structured_output(JudgeVerdict)


def evaluate_with_llm_judge(
    golden_set_path: Path,
    api_url: str | None = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    timeout_seconds: int = 60,
    min_questions: int = 20,
) -> dict[str, Any]:
    golden_set = _load_golden_set(golden_set_path)
    if len(golden_set) < min_questions:
        raise ValueError(
            f"Golden set invalido: esperado pelo menos {min_questions} perguntas, encontrado {len(golden_set)}."
        )

    records = _materialize_records(golden_set, api_url, timeout_seconds)
    judge_chain = _build_judge_chain(model_name=model_name, temperature=temperature)

    judged_records: list[dict[str, Any]] = []
    financial_scores: list[int] = []
    clarity_scores: list[int] = []
    hallucination_scores: list[int] = []
    overall_scores: list[float] = []

    for index, record in enumerate(records, start=1):
        LOGGER.info("Avaliando item %d/%d com LLM-as-judge...", index, len(records))
        verdict = judge_chain.invoke(
            {
                "question": record["question"],
                "reference_answer": record["reference_answer"],
                "candidate_answer": record["candidate_answer"],
                "contexts": "\n\n".join(record["contexts"]) if record["contexts"] else "Sem contexto adicional.",
            }
        )

        financial_scores.append(verdict.precisao_financeira.score)
        clarity_scores.append(verdict.clareza.score)
        hallucination_scores.append(verdict.ausencia_alucinacoes.score)
        overall_scores.append(verdict.nota_final)

        judged_records.append(
            {
                **record,
                "judge": verdict.model_dump(),
            }
        )

    summary = {
        "precisao_financeira_media": round(mean(financial_scores), 4),
        "clareza_media": round(mean(clarity_scores), 4),
        "ausencia_alucinacoes_media": round(mean(hallucination_scores), 4),
        "nota_final_media": round(mean(overall_scores), 4),
    }
    LOGGER.info("Resumo da avaliacao LLM-as-judge: %s", summary)

    return {
        "golden_set_path": str(golden_set_path),
        "sample_count": len(judged_records),
        "model": model_name,
        "summary": summary,
        "records": judged_records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia respostas com LLM-as-judge usando LangChain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--golden-set", required=True, type=Path, help="Caminho para o golden set JSON.")
    parser.add_argument(
        "--api-url",
        default=None,
        help="URL base da API FastAPI. O script chamara POST /chat quando answer nao estiver no JSON.",
    )
    parser.add_argument("--model", default="gemini-2.5-pro", help="Modelo Gemini usado como juiz.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura do LLM juiz.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout de chamada HTTP em segundos.")
    parser.add_argument("--min-questions", type=int, default=20, help="Quantidade minima de exemplos no golden set.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Arquivo JSON de saida com o detalhamento das notas.",
    )
    return parser.parse_args()


def _ensure_google_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY nao esta definida. Configure a chave antes de executar o avaliador LLM-as-judge."
        )


def main() -> int:
    _configure_logging()
    args = _parse_args()
    _ensure_google_api_key()

    result = evaluate_with_llm_judge(
        golden_set_path=args.golden_set,
        api_url=args.api_url,
        model_name=args.model,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        min_questions=args.min_questions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    LOGGER.info("Resultado salvo em %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())