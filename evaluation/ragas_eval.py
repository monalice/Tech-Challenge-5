"""Avalia um golden set com RAGAS usando 4 metricas obrigatorias.

Formato esperado do golden set em JSON:
[
  {
    "question": "..." ou "query": "...",
    "expected_answer": "..." ou "ground_truth": "...",
    "answer": "...",                       # opcional se --api-url for usado
    "contexts": ["ctx1", "ctx2"],         # opcional se --api-url for usado
    "expected_contexts": ["ctx1", "ctx2"] # fallback opcional
  }
]

Quando --api-url e informado, o script chama POST /chat para gerar respostas e tenta
extrair os contextos recuperados a partir do passo da ferramenta CryptoRAGTool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import requests
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragas import evaluate

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

LOGGER = logging.getLogger("evaluation.ragas")
CONTEXT_SPLIT_PATTERN = re.compile(r"\[Contexto\s+\d+\]\s*", re.IGNORECASE)
DEFAULT_OUTPUT_PATH = Path("evaluation/ragas_results.json")
DEFAULT_LLM_MODEL = "models/deep-research-preview-04-2026"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"


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
            chunks = CONTEXT_SPLIT_PATTERN.split(cleaned)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        return [cleaned]

    raise TypeError("contexts deve ser string, lista de strings ou null.")


def _extract_question(item: dict[str, Any], index: int) -> str:
    question = item.get("question") or item.get("query")
    if not question:
        raise ValueError(f"Item {index} sem 'question' ou 'query'.")
    return str(question).strip()


def _extract_ground_truth(item: dict[str, Any], index: int) -> str:
    ground_truth = item.get("ground_truth") or item.get("expected_answer") or item.get("reference_answer")
    if not ground_truth:
        raise ValueError(f"Item {index} sem 'expected_answer' ou 'ground_truth'.")
    return str(ground_truth).strip()


def _extract_precomputed_answer(item: dict[str, Any]) -> str:
    answer = item.get("answer") or item.get("generated_answer") or item.get("response")
    return str(answer).strip() if answer else ""


def _extract_precomputed_contexts(item: dict[str, Any]) -> list[str]:
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

    deduplicated: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        if context not in seen:
            deduplicated.append(context)
            seen.add(context)
    return deduplicated


def _materialize_records(
    golden_set: list[dict[str, Any]],
    api_url: str | None,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []

    for index, item in enumerate(golden_set, start=1):
        question = _extract_question(item, index)
        ground_truth = _extract_ground_truth(item, index)
        answer = _extract_precomputed_answer(item)
        contexts = _extract_precomputed_contexts(item)
        raw_payload: dict[str, Any] = {}

        if (not answer or not contexts) and api_url:
            LOGGER.info("Gerando resposta do item %d/%d via endpoint de chat...", index, len(golden_set))
            answer, fetched_contexts, raw_payload = _fetch_chat_response(api_url, question, timeout_seconds)
            if fetched_contexts:
                contexts = fetched_contexts

        if not answer:
            raise ValueError(
                f"Item {index} sem resposta gerada. Inclua 'answer' no JSON ou use --api-url para gerar respostas."
            )
        if not contexts:
            raise ValueError(
                f"Item {index} sem contextos recuperados. Inclua 'contexts' no JSON ou use --api-url com um agente que exponha passos do RAG."
            )

        record = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }
        records.append(record)
        raw_outputs.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
                "raw_response": raw_payload,
            }
        )

    return records, raw_outputs


def _coerce_metric_value(raw_value: Any) -> float:
    if isinstance(raw_value, list):
        numeric_values = [float(item) for item in raw_value]
        if not numeric_values:
            raise ValueError("A metrica retornou uma lista vazia.")
        return float(sum(numeric_values) / len(numeric_values))
    return float(raw_value)


def _extract_metrics(scores: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    if hasattr(scores, "to_pandas"):
        score_frame = scores.to_pandas()
        for metric_name in metric_names:
            if metric_name not in score_frame.columns:
                raise KeyError(f"Metrica ausente no resultado RAGAS: {metric_name}")
            metrics[metric_name] = float(score_frame[metric_name].dropna().mean())
        return metrics

    score_mapping = cast(Any, scores)
    for metric_name in metric_names:
        metrics[metric_name] = _coerce_metric_value(score_mapping[metric_name])
    return metrics


def _build_gemini_clients() -> tuple[ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings]:
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return llm, embeddings


def evaluate_golden_set(
    golden_set_path: Path,
    api_url: str | None = None,
    timeout_seconds: int = 60,
    min_questions: int = 20,
) -> dict[str, Any]:
    golden_set = _load_golden_set(golden_set_path)
    if len(golden_set) < min_questions:
        raise ValueError(
            f"Golden set invalido: esperado pelo menos {min_questions} perguntas, encontrado {len(golden_set)}."
        )

    records, raw_outputs = _materialize_records(golden_set, api_url, timeout_seconds)
    dataset = Dataset.from_list(records)
    llm, embeddings = _build_gemini_clients()

    LOGGER.info("Executando avaliacao RAGAS com %d exemplos...", len(records))
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    metrics = _extract_metrics(scores)
    LOGGER.info("Metricas RAGAS calculadas: %s", metrics)

    return {
        "golden_set_path": str(golden_set_path),
        "sample_count": len(records),
        "metrics": metrics,
        "records": raw_outputs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia um golden set com RAGAS usando faithfulness, answer_relevancy, context_precision e context_recall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--golden-set", required=True, type=Path, help="Caminho para o golden set JSON.")
    parser.add_argument(
        "--api-url",
        default=None,
        help="URL base da API FastAPI. O script chamara POST /chat quando answer/contexts nao estiverem no JSON.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout de chamada HTTP em segundos.")
    parser.add_argument("--min-questions", type=int, default=20, help="Quantidade minima de exemplos no golden set.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Arquivo JSON de saida com metricas agregadas e exemplos avaliados.",
    )
    return parser.parse_args()


def _ensure_google_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY nao esta definida. Configure a chave antes de executar a avaliacao RAGAS."
        )


def main() -> int:
    _configure_logging()
    args = _parse_args()
    _ensure_google_api_key()
    result = evaluate_golden_set(
        golden_set_path=args.golden_set,
        api_url=args.api_url,
        timeout_seconds=args.timeout,
        min_questions=args.min_questions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    LOGGER.info("Resultado salvo em %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())