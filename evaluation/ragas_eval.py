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
import math
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import requests
from datasets import Dataset

if TYPE_CHECKING:
    from langchain_aws import BedrockEmbeddings, ChatBedrock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("evaluation.ragas")
CONTEXT_SPLIT_PATTERN = re.compile(r"\[Contexto\s+\d+\]\s*", re.IGNORECASE)
DEFAULT_OUTPUT_PATH = Path("evaluation/ragas_results.json")
DEFAULT_LLM_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_EXPECTED_QUESTIONS = 21
DEFAULT_SEED = 42
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9à-ÿÀ-Ÿ_]+", re.UNICODE)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def _load_golden_set(golden_set_path: Path) -> list[dict[str, Any]]:
    with golden_set_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("O golden set deve ser uma lista JSON de objetos.")
    return data


def _get_env_optional_float(primary_key: str, fallback_key: str | None = None) -> float | None:
    raw_value = os.getenv(primary_key)
    if (raw_value is None or raw_value.strip() == "") and fallback_key:
        raw_value = os.getenv(fallback_key)
    if raw_value is None or raw_value.strip() == "":
        return None
    return float(raw_value)


def _get_env_optional_int(primary_key: str, fallback_key: str | None = None) -> int | None:
    raw_value = os.getenv(primary_key)
    if (raw_value is None or raw_value.strip() == "") and fallback_key:
        raw_value = os.getenv(fallback_key)
    if raw_value is None or raw_value.strip() == "":
        return None
    return int(raw_value)


def _resolve_ragas_llm_model() -> str:
    return os.getenv("RAGAS_LLM_MODEL") or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_LLM_MODEL


def _resolve_ragas_embedding_model() -> str:
    return os.getenv("RAGAS_EMBEDDING_MODEL") or os.getenv("BEDROCK_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _resolve_ragas_temperature() -> float:
    value = _get_env_optional_float("RAGAS_LLM_TEMPERATURE", "AGENT_LLM_TEMPERATURE")
    return value if value is not None else DEFAULT_TEMPERATURE


def _resolve_ragas_top_p() -> float | None:
    return _get_env_optional_float("RAGAS_LLM_TOP_P", "AGENT_LLM_TOP_P")


def _resolve_ragas_top_k() -> int | None:
    return _get_env_optional_int("RAGAS_LLM_TOP_K", "AGENT_LLM_TOP_K")


def _resolve_bedrock_region() -> str | None:
    return (
        os.getenv("BEDROCK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _safe_overlap_ratio(numerator_tokens: set[str], denominator_tokens: set[str]) -> float:
    if not denominator_tokens:
        return 0.0
    return float(len(numerator_tokens & denominator_tokens) / len(denominator_tokens))


def _compute_offline_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    per_metric_values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for record in records:
        question_tokens = _tokenize(str(record.get("question", "")))
        answer_tokens = _tokenize(str(record.get("answer", "")))
        ground_truth_tokens = _tokenize(str(record.get("ground_truth", "")))

        raw_contexts = record.get("contexts", [])
        context_text = " ".join(str(ctx) for ctx in raw_contexts)
        context_tokens = _tokenize(context_text)

        # Approximation for semantic groundedness with deterministic lexical overlap.
        faithfulness_score = _safe_overlap_ratio(answer_tokens, context_tokens)
        answer_relevancy_score = _safe_overlap_ratio(answer_tokens, question_tokens | ground_truth_tokens)
        context_precision_score = _safe_overlap_ratio(context_tokens, ground_truth_tokens)
        context_recall_score = _safe_overlap_ratio(ground_truth_tokens, context_tokens)

        per_metric_values["faithfulness"].append(faithfulness_score)
        per_metric_values["answer_relevancy"].append(answer_relevancy_score)
        per_metric_values["context_precision"].append(context_precision_score)
        per_metric_values["context_recall"].append(context_recall_score)

    metrics: dict[str, float] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    for metric_name, values in per_metric_values.items():
        metric_value, metric_diagnostic = _aggregate_metric_values(values, metric_name)
        metrics[metric_name] = metric_value
        diagnostics[metric_name] = metric_diagnostic

    return metrics, diagnostics


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
    return " ".join(str(question).split())


def _extract_ground_truth(item: dict[str, Any], index: int) -> str:
    ground_truth = item.get("ground_truth") or item.get("expected_answer") or item.get("reference_answer")
    if not ground_truth:
        raise ValueError(f"Item {index} sem 'expected_answer' ou 'ground_truth'.")
    return " ".join(str(ground_truth).split())


def _extract_precomputed_answer(item: dict[str, Any]) -> str:
    answer = item.get("answer") or item.get("generated_answer") or item.get("response")
    return " ".join(str(answer).split()) if answer else ""


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


def _aggregate_metric_values(raw_values: list[Any], metric_name: str) -> tuple[float, dict[str, int]]:
    finite_values: list[float] = []
    invalid_count = 0
    for value in raw_values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            invalid_count += 1
            continue

        if math.isfinite(numeric_value):
            finite_values.append(numeric_value)
        else:
            invalid_count += 1

    if not finite_values:
        raise ValueError(
            f"Metrica '{metric_name}' invalida: todos os valores retornados foram NaN/inf."
        )

    return float(sum(finite_values) / len(finite_values)), {
        "total": len(raw_values),
        "valid": len(finite_values),
        "invalid": invalid_count,
    }


def _extract_metrics(scores: Any) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    metrics: dict[str, float] = {}
    diagnostics: dict[str, dict[str, int]] = {}

    if hasattr(scores, "to_pandas"):
        score_frame = scores.to_pandas()
        for metric_name in METRIC_NAMES:
            if metric_name not in score_frame.columns:
                raise KeyError(f"Metrica ausente no resultado RAGAS: {metric_name}")
            raw_values = score_frame[metric_name].tolist()
            metric_value, metric_diagnostic = _aggregate_metric_values(raw_values, metric_name)
            metrics[metric_name] = metric_value
            diagnostics[metric_name] = metric_diagnostic
        return metrics, diagnostics

    score_mapping = cast(Any, scores)
    for metric_name in METRIC_NAMES:
        raw_metric_value = _coerce_metric_value(score_mapping[metric_name])
        metric_value, metric_diagnostic = _aggregate_metric_values([raw_metric_value], metric_name)
        metrics[metric_name] = metric_value
        diagnostics[metric_name] = metric_diagnostic
    return metrics, diagnostics


def _validate_aggregated_metrics(metrics: dict[str, float]) -> None:
    for metric_name in METRIC_NAMES:
        value = metrics.get(metric_name)
        if value is None:
            raise KeyError(f"Metrica agregada ausente: {metric_name}")
        if not math.isfinite(value):
            raise ValueError(f"Metrica agregada invalida para '{metric_name}': {value}")


def _write_json_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp_path.replace(output_path)


def _load_ragas_runtime() -> tuple[Any, Any, list[Any]]:
    from ragas import RunConfig, evaluate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return RunConfig, evaluate, [faithfulness, answer_relevancy, context_precision, context_recall]


def _build_bedrock_clients(
    llm_model: str,
    embedding_model: str,
    region_name: str,
) -> tuple["ChatBedrock", "BedrockEmbeddings"]:
    from langchain_aws import BedrockEmbeddings, ChatBedrock

    llm_kwargs: dict[str, Any] = {
        "model_id": llm_model,
        "region_name": region_name,
        "model_kwargs": {"temperature": _resolve_ragas_temperature()},
    }
    top_p = _resolve_ragas_top_p()
    if top_p is not None:
        llm_kwargs["model_kwargs"]["top_p"] = top_p
    top_k = _resolve_ragas_top_k()
    if top_k is not None:
        llm_kwargs["model_kwargs"]["top_k"] = top_k

    llm = ChatBedrock(**llm_kwargs)
    embeddings = BedrockEmbeddings(model_id=embedding_model, region_name=region_name)
    return llm, embeddings


def evaluate_golden_set(
    golden_set_path: Path,
    api_url: str | None = None,
    timeout_seconds: int = 60,
    expected_questions: int = DEFAULT_EXPECTED_QUESTIONS,
    seed: int = DEFAULT_SEED,
    enable_live_ragas: bool = False,
    strict_ragas: bool = False,
) -> dict[str, Any]:
    _set_reproducibility(seed)
    golden_set = _load_golden_set(golden_set_path)
    if len(golden_set) != expected_questions:
        raise ValueError(
            "Golden set invalido: "
            f"esperado exatamente {expected_questions} perguntas, encontrado {len(golden_set)}."
        )

    records, raw_outputs = _materialize_records(golden_set, api_url, timeout_seconds)
    dataset = Dataset.from_list(records)
    backend = "ragas"
    bedrock_region = _resolve_bedrock_region()
    llm_model = _resolve_ragas_llm_model()
    embedding_model = _resolve_ragas_embedding_model()
    metrics: dict[str, float]
    diagnostics: dict[str, dict[str, int]]

    if bedrock_region and enable_live_ragas:
        try:
            RunConfig, evaluate, ragas_metrics = _load_ragas_runtime()
            llm, embeddings = _build_bedrock_clients(
                llm_model=llm_model,
                embedding_model=embedding_model,
                region_name=bedrock_region,
            )

            # max_workers=1 serializes requests to stay within the 15 RPM free-tier quota.
            run_config = RunConfig(
                timeout=180,
                max_retries=6,
                max_wait=120,
                max_workers=1,
                seed=seed,
            )
            LOGGER.info(
                "Executando avaliacao RAGAS com %d exemplos (max_workers=%d)...",
                len(records),
                run_config.max_workers,
            )
            scores = evaluate(
                dataset,
                metrics=ragas_metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            metrics, diagnostics = _extract_metrics(scores)
            _validate_aggregated_metrics(metrics)
            LOGGER.info("Metricas RAGAS calculadas: %s", metrics)
        except Exception as error:
            if strict_ragas:
                raise
            backend = "deterministic_offline_fallback"
            LOGGER.warning(
                "Falha na avaliacao RAGAS (%s). Usando fallback deterministico para evitar NaN.",
                error,
            )
            metrics, diagnostics = _compute_offline_metrics(records)
            _validate_aggregated_metrics(metrics)
    else:
        if strict_ragas and not enable_live_ragas:
            raise EnvironmentError(
                "A avaliacao RAGAS com LLM foi desabilitada por padrao para evitar consumo acidental de cota. "
                "Use --enable-live-ragas junto com --strict-ragas para executar chamadas reais ao Amazon Bedrock."
            )
        if strict_ragas:
            raise EnvironmentError(
                "BEDROCK_AWS_REGION/AWS_REGION nao esta definida e --strict-ragas foi habilitado."
            )
        backend = "deterministic_offline_fallback"
        LOGGER.warning(
            "Avaliacao RAGAS online desabilitada ou regiao AWS ausente. Usando fallback deterministico para gerar metricas validas."
        )
        metrics, diagnostics = _compute_offline_metrics(records)
        _validate_aggregated_metrics(metrics)

    return {
        "golden_set_path": str(golden_set_path),
        "sample_count": len(records),
        "reproducibility": {
            "seed": seed,
            "expected_questions": expected_questions,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "metric_backend": backend,
        },
        "metrics": metrics,
        "metric_diagnostics": diagnostics,
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
    parser.add_argument(
        "--expected-questions",
        type=int,
        default=DEFAULT_EXPECTED_QUESTIONS,
        help="Quantidade esperada de exemplos no golden set para execucao reprodutivel.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed para reproducibilidade.")
    parser.add_argument(
        "--enable-live-ragas",
        action="store_true",
        help="Habilita chamadas reais ao Amazon Bedrock para calcular metricas RAGAS. Sem este flag, o script usa fallback deterministico para evitar consumo acidental de cota.",
    )
    parser.add_argument(
        "--strict-ragas",
        action="store_true",
        help="Falha a execucao se a avaliacao RAGAS online nao puder ser executada. Requer --enable-live-ragas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Arquivo JSON de saida com metricas agregadas e exemplos avaliados.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    _load_dotenv_file(PROJECT_ROOT / ".env")
    args = _parse_args()
    result = evaluate_golden_set(
        golden_set_path=args.golden_set,
        api_url=args.api_url,
        timeout_seconds=args.timeout,
        expected_questions=args.expected_questions,
        seed=args.seed,
        enable_live_ragas=args.enable_live_ragas,
        strict_ragas=args.strict_ragas,
    )

    _write_json_atomic(args.output, result)

    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    LOGGER.info("Resultado salvo em %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())