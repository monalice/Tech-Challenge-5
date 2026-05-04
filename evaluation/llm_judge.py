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
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, cast

import requests
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("evaluation.llm_judge")
CONTEXT_SPLIT_PATTERN = re.compile(r"\[Contexto\s+\d+\]\s*", re.IGNORECASE)
DEFAULT_OUTPUT_PATH = Path("evaluation/llm_judge_results.json")
DEFAULT_VERSIONED_OUTPUT_DIR = Path("evaluation/results/llm_judge")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_JUDGE_TEMPERATURE = 0.0
RESULT_SCHEMA_VERSION = "1.0"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9à-ÿÀ-Ÿ_]+", re.UNICODE)


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


def _resolve_judge_model() -> str:
    return os.getenv("LLM_JUDGE_MODEL") or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_JUDGE_MODEL


def _resolve_judge_temperature() -> float:
    value = _get_env_optional_float("LLM_JUDGE_TEMPERATURE")
    return value if value is not None else DEFAULT_JUDGE_TEMPERATURE


def _resolve_judge_top_p() -> float | None:
    return _get_env_optional_float("LLM_JUDGE_TOP_P")


def _resolve_judge_top_k() -> int | None:
    return _get_env_optional_int("LLM_JUDGE_TOP_K")


def _resolve_bedrock_region() -> str | None:
    return (
        os.getenv("BEDROCK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )


def _safe_filename_component(value: str) -> str:
    normalized = value.replace("models/", "").strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "_", normalized).strip("_") or "unknown-model"


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
            return [
                chunk.strip() for chunk in CONTEXT_SPLIT_PATTERN.split(cleaned) if chunk.strip()
            ]
        return [cleaned]
    raise TypeError("contexts deve ser string, lista de strings ou null.")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _safe_overlap_ratio(candidate_tokens: set[str], baseline_tokens: set[str]) -> float:
    if not baseline_tokens:
        return 0.0
    return len(candidate_tokens & baseline_tokens) / len(baseline_tokens)


def _ratio_to_score(ratio: float) -> int:
    if ratio >= 0.8:
        return 5
    if ratio >= 0.6:
        return 4
    if ratio >= 0.4:
        return 3
    if ratio >= 0.2:
        return 2
    return 1


def _build_fallback_verdict(
    question: str,
    reference_answer: str,
    candidate_answer: str,
    contexts: list[str],
) -> JudgeVerdict:
    question_tokens = _tokenize(question)
    reference_tokens = _tokenize(reference_answer)
    candidate_tokens = _tokenize(candidate_answer)
    context_tokens = _tokenize(" ".join(contexts))

    financial_ratio = _safe_overlap_ratio(candidate_tokens, reference_tokens | context_tokens)
    clarity_ratio = _safe_overlap_ratio(candidate_tokens, question_tokens | reference_tokens)

    if candidate_tokens:
        supported = len(candidate_tokens & (reference_tokens | context_tokens)) / len(
            candidate_tokens
        )
    else:
        supported = 0.0

    hallucination_safety = max(0.0, min(1.0, supported))

    precisao_score = _ratio_to_score(financial_ratio)
    clareza_score = _ratio_to_score(clarity_ratio)
    ausencia_score = _ratio_to_score(hallucination_safety)

    weighted_score = (0.4 * precisao_score) + (0.3 * clareza_score) + (0.3 * ausencia_score)
    nota_final = round(weighted_score * 2, 4)

    return JudgeVerdict(
        precisao_financeira=CriterionScore(
            score=precisao_score,
            rationale=(
                "Fallback deterministico: score baseado na sobreposicao lexical entre "
                "resposta candidata, referencia e contextos."
            ),
        ),
        clareza=CriterionScore(
            score=clareza_score,
            rationale=(
                "Fallback deterministico: score baseado na cobertura lexical da pergunta "
                "e da resposta de referencia."
            ),
        ),
        ausencia_alucinacoes=CriterionScore(
            score=ausencia_score,
            rationale=(
                "Fallback deterministico: score baseado na proporcao de tokens da resposta "
                "que aparecem na referencia e nos contextos."
            ),
        ),
        nota_final=nota_final,
        resumo=(
            "Avaliacao concluida em modo fallback deterministico por indisponibilidade "
            "temporaria do backend LLM."
        ),
    )


def _extract_question(item: dict[str, Any], index: int) -> str:
    question = item.get("question") or item.get("query")
    if not question:
        raise ValueError(f"Item {index} sem 'question' ou 'query'.")
    return str(question).strip()


def _extract_reference_answer(item: dict[str, Any], index: int) -> str:
    reference_answer = (
        item.get("expected_answer") or item.get("ground_truth") or item.get("reference_answer")
    )
    if not reference_answer:
        raise ValueError(
            f"Item {index} sem resposta de referencia ('expected_answer' ou 'ground_truth')."
        )
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


def _fetch_chat_response(
    api_url: str, question: str, timeout_seconds: int
) -> tuple[str, list[str], dict[str, Any]]:
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
            LOGGER.info(
                "Gerando resposta do item %d/%d via endpoint de chat...", index, len(golden_set)
            )
            answer, fetched_contexts, raw_payload = _fetch_chat_response(
                api_url, question, timeout_seconds
            )
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


def _build_judge_chain(
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Voce e um avaliador senior de sistemas de IA para analise e previsao do Bitcoin.

Avalie a resposta candidata usando apenas estes criterios:
1. precisao_financeira: fidelidade tecnica sobre preco do BTC, direcao
esperada, magnitude, horizonte temporal, riscos, incerteza, volatilidade
e implicacoes financeiras.
2. clareza: objetividade, organizacao, legibilidade e capacidade de
responder diretamente a pergunta sobre a previsao do BTC.
3. ausencia_alucinacoes: ausencia de afirmacoes nao suportadas pela
resposta de referencia ou pelos contextos fornecidos, especialmente
numeros, eventos de mercado, causalidades ou certezas indevidas.

Regras obrigatorias:
- Dê score inteiro de 1 a 5 para cada criterio.
- Use a resposta de referencia como baseline principal.
- Use os contextos apenas como evidencia adicional.
- Considere negativamente previsoes sem horizonte temporal claro, sem
    sinalizacao de risco ou com excesso de certeza para um ativo volatil
    como BTC.
- Considere positivamente respostas que explicam premissas, limitacoes do
    modelo e cenarios alternativos quando isso estiver alinhado com a
    referencia.
- Penalize afirmacoes categoricas sem suporte.
- A nota_final deve ficar entre 0 e 10 e refletir a media ponderada:
    40% precisao_financeira, 30% clareza, 30% ausencia_alucinacoes.
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
Verifique se a resposta candidata permanece fiel a referencia e aos contextos
ao falar de preco, tendencia, horizonte, risco e justificativas.

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

    region_name = _resolve_bedrock_region()
    if not region_name:
        raise OSError(
            "Região AWS para Bedrock ausente. Defina BEDROCK_AWS_REGION, "
            "AWS_REGION ou AWS_DEFAULT_REGION."
        )

    model_kwargs: dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        model_kwargs["top_p"] = top_p
    if top_k is not None:
        model_kwargs["top_k"] = top_k

    chat_bedrock_cls = cast(Any, ChatBedrock)
    llm = chat_bedrock_cls(
        model_id=model_name,
        region_name=region_name,
        model_kwargs=model_kwargs,
    )
    return prompt | llm.with_structured_output(JudgeVerdict)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _build_versioned_output_path(
    output_dir: Path,
    model_name: str,
    generated_at: datetime,
) -> Path:
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    model_fragment = _safe_filename_component(model_name)
    base_name = f"llm_judge_results_{timestamp}_{model_fragment}"
    candidate = output_dir / f"{base_name}.json"

    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{suffix}.json"
        suffix += 1

    return candidate


def evaluate_with_llm_judge(
    golden_set_path: Path,
    api_url: str | None = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    top_p: float | None = None,
    top_k: int | None = None,
    timeout_seconds: int = 60,
    min_questions: int = 20,
    strict_judge: bool = False,
) -> dict[str, Any]:
    golden_set = _load_golden_set(golden_set_path)
    if len(golden_set) < min_questions:
        raise ValueError(
            "Golden set invalido: esperado pelo menos "
            f"{min_questions} perguntas, encontrado {len(golden_set)}."
        )

    records = _materialize_records(golden_set, api_url, timeout_seconds)
    judge_chain = _build_judge_chain(
        model_name=model_name, temperature=temperature, top_p=top_p, top_k=top_k
    )

    judged_records: list[dict[str, Any]] = []
    financial_scores: list[int] = []
    clarity_scores: list[int] = []
    hallucination_scores: list[int] = []
    overall_scores: list[float] = []
    backend_counts = {"llm": 0, "deterministic_fallback": 0}
    fallback_only_mode = False

    for index, record in enumerate(records, start=1):
        LOGGER.info("Avaliando item %d/%d com LLM-as-judge...", index, len(records))
        backend = "llm"
        if fallback_only_mode:
            verdict = _build_fallback_verdict(
                question=record["question"],
                reference_answer=record["reference_answer"],
                candidate_answer=record["candidate_answer"],
                contexts=record["contexts"],
            )
            backend = "deterministic_fallback"
        else:
            try:
                verdict = judge_chain.invoke(
                    {
                        "question": record["question"],
                        "reference_answer": record["reference_answer"],
                        "candidate_answer": record["candidate_answer"],
                        "contexts": "\n\n".join(record["contexts"])
                        if record["contexts"]
                        else "Sem contexto adicional.",
                    }
                )
            except Exception as error:
                if strict_judge:
                    raise
                LOGGER.warning(
                    "Falha no backend LLM no item %d/%d (%s). "
                    "Ativando fallback deterministico para este e os proximos itens.",
                    index,
                    len(records),
                    error,
                )
                fallback_only_mode = True
                verdict = _build_fallback_verdict(
                    question=record["question"],
                    reference_answer=record["reference_answer"],
                    candidate_answer=record["candidate_answer"],
                    contexts=record["contexts"],
                )
                backend = "deterministic_fallback"

        financial_scores.append(verdict.precisao_financeira.score)
        clarity_scores.append(verdict.clareza.score)
        hallucination_scores.append(verdict.ausencia_alucinacoes.score)
        overall_scores.append(float(verdict.nota_final))
        backend_counts[backend] += 1

        judged_records.append(
            {
                **record,
                "judge_backend": backend,
                "judge": verdict.model_dump(),
            }
        )

    summary = {
        "precisao_financeira_media": round(mean(financial_scores), 4),
        "clareza_media": round(mean(clarity_scores), 4),
        "ausencia_alucinacoes_media": round(mean(hallucination_scores), 4),
        "nota_final_media": round(mean(overall_scores), 4),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError("Resumo invalido: valores nao finitos detectados no LLM-as-judge.")

    LOGGER.info("Resumo da avaliacao LLM-as-judge: %s", summary)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_type": "llm_judge_3_criteria",
        "criteria": [
            "precisao_financeira",
            "clareza",
            "ausencia_alucinacoes",
        ],
        "golden_set_path": str(golden_set_path),
        "sample_count": len(judged_records),
        "model": model_name,
        "judge_backend_counts": backend_counts,
        "summary": summary,
        "records": judged_records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia respostas com LLM-as-judge usando LangChain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--golden-set", required=True, type=Path, help="Caminho para o golden set JSON."
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "URL base da API FastAPI. O script chamara POST /chat "
            "quando answer nao estiver no JSON."
        ),
    )
    parser.add_argument(
        "--model", default=_resolve_judge_model(), help="Modelo Bedrock usado como juiz."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_resolve_judge_temperature(),
        help="Temperatura do LLM juiz.",
    )
    parser.add_argument(
        "--top-p", type=float, default=_resolve_judge_top_p(), help="Top-p do LLM juiz."
    )
    parser.add_argument(
        "--top-k", type=int, default=_resolve_judge_top_k(), help="Top-k do LLM juiz."
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout de chamada HTTP em segundos."
    )
    parser.add_argument(
        "--min-questions", type=int, default=20, help="Quantidade minima de exemplos no golden set."
    )
    parser.add_argument(
        "--strict-judge",
        action="store_true",
        help="Falha a execucao se o backend LLM indisponivel (sem fallback deterministico).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Arquivo JSON estavel (latest) com o detalhamento das notas.",
    )
    parser.add_argument(
        "--versioned-output-dir",
        type=Path,
        default=DEFAULT_VERSIONED_OUTPUT_DIR,
        help="Diretorio para armazenar arquivos versionados por execucao.",
    )
    parser.add_argument(
        "--skip-versioned-output",
        action="store_true",
        help="Nao gera arquivo versionado por execucao (mantem apenas --output).",
    )
    return parser.parse_args()


def _ensure_bedrock_configuration() -> None:
    if not _resolve_bedrock_region():
        raise OSError(
            "Região AWS para Bedrock ausente. Configure BEDROCK_AWS_REGION, AWS_REGION "
            "ou AWS_DEFAULT_REGION antes de executar o avaliador LLM-as-judge."
        )


def main() -> int:
    _configure_logging()
    _load_dotenv_file(PROJECT_ROOT / ".env")
    args = _parse_args()
    _ensure_bedrock_configuration()

    result = evaluate_with_llm_judge(
        golden_set_path=args.golden_set,
        api_url=args.api_url,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        timeout_seconds=args.timeout,
        min_questions=args.min_questions,
        strict_judge=args.strict_judge,
    )

    generated_at = datetime.now(UTC)
    generated_at_iso = generated_at.isoformat().replace("+00:00", "Z")
    run_config = {
        "api_url": args.api_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "timeout_seconds": args.timeout,
        "min_questions": args.min_questions,
    }
    result["generated_at_utc"] = generated_at_iso
    result["run_config"] = run_config
    result["output_path"] = str(args.output)

    _write_json_atomic(args.output, result)

    if not args.skip_versioned_output:
        versioned_path = _build_versioned_output_path(
            output_dir=args.versioned_output_dir,
            model_name=args.model,
            generated_at=generated_at,
        )
        _write_json_atomic(versioned_path, result)
        result["versioned_output_path"] = str(versioned_path)
        LOGGER.info("Resultado versionado salvo em %s", versioned_path)

        # Keep latest output and versioned artifact with identical payload.
        result["output_path"] = str(args.output)
        _write_json_atomic(args.output, result)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    LOGGER.info("Resultado salvo em %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
