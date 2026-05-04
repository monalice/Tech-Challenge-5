import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.agent.llm_config import publish_cloudwatch_llm_metrics
from src.delivery.api.schemas import ChatRequest, ChatResponse
from src.domain.inference import InferenceService
from src.domain.ports import LLMPort, LoadedArtifacts
from src.security.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger("stockcast")

router = APIRouter()

input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat com o Agente LLM (ReAct)",
    description=(
        "Recebe uma mensagem em linguagem natural e a processa com um Agente ReAct "
        "(LangChain + Amazon Bedrock). O agente orquestra 3 ferramentas: PrevisaoBitcoinTool "
        "(inferência LSTM), CotacaoAtualTool (cotação em tempo real) e CryptoRAGTool "
        "(contexto de mercado simulado). Requer acesso AWS com permissões para Amazon Bedrock."
    ),
)
def chat(http_request: Request, request: ChatRequest) -> dict[str, Any]:
    start_proc = time.perf_counter()
    is_error = False

    try:
        input_validation = input_guardrail.apply(request.message)
        if not input_validation.allowed:
            is_error = True
            raise HTTPException(status_code=400, detail=input_validation.reason)
        llm_input = input_validation.sanitized_text or request.message

        # Import lazy para evitar importação circular no nível de módulo
        from src.agent.react_agent import build_agent  # noqa: PLC0415

        try:
            _artifacts: LoadedArtifacts | None = getattr(http_request.app.state, "artifacts", None)
            _service: InferenceService | None = getattr(http_request.app.state, "service", None)
            _agent_llm: LLMPort | None = getattr(http_request.app.state, "agent_llm", None)
            if _artifacts is None or _service is None or _agent_llm is None:
                is_error = True
                raise HTTPException(
                    status_code=503,
                    detail="Artefatos ou LLM do agente não disponíveis.",
                )
            agent_executor = build_agent(_artifacts, _service, _agent_llm)
        except OSError as exc:
            is_error = True
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            result = agent_executor.invoke({"input": llm_input})
        except Exception as exc:
            is_error = True
            logger.error("Erro no agente ReAct: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro no agente: {exc}") from exc

        steps: list[dict[str, str]] = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append(
                {
                    "tool": getattr(action, "tool", str(action)),
                    "tool_input": str(getattr(action, "tool_input", "")),
                    "observation": output_guardrail.sanitize(str(observation)),
                }
            )

        safe_output = output_guardrail.sanitize(result.get("output", ""))
        return {"response": safe_output, "steps": steps}
    except HTTPException:
        is_error = True
        raise
    finally:
        latency_ms = (time.perf_counter() - start_proc) * 1000
        publish_cloudwatch_llm_metrics(latency_ms=latency_ms, is_error=is_error)
