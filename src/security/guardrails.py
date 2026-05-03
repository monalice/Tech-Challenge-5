import os
import re
from dataclasses import dataclass
from typing import Any


def resolve_aws_region() -> str | None:
    """Resolve a região AWS para Bedrock a partir de variáveis de ambiente."""
    return (
        os.getenv("BEDROCK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )


try:
    import boto3  # type: ignore[import-not-found]
except ImportError:
    boto3 = None  # type: ignore[assignment]

try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]
except ImportError:
    AnalyzerEngine = None  # type: ignore[assignment]
    AnonymizerEngine = None  # type: ignore[assignment]


@dataclass
class InputValidationResult:
    allowed: bool
    reason: str | None = None
    sanitized_text: str | None = None


class _BedrockGuardrailBase:
    def __init__(
        self,
        *,
        region_name: str | None = None,
        guardrail_identifier: str | None = None,
        guardrail_version: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.region_name = region_name or resolve_aws_region()
        self.guardrail_identifier = guardrail_identifier or os.getenv("BEDROCK_GUARDRAIL_ID")
        self.guardrail_version = guardrail_version or os.getenv("BEDROCK_GUARDRAIL_VERSION")
        self._client = client

    def _ensure_client(self) -> Any:
        """Instancia ou retorna o cliente boto3 bedrock-runtime, criando-o se necessário.

        Returns:
            Cliente boto3 ``bedrock-runtime`` pronto para uso.

        Raises:
            RuntimeError: Se a região AWS ou boto3 não estiverem disponíveis.
        """
        if self._client is None:
            if not self.region_name:
                raise RuntimeError(
                    "Região AWS não configurada para Amazon Bedrock Guardrails. "
                    "Use BEDROCK_AWS_REGION, AWS_REGION ou AWS_DEFAULT_REGION."
                )
            if boto3 is None:
                raise RuntimeError(
                    "boto3 não está instalado. Adicione a dependência antes de "
                    "usar Amazon Bedrock Guardrails."
                )
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def _ensure_guardrail_config(self) -> None:
        if not self.guardrail_identifier or not self.guardrail_version:
            raise RuntimeError(
                "Amazon Bedrock Guardrails não configurado. "
                "Defina BEDROCK_GUARDRAIL_ID e BEDROCK_GUARDRAIL_VERSION."
            )

    def _apply_guardrail(
        self,
        text: str,
        *,
        source: str,
        qualifiers: list[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_guardrail_config()
        client = self._ensure_client()
        content_block: dict[str, Any] = {"text": {"text": text}}
        if qualifiers:
            content_block["text"]["qualifiers"] = qualifiers
        response = client.apply_guardrail(
            guardrailIdentifier=self.guardrail_identifier,
            guardrailVersion=self.guardrail_version,
            source=source,
            content=[content_block],
            outputScope="FULL",
        )
        return dict(response)

    @staticmethod
    def _iter_assessment_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for assessment in response.get("assessments", []):
            if not isinstance(assessment, dict):
                continue

            content_policy = assessment.get("contentPolicy", {})
            for entry in content_policy.get("filters", []):
                if isinstance(entry, dict):
                    items.append(entry)

            sensitive_policy = assessment.get("sensitiveInformationPolicy", {})
            for entry in sensitive_policy.get("piiEntities", []):
                if isinstance(entry, dict):
                    items.append(entry)
            for entry in sensitive_policy.get("regexes", []):
                if isinstance(entry, dict):
                    items.append(entry)

            topic_policy = assessment.get("topicPolicy", {})
            for entry in topic_policy.get("topics", []):
                if isinstance(entry, dict):
                    items.append(entry)

            word_policy = assessment.get("wordPolicy", {})
            for key in ("customWords", "managedWordLists"):
                for entry in word_policy.get(key, []):
                    if isinstance(entry, dict):
                        items.append(entry)
        return items

    @classmethod
    def _has_blocked_intervention(cls, response: dict[str, Any]) -> bool:
        for entry in cls._iter_assessment_entries(response):
            if entry.get("detected") and entry.get("action") == "BLOCKED":
                return True
        return False

    @classmethod
    def _has_prompt_attack(cls, response: dict[str, Any]) -> bool:
        for entry in cls._iter_assessment_entries(response):
            if (
                entry.get("detected")
                and entry.get("action") == "BLOCKED"
                and entry.get("type") == "PROMPT_ATTACK"
            ):
                return True
        return False

    @staticmethod
    def _extract_output_text(response: dict[str, Any], fallback_text: str) -> str:
        output_texts: list[str] = []
        for output in response.get("outputs", []):
            if isinstance(output, dict):
                text = output.get("text")
                if isinstance(text, str) and text:
                    output_texts.append(text)
        if output_texts:
            return "\n".join(output_texts)
        return fallback_text

    @classmethod
    def _build_reason(cls, response: dict[str, Any], default_reason: str) -> str:
        reason = response.get("actionReason")
        if isinstance(reason, str) and reason.strip():
            return reason

        labels: list[str] = []
        for entry in cls._iter_assessment_entries(response):
            if not entry.get("detected"):
                continue
            label = entry.get("type") or entry.get("name") or entry.get("match")
            if isinstance(label, str) and label not in labels:
                labels.append(label)
        if labels:
            return f"{default_reason} Políticas acionadas: {', '.join(labels)}."
        return default_reason


class InputGuardrail(_BedrockGuardrailBase):
    """Valida entrada do usuário contra prompt injection e context stuffing."""

    MAX_INPUT_CHARS = 4096
    PROMPT_INJECTION_PATTERNS = (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(todas\s+as\s+)?instru[cç][oõ]es\s+anteriores",
        r"disregard\s+(all\s+)?(rules|instructions)",
        r"bypass\s+(the\s+)?(guardrails|safety|security)",
        r"reveal\s+(system|hidden)\s+prompt",
        r"act\s+as\s+(developer|system)",
        r"sudo\s+override",
        r"forget\s+(everything|all\s+previous\s+instructions)",
    )

    def __init__(
        self,
        max_input_chars: int = MAX_INPUT_CHARS,
        *,
        region_name: str | None = None,
        guardrail_identifier: str | None = None,
        guardrail_version: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            guardrail_identifier=guardrail_identifier,
            guardrail_version=guardrail_version,
            client=client,
        )
        self.max_input_chars = max_input_chars

    def _matches_prompt_injection(self, text: str) -> bool:
        lowered_text = text.lower()
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered_text, flags=re.IGNORECASE):
                return True
        return False

    def apply(self, text: str) -> InputValidationResult:
        """Valida o texto de entrada contra o Amazon Bedrock Guardrails.

        Rejeita entradas vazias, acima do limite de caracteres, com prompt
        injection ou com intervenções bloqueadas pela política configurada.

        Args:
            text: Texto de entrada fornecido pelo usuário.

        Returns:
            :class:`InputValidationResult` com ``allowed=True`` e o texto
            saneado quando a entrada é permitida, ou ``allowed=False`` com o
            motivo do bloqueio.
        """
        if not isinstance(text, str) or not text.strip():
            return InputValidationResult(
                allowed=False,
                reason="A mensagem deve ser um texto não vazio.",
            )

        if len(text) > self.max_input_chars:
            return InputValidationResult(
                allowed=False,
                reason=f"Input acima de {self.max_input_chars} caracteres (context stuffing).",
            )

        if self._matches_prompt_injection(text):
            return InputValidationResult(
                allowed=False,
                reason="Entrada bloqueada por padrão de prompt injection.",
            )

        # Se Bedrock Guardrails não estiver configurado no ambiente, aplica
        # apenas validações locais (prompt injection + tamanho) sem bloquear uso.
        if not self.guardrail_identifier or not self.guardrail_version:
            return InputValidationResult(
                allowed=True,
                sanitized_text=text,
            )

        try:
            response = self._apply_guardrail(text, source="INPUT", qualifiers=["query"])
        except Exception as exc:
            return InputValidationResult(
                allowed=False,
                reason=f"Falha ao validar a entrada no Amazon Bedrock Guardrails: {exc}",
            )

        if self._has_prompt_attack(response) or self._has_blocked_intervention(response):
            return InputValidationResult(
                allowed=False,
                reason=self._build_reason(
                    response,
                    "Entrada bloqueada pelo Amazon Bedrock Guardrails.",
                ),
            )

        return InputValidationResult(
            allowed=True,
            sanitized_text=self._extract_output_text(response, text),
        )

    def validate(self, text: str) -> InputValidationResult:
        """Alias de :meth:`apply` para compatibilidade com interfaces legadas.

        Args:
            text: Texto de entrada a ser validado.

        Returns:
            Resultado de :meth:`apply`.
        """
        return self.apply(text)


class OutputGuardrail(_BedrockGuardrailBase):
    """Aplica Amazon Bedrock Guardrails na saída do LLM para PII e conteúdo bloqueado."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        guardrail_identifier: str | None = None,
        guardrail_version: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            guardrail_identifier=guardrail_identifier,
            guardrail_version=guardrail_version,
            client=client,
        )

    @staticmethod
    def _anonymize_pii_with_regex(text: str) -> str:
        """Aplica anonimização local para PII comum quando Bedrock/Presidio falham."""
        redacted = text
        # CPF: 000.000.000-00 ou 11 dígitos contínuos
        redacted = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF_REDACTED]", redacted)
        redacted = re.sub(r"\b\d{11}\b", "[CPF_REDACTED]", redacted)
        # E-mail
        redacted = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "[EMAIL_REDACTED]",
            redacted,
        )
        # Telefone BR/intl simplificado
        redacted = re.sub(
            r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}",
            "[PHONE_REDACTED]",
            redacted,
        )
        return redacted

    @staticmethod
    def _anonymize_pii_with_presidio(text: str) -> str:
        """Tenta anonimização via Presidio; usa fallback regex se indisponível."""
        if AnalyzerEngine is None or AnonymizerEngine is None:
            return OutputGuardrail._anonymize_pii_with_regex(text)

        try:
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF", "CPF"]
            results = analyzer.analyze(text=text, language="pt", entities=entities)
            if not results:
                results = analyzer.analyze(text=text, language="en", entities=entities)
            if not results:
                return OutputGuardrail._anonymize_pii_with_regex(text)
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return str(anonymized.text)
        except Exception:
            return OutputGuardrail._anonymize_pii_with_regex(text)

    def sanitize(self, text: str) -> str:
        """Aplica guardrails na saída do LLM, redatando PII e conteúdo bloqueado.

        Args:
            text: Texto de saída gerado pelo LLM.

        Returns:
            Texto saneado pelo Amazon Bedrock Guardrails, ou mensagem de
            retenção se o conteúdo for bloqueado ou ocorrer falha na API.
        """
        if not text:
            return text

        # Sem Bedrock configurado, aplica sanitização local (Presidio/regex)
        # para manter o guardrail de saída funcional em ambientes de desenvolvimento.
        if not self.guardrail_identifier or not self.guardrail_version:
            return self._anonymize_pii_with_presidio(text)

        try:
            response = self._apply_guardrail(
                text,
                source="OUTPUT",
                qualifiers=["guard_content"],
            )
        except Exception as exc:
            # Mantém mensagem explícita de falha para observabilidade e testes,
            # anexando saída com PII anonimizada via fallback local.
            sanitized_fallback = self._anonymize_pii_with_presidio(text)
            return (
                "Saída retida por falha ao aplicar Amazon Bedrock Guardrails: "
                f"{exc}. Conteúdo fallback anonimizado: {sanitized_fallback}"
            )

        if self._has_blocked_intervention(response):
            sanitized_text = self._extract_output_text(response, "")
            if sanitized_text:
                return self._anonymize_pii_with_presidio(sanitized_text)
            return self._anonymize_pii_with_presidio(
                self._build_reason(
                    response,
                    "Conteúdo retido pelo Amazon Bedrock Guardrails.",
                )
            )

        return self._anonymize_pii_with_presidio(self._extract_output_text(response, text))
