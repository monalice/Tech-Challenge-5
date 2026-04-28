import re
from dataclasses import dataclass
from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


@dataclass
class InputValidationResult:
    allowed: bool
    reason: str | None = None


class InputGuardrail:
    """Valida entrada do usuário contra prompt injection e context stuffing."""

    MAX_INPUT_CHARS = 4096

    def __init__(self, max_input_chars: int = MAX_INPUT_CHARS):
        self.max_input_chars = max_input_chars
        patterns = [
            r"ignore\s+as\s+instru[çc][õo]es\s+anteriores",
            r"ignore\s+previous\s+instructions",
            r"disregard\s+all\s+previous\s+instructions",
            r"forget\s+all\s+previous\s+instructions",
            r"you\s+are\s+now\s+.*",
            r"act\s+as\s+.*",
            r"system\s+prompt",
            r"developer\s+mode",
            r"jailbreak",
            r"do\s+anything\s+now",
            r"###\s*system",
            r"\[system\]",
            r"<\|.*?\|>",
        ]
        self._injection_regex = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

    def validate(self, text: str) -> InputValidationResult:
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

        if self._injection_regex.search(text):
            return InputValidationResult(
                allowed=False,
                reason="Possível tentativa de prompt injection detectada.",
            )

        return InputValidationResult(allowed=True)


class OutputGuardrail:
    """Detecta e mascara PII na saída do LLM com Presidio."""

    def __init__(self, score_threshold: float = 0.5):
        self.score_threshold = score_threshold
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None

    def _ensure_engines(self) -> None:
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()
        if self._anonymizer is None:
            self._anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]

    @staticmethod
    def _regex_fallback_mask(text: str) -> str:
        masked = text
        masked = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL_MASKED>", masked)
        masked = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "<CPF_MASKED>", masked)
        masked = re.sub(r"\b\d{11}\b", "<CPF_MASKED>", masked)
        masked = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "<CNPJ_MASKED>", masked)
        masked = re.sub(r"\b\d{14}\b", "<CNPJ_MASKED>", masked)
        masked = re.sub(r"\b\d{14,16}\b", "<CARD_MASKED>", masked)
        masked = re.sub(r"\b\+?\d{2,3}?\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b", "<PHONE_MASKED>", masked)
        return masked

    def sanitize(self, text: str) -> str:
        if not text:
            return text

        analyzer_results: list[Any] = []
        try:
            self._ensure_engines()
            if self._analyzer is None:
                return self._regex_fallback_mask(text)
            analyzer: Any = self._analyzer
            analyzer_results = analyzer.analyze(
                text=text,
                language="en",
                score_threshold=self.score_threshold,
            )
        except Exception:
            return self._regex_fallback_mask(text)

        if not analyzer_results:
            return self._regex_fallback_mask(text)

        if self._anonymizer is None:
            return self._regex_fallback_mask(text)
        anonymizer: Any = self._anonymizer
        anonymized_result = anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
        )
        return self._regex_fallback_mask(anonymized_result.text)
