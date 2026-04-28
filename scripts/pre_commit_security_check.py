from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_SENSITIVE_FILENAMES = {
    ".env.example",
}

FORBIDDEN_FILE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|/)\.env$"), "Arquivo .env nao pode ser commitado"),
    (
        re.compile(r"(^|/)\.env\.[^/]+$"),
        "Arquivos .env.* nao podem ser commitados (exceto .env.example)",
    ),
    (
        re.compile(r"(^|/)mlruns/"),
        "Diretorio mlruns nao pode ser commitado (artefatos temporarios)",
    ),
    (re.compile(r"\.log$"), "Arquivos .log nao podem ser commitados"),
    (
        re.compile(r"(^|/)train_(out|err)\.txt$"),
        "Arquivos de log de treinamento nao podem ser commitados",
    ),
    (
        re.compile(r"(^|/)models/.*\.(keras|h5|hdf5|pkl|joblib|pt|bin|safetensors|gz)$"),
        "Artefatos binarios de modelo em models/ nao podem ser commitados",
    ),
]

FORBIDDEN_CONTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        "Possivel GOOGLE_API_KEY detectada",
    ),
    (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "Possivel AWS Access Key ID detectada",
    ),
    (
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "Possivel token secreto detectado",
    ),
    (
        re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA|PRIVATE) KEY-----"),
        "Bloco de chave privada detectado",
    ),
]


def _to_repo_style_path(file_path: str) -> str:
    return file_path.replace("\\", "/")


def _is_binary(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    return b"\x00" in sample


def _should_skip_filename_checks(path_str: str) -> bool:
    name = Path(path_str).name
    return name in ALLOWED_SENSITIVE_FILENAMES


def _should_skip_content_checks(path_str: str) -> bool:
    # .env.example pode conter placeholders e nunca deve carregar segredo real.
    return Path(path_str).name == ".env.example"


def main(argv: list[str]) -> int:
    issues: list[str] = []

    for raw_path in argv[1:]:
        normalized = _to_repo_style_path(raw_path)

        if not _should_skip_filename_checks(normalized):
            for pattern, reason in FORBIDDEN_FILE_PATTERNS:
                if pattern.search(normalized):
                    issues.append(f"{normalized}: {reason}")

        path_obj = Path(raw_path)
        if not path_obj.exists() or _should_skip_content_checks(normalized):
            continue

        if _is_binary(path_obj):
            continue

        try:
            content = path_obj.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for pattern, reason in FORBIDDEN_CONTENT_PATTERNS:
            if pattern.search(content):
                issues.append(f"{normalized}: {reason}")

    if not issues:
        return 0

    print("[SECURITY CHECKLIST] Commit bloqueado por risco de segredo/artefato sensivel:")
    for issue in issues:
        print(f"- {issue}")

    print("\nAcoes recomendadas:")
    print("1) Remova o segredo do arquivo e rotacione a credencial comprometida.")
    print("2) Nao comite .env, logs operacionais ou artefatos binarios de modelo.")
    print("3) Use .env.example apenas com placeholders e sem valores reais.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
