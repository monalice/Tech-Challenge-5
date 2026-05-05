from __future__ import annotations

from typing import Any

# IMPORTANTE (Windows): tensorflow deve ser importado cedo para evitar conflito
# de DLL que causa crash (exit code -1073741819)
import tensorflow as tf

from src.adapters.ml.s3_model_manager import S3ModelManager


def load_trained_model(model_path: str, s3_bucket: str | None = None) -> Any:
    """Carrega um modelo Keras/TensorFlow a partir do caminho local ou S3.

    Args:
        model_path: Caminho local (ex: 'models/lstm_btc_hourly.keras')
                   ou S3 (ex: 's3://bucket/models/lstm_btc_hourly.keras').
        s3_bucket: Bucket S3 opcional. Se não fornecido, tenta usar variável de env.

    Returns:
        Modelo Keras carregado.

    Raises:
        FileNotFoundError: Se o modelo não existir.
        RuntimeError: Se TensorFlow/Keras não estiver disponível.
    """
    keras_module = getattr(tf, "keras", None)
    if keras_module is None or not hasattr(keras_module, "models"):
        raise RuntimeError("TensorFlow/Keras indisponível para carregar o modelo")

    # Se o caminho começa com 's3://', usa S3ModelManager
    if model_path.startswith("s3://"):
        manager = S3ModelManager(bucket_name=s3_bucket)
        # Extrai nome do arquivo do caminho S3
        file_name = model_path.split("/")[-1]
        return manager.load_model(file_name, use_s3=True)

    # Fallback local
    return keras_module.models.load_model(model_path)
