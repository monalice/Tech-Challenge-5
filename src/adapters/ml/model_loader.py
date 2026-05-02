from __future__ import annotations

from typing import Any

# IMPORTANTE (Windows): tensorflow deve ser importado cedo para evitar conflito
# de DLL que causa crash (exit code -1073741819)
import tensorflow as tf


def load_trained_model(model_path: str) -> Any:
    """Carrega um modelo Keras/TensorFlow a partir do caminho especificado."""
    keras_module = getattr(tf, "keras", None)
    if keras_module is None or not hasattr(keras_module, "models"):
        raise RuntimeError("TensorFlow/Keras indisponível para carregar o modelo")
    return keras_module.models.load_model(model_path)
