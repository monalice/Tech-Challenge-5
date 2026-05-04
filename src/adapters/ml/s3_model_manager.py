"""Gerenciador de artefatos de modelo em S3."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class S3ModelManager:
    """Gerencia salvamento e carregamento de modelos/scalers em S3."""

    def __init__(
        self,
        bucket_name: str | None = None,
        prefix: str = "models",
    ):
        """Inicializa o gerenciador.

        Args:
            bucket_name: Nome do bucket S3. Se None, usa variável de ambiente
                        S3_MODELS_BUCKET. Se vazio, desativa S3 (fallback local).
            prefix: Prefixo de caminho dentro do bucket (padrão: 'models').
        """
        self.bucket_name = bucket_name or os.getenv("S3_MODELS_BUCKET", "").strip()
        self.prefix = prefix.strip("/")
        self.s3_enabled = bool(self.bucket_name)

        if self.s3_enabled:
            try:
                import boto3  # noqa: PLC0415

                self.s3_client = boto3.client("s3")
                logger.info(
                    "S3ModelManager inicializado com bucket=%s, prefix=%s",
                    self.bucket_name,
                    self.prefix,
                )
            except ImportError as e:
                logger.error("boto3 não disponível; S3 desativado: %s", e)
                self.s3_enabled = False
                self.s3_client = None
        else:
            self.s3_client = None
            logger.info("S3 desativado; usando fallback local (models/)")

    def _s3_key(self, file_name: str) -> str:
        """Constrói chave S3 a partir do nome do arquivo."""
        return f"{self.prefix}/{file_name}" if self.prefix else file_name

    def save_model(self, model: Any, file_name: str, use_s3: bool | None = None) -> str:
        """Salva um modelo Keras em S3 ou local.

        Args:
            model: Modelo Keras.
            file_name: Nome do arquivo (ex: 'lstm_btc_hourly.keras').
            use_s3: Force S3 (True) ou local (False). Se None, usa auto-detect.

        Returns:
            Caminho do artefato salvo (s3://... ou local).
        """
        if use_s3 is None:
            use_s3 = self.s3_enabled

        if use_s3 and self.s3_enabled:
            return self._save_model_s3(model, file_name)
        return self._save_model_local(model, file_name)

    def _save_model_s3(self, model: Any, file_name: str) -> str:
        """Salva modelo no S3."""
        try:
            _, ext = os.path.splitext(file_name)
            save_suffix = ext if ext in {".keras", ".h5"} else ".keras"

            with tempfile.NamedTemporaryFile(suffix=save_suffix, delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                model.save(tmp_path)
                with open(tmp_path, "rb") as model_file:
                    model_bytes = model_file.read()
            finally:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

            key = self._s3_key(file_name)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=model_bytes,
            )
            s3_path = f"s3://{self.bucket_name}/{key}"
            logger.info("Modelo salvo em S3: %s", s3_path)
            return s3_path
        except Exception as e:
            logger.error("Erro ao salvar modelo em S3: %s", e)
            raise

    def _save_model_local(self, model: Any, file_name: str) -> str:
        """Salva modelo localmente (fallback)."""
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{file_name}"
        model.save(local_path)
        logger.info("Modelo salvo localmente (S3 indisponível): %s", local_path)
        return local_path

    def save_joblib(self, obj: Any, file_name: str, use_s3: bool | None = None) -> str:
        """Salva objeto Python (scaler, etc.) via joblib em S3 ou local.

        Args:
            obj: Objeto a serializar (ex: scaler sklearn).
            file_name: Nome do arquivo (ex: 'scaler_btc.gz').
            use_s3: Force S3 (True) ou local (False). Se None, usa auto-detect.

        Returns:
            Caminho do artefato salvo.
        """
        if use_s3 is None:
            use_s3 = self.s3_enabled

        if use_s3 and self.s3_enabled:
            return self._save_joblib_s3(obj, file_name)
        return self._save_joblib_local(obj, file_name)

    def _save_joblib_s3(self, obj: Any, file_name: str) -> str:
        """Salva joblib no S3."""
        try:
            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            buffer.seek(0)

            key = self._s3_key(file_name)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
            )
            s3_path = f"s3://{self.bucket_name}/{key}"
            logger.info("Joblib salvo em S3: %s", s3_path)
            return s3_path
        except Exception as e:
            logger.error("Erro ao salvar joblib em S3: %s", e)
            raise

    def _save_joblib_local(self, obj: Any, file_name: str) -> str:
        """Salva joblib localmente (fallback)."""
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{file_name}"
        joblib.dump(obj, local_path)
        logger.info("Joblib salvo localmente (S3 indisponível): %s", local_path)
        return local_path

    def load_model(self, file_name: str, use_s3: bool | None = None) -> Any:
        """Carrega um modelo Keras de S3 ou local.

        Args:
            file_name: Nome do arquivo (ex: 'lstm_btc_hourly.keras').
            use_s3: Force S3 (True) ou local (False). Se None, usa auto-detect.

        Returns:
            Modelo Keras carregado.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if use_s3 is None:
            use_s3 = self.s3_enabled

        if use_s3 and self.s3_enabled:
            return self._load_model_s3(file_name)
        return self._load_model_local(file_name)

    def _load_model_s3(self, file_name: str) -> Any:
        """Carrega modelo do S3."""
        try:
            import tensorflow as tf  # noqa: PLC0415

            key = self._s3_key(file_name)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            model_bytes = response["Body"].read()

            keras_module = getattr(tf, "keras", None)
            if keras_module is None or not hasattr(keras_module, "models"):
                raise RuntimeError("TensorFlow/Keras indisponível")

            _, ext = os.path.splitext(file_name)
            load_suffix = ext if ext in {".keras", ".h5"} else ".keras"
            with tempfile.NamedTemporaryFile(suffix=load_suffix, delete=False) as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name

            try:
                model = keras_module.models.load_model(tmp_path)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

            logger.info("Modelo carregado de S3: s3://%s/%s", self.bucket_name, key)
            return model
        except self.s3_client.exceptions.NoSuchKey as e:
            logger.error("Modelo não encontrado em S3: s3://%s/%s", self.bucket_name, key)
            raise FileNotFoundError(f"Modelo não encontrado em S3: {key}") from e
        except Exception as e:
            logger.error("Erro ao carregar modelo de S3: %s", e)
            raise

    def _load_model_local(self, file_name: str) -> Any:
        """Carrega modelo localmente (fallback)."""
        import tensorflow as tf  # noqa: PLC0415

        local_path = f"models/{file_name}"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Modelo não encontrado localmente: {local_path}")

        keras_module = getattr(tf, "keras", None)
        if keras_module is None or not hasattr(keras_module, "models"):
            raise RuntimeError("TensorFlow/Keras indisponível")

        model = keras_module.models.load_model(local_path)
        logger.info("Modelo carregado localmente: %s", local_path)
        return model

    def load_joblib(self, file_name: str, use_s3: bool | None = None) -> Any:
        """Carrega objeto Python (scaler, etc.) via joblib de S3 ou local.

        Args:
            file_name: Nome do arquivo (ex: 'scaler_btc.gz').
            use_s3: Force S3 (True) ou local (False). Se None, usa auto-detect.

        Returns:
            Objeto desserializado.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if use_s3 is None:
            use_s3 = self.s3_enabled

        if use_s3 and self.s3_enabled:
            return self._load_joblib_s3(file_name)
        return self._load_joblib_local(file_name)

    def _load_joblib_s3(self, file_name: str) -> Any:
        """Carrega joblib do S3."""
        try:
            key = self._s3_key(file_name)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)

            obj = joblib.load(buffer)
            logger.info("Joblib carregado de S3: s3://%s/%s", self.bucket_name, key)
            return obj
        except self.s3_client.exceptions.NoSuchKey as e:
            logger.error("Joblib não encontrado em S3: s3://%s/%s", self.bucket_name, key)
            raise FileNotFoundError(f"Joblib não encontrado em S3: {key}") from e
        except Exception as e:
            logger.error("Erro ao carregar joblib de S3: %s", e)
            raise

    def _load_joblib_local(self, file_name: str) -> Any:
        """Carrega joblib localmente (fallback)."""
        local_path = f"models/{file_name}"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Joblib não encontrado localmente: {local_path}")

        obj = joblib.load(local_path)
        logger.info("Joblib carregado localmente: %s", local_path)
        return obj
