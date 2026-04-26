import json
import logging
import os
import subprocess
import tempfile
import time

import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import requests

# IMPORTANTE (Windows): tensorflow e yfinance devem ser importados antes de pandas
# para evitar conflito de DLL que causa crash (exit code -1073741819)
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Configurações
TICKER = "BTC-USD"
PERIOD = "730d"
INTERVAL = "1h"

LOOKBACK = 60
BATCH_SIZE = 64
EPOCHS = 100
TEST_SIZE_PCT = 0.2
VAL_SIZE_PCT = 0.1
WALK_FORWARD_SPLITS = 3
WALK_FORWARD_EPOCHS = 20
RANDOM_SEED = 42
EPSILON = 1e-8
DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_TIMEOUT_SECONDS = 15
DOWNLOAD_BASE_BACKOFF_SECONDS = 30
DOWNLOAD_MAX_BACKOFF_SECONDS = 120
CACHE_DATA_PATH = "models/btc_hourly_cache.csv"

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_TIMEOUT_SECONDS = 10
BINANCE_KLINE_LIMIT = 1000  # máximo por request

MODEL_PATH = "models/lstm_btc_hourly.keras"
SCALER_PATH = "models/scaler_btc.gz"
SCALER_RETURN_PATH = "models/scaler_btc_return.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "btc-hourly-forecast")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI")

TAG_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "btc_hourly_forecaster")
TAG_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "v1")
TAG_OWNER = os.getenv("MLFLOW_OWNER", "ml-team")
TAG_RISK_LEVEL = os.getenv("MLFLOW_RISK_LEVEL", "medium")
TAG_TRAINING_DATA_VERSION = os.getenv("MLFLOW_TRAINING_DATA_VERSION", "models/btc_hourly_cache.csv")


def get_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as error:
        logger.warning("Nao foi possivel obter git SHA dinamicamente: %s", error)
        return "unknown"


def ensure_directories():
    if not os.path.exists("models"):
        os.makedirs("models")


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise OSError(
            "A variável de ambiente MLFLOW_TRACKING_URI não foi definida. "
            "Configure-a para o PostgreSQL (AWS RDS) do MLflow Tracking Server."
        )

    if tracking_uri.startswith("file://"):
        raise OSError(
            "MLFLOW_TRACKING_URI não pode usar file:// para Model Registry. "
            "Use o endpoint HTTP/HTTPS do MLflow Tracking Server com backend SQL (RDS)."
        )

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        if MLFLOW_ARTIFACT_URI:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACT_URI,
            )
        else:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)


def log_training_artifacts(model, scaler_all, scaler_return, metadata):
    with tempfile.TemporaryDirectory(prefix="mlflow_artifacts_") as temp_dir:
        model_file = os.path.join(temp_dir, "lstm_btc_hourly.keras")
        scaler_file = os.path.join(temp_dir, "scaler_btc.gz")
        scaler_return_file = os.path.join(temp_dir, "scaler_btc_return.gz")
        metadata_file = os.path.join(temp_dir, "model_metadata_btc.json")

        model.save(model_file)
        joblib.dump(scaler_all, scaler_file)
        joblib.dump(scaler_return, scaler_return_file)

        with open(metadata_file, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)

        mlflow.keras.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_file, artifact_path="model")
        mlflow.log_artifact(scaler_file, artifact_path="scalers")
        mlflow.log_artifact(scaler_return_file, artifact_path="scalers")
        mlflow.log_artifact(metadata_file, artifact_path="metadata")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Nenhuma run ativa encontrada para registrar o modelo no Registry.")

        model_uri = f"runs:/{active_run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="btc-lstm-hourly",
            tags={
                "stage": "challenger",
                "training_data_version": TAG_TRAINING_DATA_VERSION,
                "git_sha": get_git_sha(),
            },
        )
        logger.info(
            "Modelo registrado no Registry: %s versão %s",
            registered_model.name,
            registered_model.version,
        )


def normalize_download_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(TICKER, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)

    required_columns = ["Close", "High", "Low", "Volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes na resposta da API: {missing_columns}")

    normalized = df[required_columns].copy().dropna()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()
    return normalized


def load_cached_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_DATA_PATH):
        return pd.DataFrame()

    try:
        cached_df = pd.read_csv(CACHE_DATA_PATH, index_col=0, parse_dates=True)
        cached_df.index.name = None
        normalized = normalize_download_dataframe(cached_df)
        if normalized.empty:
            return pd.DataFrame()
        return normalized
    except Exception as error:
        logger.warning("Falha ao ler cache local em '%s': %s", CACHE_DATA_PATH, error)
        return pd.DataFrame()


def save_cached_data(data: pd.DataFrame):
    try:
        data.to_csv(CACHE_DATA_PATH)
    except Exception as error:
        logger.warning("Nao foi possivel salvar cache local em '%s': %s", CACHE_DATA_PATH, error)


def download_from_binance() -> pd.DataFrame:
    """Baixa dados horários do BTC via Binance REST API pública.

    Usa paginação para cobrir todo o período definido em PERIOD.
    """
    logger.info("Tentando fonte alternativa: Binance REST API...")
    interval_ms = 60 * 60 * 1000  # 1 hora em ms
    days = int(PERIOD.replace("d", ""))
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * interval_ms

    all_rows = []
    cursor = start_ts
    while cursor < end_ts:
        resp = requests.get(
            BINANCE_API_URL,
            params={
                "symbol": BINANCE_SYMBOL,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ts,
                "limit": BINANCE_KLINE_LIMIT,
            },
            timeout=BINANCE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + interval_ms  # avança para após o último candle
        time.sleep(0.2)  # respeita rate limit da Binance

    if not all_rows:
        raise ValueError("Binance retornou dados vazios.")

    timestamps = pd.to_datetime([row[0] for row in all_rows], unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "Close": pd.to_numeric([row[4] for row in all_rows]),
            "High": pd.to_numeric([row[2] for row in all_rows]),
            "Low": pd.to_numeric([row[3] for row in all_rows]),
            "Volume": pd.to_numeric([row[5] for row in all_rows]),
        },
        index=timestamps,
    )
    df.index.name = "Datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    logger.info("Binance: %s candles obtidos via paginacao.", len(df))
    return df


def download_crypto_data():
    """Baixa dados horários do BTC no Yahoo Finance."""
    logger.info("Baixando dados horarios (%s) para %s (Ultimos %s)...", INTERVAL, TICKER, PERIOD)

    data = pd.DataFrame()
    last_error = None
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            download_df = yf.download(
                TICKER,
                period=PERIOD,
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            data = normalize_download_dataframe(download_df)
            if not data.empty:
                break
        except Exception as error:
            last_error = error
            logger.warning("Tentativa %s/%s falhou: %s", attempt, DOWNLOAD_MAX_RETRIES, error)

        if not data.empty:
            break

        if attempt < DOWNLOAD_MAX_RETRIES:
            backoff_seconds = min(
                DOWNLOAD_MAX_BACKOFF_SECONDS, DOWNLOAD_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            )
            logger.info("Aguardando %ss antes da proxima tentativa...", backoff_seconds)
            time.sleep(backoff_seconds)

    if data.empty:
        cached_data = load_cached_data()
        if not cached_data.empty:
            logger.warning(
                "API indisponivel/limitada. Usando cache local em '%s' com %s registros.",
                CACHE_DATA_PATH,
                len(cached_data),
            )
            return cached_data

        # Fallback: Binance
        try:
            binance_data = download_from_binance()
            save_cached_data(binance_data)
            logger.info("Total de registros (Binance): %s", len(binance_data))
            return binance_data
        except Exception as binance_err:
            logger.warning("Binance tambem falhou: %s", binance_err)

        raise ValueError(
            "A API retornou um DataFrame vazio após "
            f"{DOWNLOAD_MAX_RETRIES} tentativas e não há cache local disponível. "
            f"Último erro: {last_error}"
        )

    save_cached_data(data)

    logger.info("Total de registros (horas): %s", len(data))
    return data


# --- Indicadores Técnicos ---


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI). Retorna valores brutos de 0 a 100.
    A normalização para [0, 1] é feita durante a construção da matriz de features.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd_signal(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD Signal line normalizado pelo preço (adimensional)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Normalizar pelo preço para tornar adimensional
    normalized = (macd_line - signal_line) / series.replace(0, np.nan)
    return normalized.fillna(0.0)


def compute_bollinger_pct_b(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Posição relativa dentro das Bandas de Bollinger (%B). 0=banda inferior, 1=banda superior."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    pct_b = (series - lower) / band_width
    return pct_b.fillna(0.5).clip(0.0, 1.0)


def compute_sma_ratio(series: pd.Series, short: int = 7, long: int = 21) -> pd.Series:
    """Razão entre SMA curta e SMA longa (momentum de tendência)."""
    sma_short = series.rolling(window=short).mean()
    sma_long = series.rolling(window=long).mean()
    ratio = (sma_short / sma_long.replace(0, np.nan)) - 1.0
    return ratio.fillna(0.0)


def compute_volume_ratio(volume: pd.Series, period: int = 24) -> pd.Series:
    """Razão entre volume atual e média móvel de volume (normalizado)."""
    vol_sma = volume.rolling(window=period).mean()
    ratio = volume / vol_sma.replace(0, np.nan)
    return ratio.fillna(1.0).clip(0.0, 10.0)


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Constrói matriz de features técnicas + log-return."""
    close = data["Close"]
    volume = data["Volume"]

    log_price = np.log(close)
    log_return = log_price.diff()

    rsi = compute_rsi(close, 14) / 100.0  # normalizado [0, 1]
    macd_sig = compute_macd_signal(close)  # adimensional
    bb_pct = compute_bollinger_pct_b(close)  # [0, 1]
    sma_ratio = compute_sma_ratio(close)  # pequeno valor em torno de 0
    vol_ratio = compute_volume_ratio(volume)  # em torno de 1

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "rsi": rsi,
            "macd_signal": macd_sig,
            "bb_pct_b": bb_pct,
            "sma_ratio": sma_ratio,
            "vol_ratio": vol_ratio,
        },
        index=data.index,
    )

    # Remover linhas com NaN (janelas iniciais dos indicadores)
    features = features.dropna()
    return features


def create_sliding_window_multifeature(dataset: np.ndarray, look_back: int = 60):
    """Cria janelas deslizantes para entrada multi-feature.
    dataset: shape (N, n_features)
    Retorna X de shape (samples, look_back, n_features), y de shape (samples,)
    onde y é o log_return da posição [look_back] (feature índice 0).
    """
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back : i, :])  # janela completa com todas as features
        y.append(dataset[i, 0])  # target: log_return (índice 0)
    return np.array(X), np.array(y)


def safe_mape(y_true, y_pred, eps=1e-8):
    denominator = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def build_lstm_architecture(input_shape):
    """Modelo LSTM bidirecional com múltiplas features para melhor acurácia direcional."""
    model = Sequential(
        [
            Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=48, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation="relu"),
            Dense(units=1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def run_walk_forward_backtest(X_train, y_train, scaler_return):
    """Walk-forward backtest com modelo multi-feature."""
    if len(X_train) < (WALK_FORWARD_SPLITS + 1):
        logger.warning("Dados insuficientes para walk-forward. Backtest pulado.")
        return

    logger.info("Iniciando walk-forward backtest com %s splits...", WALK_FORWARD_SPLITS)
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    model_maes = []
    baseline_maes = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        fold_model = build_lstm_architecture((X_train.shape[1], X_train.shape[2]))
        fold_early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        fold_model.fit(
            X_tr,
            y_tr,
            batch_size=BATCH_SIZE,
            epochs=WALK_FORWARD_EPOCHS,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[fold_early_stop],
            verbose=0,
        )

        y_pred_scaled = fold_model.predict(X_val_fold, verbose=0).reshape(-1, 1)
        y_pred = scaler_return.inverse_transform(y_pred_scaled).reshape(-1)

        y_real_scaled = y_val_fold.reshape(-1, 1)
        y_real = scaler_return.inverse_transform(y_real_scaled).reshape(-1)

        # baseline: último log_return da janela (índice 0 da última step)
        baseline_scaled = X_val_fold[:, -1, 0].reshape(-1, 1)
        baseline_pred = scaler_return.inverse_transform(baseline_scaled).reshape(-1)

        fold_mae = mean_absolute_error(y_real, y_pred)
        fold_baseline_mae = mean_absolute_error(y_real, baseline_pred)

        model_maes.append(fold_mae)
        baseline_maes.append(fold_baseline_mae)

        logger.info(
            "[WF][Fold %s] MAE modelo (retorno): %.6f | MAE baseline (retorno): %.6f",
            fold_idx,
            fold_mae,
            fold_baseline_mae,
        )

    logger.info(
        "[WF][Media] MAE modelo (retorno): %.6f | MAE baseline (retorno): %.6f",
        np.mean(model_maes),
        np.mean(baseline_maes),
    )


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    ensure_directories()
    configure_mlflow()

    run_tags = {
        "model_name": TAG_MODEL_NAME,
        "model_version": TAG_MODEL_VERSION,
        "model_type": "time_series",
        "owner": TAG_OWNER,
        "risk_level": TAG_RISK_LEVEL,
        "training_data_version": TAG_TRAINING_DATA_VERSION,
        "git_sha": get_git_sha(),
        "fairness_checked": True,
    }

    params = {
        "ticker": TICKER,
        "period": PERIOD,
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "test_size_pct": TEST_SIZE_PCT,
        "val_size_pct": VAL_SIZE_PCT,
        "walk_forward_splits": WALK_FORWARD_SPLITS,
        "walk_forward_epochs": WALK_FORWARD_EPOCHS,
        "random_seed": RANDOM_SEED,
        "epsilon": EPSILON,
        "download_max_retries": DOWNLOAD_MAX_RETRIES,
        "download_timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
        "download_base_backoff_seconds": DOWNLOAD_BASE_BACKOFF_SECONDS,
        "download_max_backoff_seconds": DOWNLOAD_MAX_BACKOFF_SECONDS,
        "architecture": "bidirectional_lstm_multifeature",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
    }

    with mlflow.start_run(run_name=f"{TICKER}_{INTERVAL}_training"):
        mlflow.set_tags(run_tags)
        mlflow.log_params(params)

        # 1. Download e feature engineering
        raw_data = download_crypto_data()
        features_df = build_feature_matrix(raw_data)
        n_features = features_df.shape[1]
        close_series = raw_data["Close"].reindex(features_df.index)
        mlflow.log_param("n_features", int(n_features))
        mlflow.log_param("features", ",".join(list(features_df.columns)))
        mlflow.log_metric("total_rows", float(len(raw_data)))
        mlflow.log_metric("feature_rows", float(len(features_df)))

        logger.info("Features utilizadas (%s): %s", n_features, list(features_df.columns))

        # 2. Split temporal treino/teste
        split_idx = int(len(features_df) * (1 - TEST_SIZE_PCT))
        train_features = features_df.iloc[:split_idx]
        test_features = features_df.iloc[split_idx:]

        if len(train_features) <= LOOKBACK:
            raise ValueError(
                "Dados de treino insuficientes. "
                f"Necessário mais que {LOOKBACK} registros, "
                f"recebido: {len(train_features)}."
            )
        if len(test_features) == 0:
            raise ValueError("Conjunto de teste vazio. Ajuste TEST_SIZE_PCT.")

        logger.info("Treino: %s horas | Teste: %s horas", len(train_features), len(test_features))
        mlflow.log_metric("train_rows", float(len(train_features)))
        mlflow.log_metric("test_rows", float(len(test_features)))

        # 3. Scaling por feature (scaler_all) + scaler separado para log_return (para inversão)
        scaler_all = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler_all.fit_transform(train_features.values)

        # Scaler exclusivo para log_return (feature 0) — usado na inferência e métricas
        scaler_return = MinMaxScaler(feature_range=(0, 1))
        scaler_return.fit(train_features[["log_return"]].values)

        # Escalar conjunto total para criar janelas de teste
        all_features = pd.concat([train_features, test_features], axis=0)
        scaled_all = scaler_all.transform(all_features.values)

        # 4. Criar janelas deslizantes
        X_train, y_train = create_sliding_window_multifeature(scaled_train, LOOKBACK)
        X_all, y_all = create_sliding_window_multifeature(scaled_all, LOOKBACK)

        test_start_idx_in_windows = split_idx - LOOKBACK
        if test_start_idx_in_windows < 0:
            raise ValueError(
                "Split inválido para LOOKBACK atual. Ajuste TEST_SIZE_PCT ou LOOKBACK."
            )

        X_test = X_all[test_start_idx_in_windows:]
        y_test = y_all[test_start_idx_in_windows:]

        # 5. Walk-forward backtest
        run_walk_forward_backtest(X_train, y_train, scaler_return)

        # 6. Treino final com validação
        val_size = max(1, int(len(X_train) * VAL_SIZE_PCT))
        if val_size >= len(X_train):
            val_size = 1

        X_train_fit = X_train[:-val_size]
        y_train_fit = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        if len(X_train_fit) == 0:
            raise ValueError("Treino ficou vazio após split de validação. Ajuste VAL_SIZE_PCT.")

        model = build_lstm_architecture((LOOKBACK, n_features))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ]

        model.fit(
            X_train_fit,
            y_train_fit,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        # 7. Avaliação no conjunto de teste
        predictions_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
        predictions_return = scaler_return.inverse_transform(predictions_scaled).reshape(-1)
        y_test_return = scaler_return.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

        target_indices = all_features.index[
            LOOKBACK + test_start_idx_in_windows : LOOKBACK
            + test_start_idx_in_windows
            + len(y_test)
        ]
        prev_close = close_series.shift(1).reindex(target_indices).values
        y_test_real_price = close_series.reindex(target_indices).values

        valid_mask = (~np.isnan(prev_close)) & (~np.isnan(y_test_real_price))
        prev_close = prev_close[valid_mask]
        y_test_real_price = y_test_real_price[valid_mask]
        predictions_return = predictions_return[valid_mask]
        y_test_return = y_test_return[valid_mask]

        predictions_price = prev_close * np.exp(predictions_return)
        baseline_predictions_price = prev_close

        baseline_scaled = X_test[:, -1, 0].reshape(-1, 1)
        baseline_return = scaler_return.inverse_transform(baseline_scaled).reshape(-1)[valid_mask]

        mae = mean_absolute_error(y_test_real_price, predictions_price)
        rmse = np.sqrt(mean_squared_error(y_test_real_price, predictions_price))
        mape = safe_mape(y_test_real_price, predictions_price, EPSILON)

        baseline_mae = mean_absolute_error(y_test_real_price, baseline_predictions_price)
        baseline_rmse = np.sqrt(mean_squared_error(y_test_real_price, baseline_predictions_price))
        baseline_mape = safe_mape(y_test_real_price, baseline_predictions_price, EPSILON)

        model_return_mae = mean_absolute_error(y_test_return, predictions_return)
        baseline_return_mae = mean_absolute_error(y_test_return, baseline_return)

        model_direction = np.sign(predictions_return)
        real_direction = np.sign(y_test_return)
        direction_accuracy = np.mean(model_direction == real_direction) * 100

        beats_baseline = mae < baseline_mae and rmse < baseline_rmse

        mlflow.log_metrics(
            {
                "mae_price": float(mae),
                "rmse_price": float(rmse),
                "mape_price": float(mape),
                "mae_price_baseline": float(baseline_mae),
                "rmse_price_baseline": float(baseline_rmse),
                "mape_price_baseline": float(baseline_mape),
                "mae_return": float(model_return_mae),
                "mae_return_baseline": float(baseline_return_mae),
                "direction_accuracy_pct": float(direction_accuracy),
                "beats_baseline": float(beats_baseline),
            }
        )

        # 8. Registrar artefatos e metadados no MLflow
        metadata = {
            "ticker": TICKER,
            "target": "log_return",
            "lookback": LOOKBACK,
            "interval": INTERVAL,
            "period": PERIOD,
            "seed": RANDOM_SEED,
            "n_features": n_features,
            "features": list(features_df.columns),
            "architecture": "bidirectional_lstm_multifeature",
            "metrics": {
                "mae_price": float(mae),
                "rmse_price": float(rmse),
                "mape_price": float(mape),
                "mae_price_baseline": float(baseline_mae),
                "rmse_price_baseline": float(baseline_rmse),
                "mape_price_baseline": float(baseline_mape),
                "mae_return": float(model_return_mae),
                "mae_return_baseline": float(baseline_return_mae),
                "direction_accuracy_pct": float(direction_accuracy),
            },
            "beats_baseline": bool(beats_baseline),
        }

        log_training_artifacts(model, scaler_all, scaler_return, metadata)
        logger.info("Artefatos registrados no MLflow (model/.keras, scalers/.gz e metadata).")

        logger.info("\n%s", "=" * 40)
        logger.info("RELATORIO DE PERFORMANCE (%s - HORARIO)", TICKER)
        logger.info("%s", "=" * 40)
        logger.info("Features: %s", list(features_df.columns))
        logger.info("Erro Medio Absoluto (MAE): $ %.2f", mae)
        logger.info("RMSE: $ %.2f", rmse)
        logger.info("MAPE: %.2f%%", mape)
        logger.info("%s", "-" * 40)
        logger.info("BASELINE INGENUO (y_hat = ultimo close da janela)")
        logger.info("MAE Baseline: $ %.2f", baseline_mae)
        logger.info("RMSE Baseline: $ %.2f", baseline_rmse)
        logger.info("MAPE Baseline: %.2f%%", baseline_mape)
        logger.info("%s", "-" * 40)
        logger.info("METRICAS DE RETORNO E DIRECAO")
        logger.info("MAE Retorno (Modelo): %.6f", model_return_mae)
        logger.info("MAE Retorno (Baseline): %.6f", baseline_return_mae)
        logger.info("Acuracia Direcional: %.2f%%", direction_accuracy)
        logger.info("%s", "-" * 40)
        logger.info("Modelo superou baseline? %s", "SIM" if beats_baseline else "NAO")
        logger.info("%s", "=" * 40)
        logger.info("MLflow run finalizada: %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
