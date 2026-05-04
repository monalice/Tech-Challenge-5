FROM python:3.11-slim

ARG VCS_REF=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    TF_CPP_MIN_LOG_LEVEL="2" \
    GIT_PYTHON_REFRESH="quiet" \
    TRAINING_GIT_SHA="${VCS_REF}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY training ./training
COPY scripts ./scripts
COPY models ./models
COPY dvc.lock ./dvc.lock
COPY evaluation/fairness_report.json ./evaluation/fairness_report.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
