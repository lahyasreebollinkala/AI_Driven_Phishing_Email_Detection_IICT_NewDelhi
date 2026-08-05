# =============================================================================
# AI-Driven Phishing Email Detection Using NLP — Dockerfile
#
# Builds a single runtime image that contains the Streamlit UI, the FastAPI
# REST API, the pre-trained model artifacts, and the cleaned dataset.
#
# The pre-trained artifacts (models/, reports/, images/) are copied into the
# image so the container is immediately usable without re-training.
# =============================================================================
FROM python:3.12-slim

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DISABLE_IMPORT_SECURITY=1 \
    PYTHONSAFEPATH=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# ---------------------------------------------------------------------------
# System dependencies (OpenMP for XGBoost/LightGBM)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# The pinned core ML versions match the environment that produced the
# serialized model artifacts (joblib pickles are version-sensitive).
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        numpy==2.5.1 \
        scipy==1.18.0 \
        scikit-learn==1.9.0 \
        joblib==1.5.3 \
        pandas==3.0.5 \
        xgboost==3.4.0 \
        lightgbm==4.7.0 \
        streamlit==1.60.0 \
        fastapi==0.141.1 \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# NLTK data (tokenizer + lemmatizer resources used by the preprocessing step)
# ---------------------------------------------------------------------------
RUN python -c "import nltk; \
    [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4')]"

# ---------------------------------------------------------------------------
# Application code and pre-trained artifacts
# ---------------------------------------------------------------------------
COPY main.py .
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY models/ models/
COPY reports/ reports/
COPY images/ images/
COPY dataset/processed/cleaned_dataset.csv dataset/processed/

# ---------------------------------------------------------------------------
# Ports: 8501 (Streamlit UI), 8000 (FastAPI REST API)
# ---------------------------------------------------------------------------
EXPOSE 8501 8000

# Default command: launch the Streamlit web application.
# Override in docker-compose.yml (or via `docker run ... command=`) to run the
# FastAPI service instead.
CMD ["streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
