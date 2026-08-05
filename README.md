# AI-Driven Phishing Email Detection Using NLP

A production-grade machine learning system that detects phishing emails using Natural Language Processing (NLP), feature engineering, and six benchmarked classifiers. Ships with a Streamlit dashboard, a FastAPI REST API, a CLI training pipeline, unit tests, and Docker support.

---

## Key Features

- **Multi-Source Data Ingestion** — merges 7 public email datasets (~165K records: Enron, SpamAssassin, CEAS 08, Ling, Nazario, Nigerian Fraud, Phishing Email) into a unified corpus.
- **NLP Preprocessing Pipeline** — HTML stripping, URL/email removal, signature-stripping, punctuation/number/emoji removal, tokenization, stopword removal, and WordNet lemmatization.
- **Advanced Feature Engineering** — combines a 10,000-term TF-IDF (unigrams + bigrams) sparse matrix with 15 engineered metadata features (URL count, uppercase ratio, social-engineering keyword score, subject stats, etc.).
- **6 ML Classifiers Benchmarked** — Logistic Regression, Naive Bayes, Random Forest, XGBoost, LightGBM, and an MLP Neural Network, each evaluated on accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- **Production Inference Engine** — classify raw text, `.txt`/`.eml` uploads, or CSV batches, with risk scoring (0–100), risk levels, suspicious-keyword highlighting, URL extraction, natural-language explanations, and downloadable reports.
- **FastAPI REST API** — `/predict`, `/predict/file`, `/predict/batch`, `/predict/csv`, `/models`, with auto-generated docs at `/docs`.
- **Streamlit Dashboard** — cyber-themed UI with live risk gauge, model leaderboard, dataset analytics, confusion matrices, ROC/PR curves, and feature importance.
- **Docker Ready** — one-command startup for the UI and API.

---

## Project Architecture

```
AI_Driven_Phishing_Email_Detection/
│
├── dataset/
│   ├── raw/                   # 7 source CSV datasets (not versioned)
│   └── processed/             # merged_dataset.csv, cleaned_dataset.csv
│
├── src/                       # Core pipeline modules
│   ├── __init__.py            # NLTK import-security workaround
│   ├── config.py              # Paths, hyperparameters, constants (.env aware)
│   ├── utils.py               # Logging, timers, artifact/JSON I/O
│   ├── preprocessing.py       # DatasetLoader + TextPreprocessor
│   ├── feature_engineering.py # TF-IDF + 15 metadata features
│   ├── train_models.py        # Multi-model training pipeline
│   ├── evaluate_models.py     # Metrics, comparison table, visualizations
│   ├── predict.py             # PhishingDetector inference engine
│   └── api.py                 # FastAPI REST API
│
├── app/                       # Streamlit application
│   ├── app.py                 # Entry point (multipage)
│   ├── components/            # sidebar, theme, cards, detector, history, footer
│   └── pages/                 # dashboard, email_detector, dataset_analytics,
│                              # model_performance, about
│
├── models/                    # Fitted models, vectorizer, scaler, best model
├── reports/                   # model_comparison.csv, *.json evaluation reports
├── images/                    # Wordclouds, ROC/PR curves, confusion matrices
├── tests/                     # Pytest unit + integration tests
│
├── main.py                    # CLI training pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.10+** (developed and tested on Python 3.12/3.14)
- A machine with ~8 GB free RAM (training the Neural Network is memory-heavy)

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment (optional)

```bash
cp .env.example .env   # then edit values as needed
```

`USE_CACHED_CLEANED_DATA=true` (default) reuses `dataset/processed/cleaned_dataset.csv` so repeated training runs skip the slow re-cleaning step.

---

## Training Pipeline (CLI)

Run the complete data ingestion → preprocessing → feature engineering → training → evaluation → visualization pipeline:

```bash
python main.py
```

Useful flags:

| Flag | Description |
|---|---|
| `--no-clean` | Skip dataset loading/cleaning and train straight from cached data |
| `--models LR NB` | Train only the specified models (names are display names) |
| `--save-train-test` | Persist the train/test CSVs to `dataset/processed/` |

Outputs:

- `models/*.joblib` — all 6 trained classifiers
- `models/best_model.joblib` + `models/best_model_info.json` — best model + metadata
- `models/tfidf_vectorizer.joblib`, `models/metadata_scaler.joblib`, `models/feature_names.joblib`
- `reports/model_comparison.csv`, `reports/model_results.json`, `reports/classification_reports.json`, `reports/dataset_stats.json`
- `images/*.png` — wordclouds, class distribution, length histogram, top words, confusion matrices, ROC/PR curves, feature importance, correlation heatmap

> **Note:** A full run with all 6 models takes ~25–30 minutes (Neural Network ≈ 13 min, XGBoost ≈ 4–5 min).

---

## Running the Web Application

The easiest way is the one-click launcher — no copying URLs or pasting them into
the browser:

**Windows:** double-click `start.bat`, or run:

```bash
python run.py
```

The launcher starts the Streamlit server headlessly, waits until it is ready,
and opens the app automatically in your default browser at
`http://127.0.0.1:8501`. Streamlit log output goes to `logs/app_streamlit.log`
instead of the terminal. Close the launcher window (or press `Ctrl+C`) to stop
the app.

Advanced options:

```bash
python run.py --port 8502   # use a different port
python run.py --no-browser  # start the server without opening the browser
```

If the app is already running, running the launcher again simply opens the
browser to the existing instance.

Pages:

- **Dashboard** — KPIs, class distribution, model leaderboard, prediction history
- **Email Detector** — single email / batch CSV scanning with live risk gauge, explanations, and report download
- **Dataset Analytics** — distributions, wordclouds, top words, length histograms, correlation heatmap
- **Model Performance** — comparison table, ROC/PR curves, confusion matrices, feature importance, per-class classification reports
- **About** — architecture and tech stack

---

## REST API

Start the FastAPI server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check + model status |
| `/models` | GET | List available models |
| `/predict` | POST | Predict a single email (JSON) |
| `/predict/file` | POST | Predict from an uploaded `.txt`/`.eml` file |
| `/predict/batch` | POST | Predict a list of emails |
| `/predict/csv` | POST | Predict on an uploaded CSV batch |

### Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"URGENT: Your account has been suspended. Verify your password now.","subject":"Action required"}'

# File upload
curl -X POST http://localhost:8000/predict/file \
  -F "file=@./sample.eml"

# Batch
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Hello team, meeting at 3pm", "URGENT verify your account"]}'
```

---

## Model Performance (latest full run)

Trained on ~127K emails (80/20 stratified split, 10,015 combined features).

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Train Time |
|---|---|---|---|---|---|---|---|
| **Neural Network (MLP)** | **99.31%** | **99.49%** | **99.18%** | **99.34%** | **99.96%** | **99.96%** | ~13.1 min |
| Logistic Regression | 98.89% | 98.89% | 98.96% | 98.93% | 99.92% | 99.92% | ~5.8 s |
| LightGBM | 98.81% | 98.60% | 99.10% | 98.85% | 99.94% | 99.95% | ~2.0 min |
| XGBoost | 97.90% | 97.19% | 98.79% | 97.99% | 99.81% | 99.82% | ~4.3 min |
| Random Forest | 97.89% | 97.32% | 98.64% | 97.98% | 99.80% | 99.81% | ~56 s |
| Naive Bayes | 96.30% | 98.83% | 93.95% | 96.33% | 99.57% | 99.58% | ~0.04 s |

Full details in `reports/model_comparison.csv` and `reports/classification_reports.json`.

---

## Running Tests

```bash
python -m pytest tests -q
```

Unit tests cover config paths, the NLP cleaning pipeline, metadata feature extraction, prediction helpers, and report generation. End-to-end inference tests run automatically when `models/best_model.joblib` exists.

---

## Docker

Build the image and start both the UI and API:

```bash
docker compose up --build
```

- Streamlit UI → `http://localhost:8501`
- FastAPI API → `http://localhost:8000` (docs at `/docs`)

Or run a single service manually:

```bash
# Streamlit only
docker build -t ai-phishing-shield .
docker run -p 8501:8501 ai-phishing-shield

# API only
docker run -p 8000:8000 ai-phishing-shield \
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The image bundles the pre-trained artifacts, so it is usable immediately without re-training.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `SecurityError` / blocked NLTK imports on import | `src/__init__.py` sets `NLTK_DISABLE_IMPORT_SECURITY=1` and `PYTHONSAFEPATH=1`. Keep these lines; do not run from inside a venv subdirectory that trips NLTK's import hook. |
| `LookupError: punkt_tab not found` | `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"` |
| Models won't load / pickling errors | The joblib artifacts are sensitive to library versions. Keep `numpy`, `scikit-learn`, `pandas`, `xgboost`, `lightgbm` aligned with the versions that trained the models (see the Dockerfile pins). |
| Training is very slow | Set `USE_CACHED_CLEANED_DATA=true` and reuse the existing cleaned dataset; or use `--models LR NB` for a quick smoke run. |
| Streamlit port already in use | The launcher auto-selects a free port; or pass `python run.py --port 8502`. |
| MemoryError during Neural Network training | Reduce `hidden_layer_sizes` / `batch_size` in `src/config.py` (`MODEL_CONFIGS["Neural Network"]`). |

---

## Project Context

- **Project Title:** AI-Driven Phishing Email Detection Using NLP
- **Institution:** Indian Institute of Computing and Technology (IICT)
- **Author:** B.Tech Final Year Student Group
- **Academic Year:** 2025 – 2026
- **Objective:** Design a production-grade machine learning application that uses NLP text analytics and structural metadata to detect email threat vectors.
