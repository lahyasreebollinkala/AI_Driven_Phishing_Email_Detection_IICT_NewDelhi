# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Source Package Initializer
# =============================================================================
"""
src package — Contains all core modules for the phishing detection pipeline.

Modules:
    config              : Central configuration (paths, hyperparams, constants)
    utils               : Logging, timing, I/O helpers
    preprocessing       : NLP text cleaning pipeline
    feature_engineering : TF-IDF + metadata feature extraction
    train_models        : Model training with timing
    evaluate_models     : Metrics computation, comparison charts
    predict             : Inference engine for production use
    api                 : FastAPI REST interface
"""

import os

# ---------------------------------------------------------------------------
# NLTK 3.10+ import-security workaround
# ---------------------------------------------------------------------------
# NLTK installs a MetaPathFinder that blocks its own dependencies from being
# imported when their on-disk location resolves *inside* the current working
# directory. Because this project's virtualenv lives at `.venv/` inside the
# project root, EVERY site-packages module triggers that check, which breaks
# `import nltk` entirely when running from the project root.
#
# The NLTK team explicitly provides `NLTK_DISABLE_IMPORT_SECURITY=1` as the
# escape hatch for exactly this situation (venv inside a trusted project dir).
# We set it before any NLTK code is imported. See .env.example and README.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
# Also isolate the CWD from sys.path for freshly spawned worker interpreters
# (joblib/loky) as recommended by NLTK's security docs.
os.environ.setdefault("PYTHONSAFEPATH", "1")

__version__ = "2.0.0"
__author__ = "AI Phishing Detection Team"
