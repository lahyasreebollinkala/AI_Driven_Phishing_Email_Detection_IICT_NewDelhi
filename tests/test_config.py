"""Tests for src/config.py — paths, label maps, and helpers."""

import os

import pytest

from src import config


def test_project_root_exists():
    assert os.path.isdir(config.PROJECT_ROOT)


def test_directory_paths_point_inside_project():
    for attr in ("MODELS_DIR", "REPORTS_DIR", "IMAGES_DIR", "LOGS_DIR", "PROCESSED_DATA_DIR"):
        path = getattr(config, attr)
        assert path.startswith(config.PROJECT_ROOT), attr


def test_label_maps_are_inverse():
    for numeric, name in config.LABEL_MAP.items():
        assert config.LABEL_REVERSE_MAP[name] == numeric


def test_model_configs_cover_all_models():
    expected = {
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Neural Network",
    }
    assert set(config.MODEL_CONFIGS.keys()) == expected


def test_every_model_config_has_class_and_params():
    for name, cfg in config.MODEL_CONFIGS.items():
        assert "class" in cfg, name
        assert isinstance(cfg["params"], dict), name


def test_risk_thresholds_are_monotonic():
    t = config.RISK_THRESHOLDS
    assert t["low"] < t["medium"] < t["high"] < t["critical"]


def test_model_file_path_generates_expected_name():
    assert config.model_file_path("Random Forest") == os.path.join(
        config.MODELS_DIR, "random_forest.joblib"
    )


def test_model_file_path_roundtrip_for_all_configs():
    for name in config.MODEL_CONFIGS:
        assert config.model_file_path(name).endswith(".joblib")
