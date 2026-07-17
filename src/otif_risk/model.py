"""Train and evaluate the standalone OTIF-miss risk model."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "otif_miss"
ID_COLUMNS = ("order_id", "as_of_timestamp")
ENDPOINT = "OTIF_MISS"
ThresholdStrategy = Literal["capacity", "recall_floor", "f1_max"]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    strategy: ThresholdStrategy
    validation_metrics: dict[str, Any]


@dataclass
class SigmoidCalibrator:
    """Platt-style sigmoid calibration fitted on validation predictions."""

    model: LogisticRegression | None = None
    constant: float | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> SigmoidCalibrator:
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if np.unique(labels).size < 2:
            self.constant = float((labels.sum() + 1) / (labels.size + 2))
            self.model = None
            return self
        logits = _logit(probabilities).reshape(-1, 1)
        self.model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
        self.model.fit(logits, labels)
        self.constant = None
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if self.constant is not None:
            return np.full(probabilities.shape, self.constant, dtype=float)
        if self.model is None:
            raise RuntimeError("calibrator has not been fitted")
        return self.model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


@dataclass
class RiskBundle:
    """Serializable preprocessing, model, calibration, and decision threshold."""

    pipeline: Pipeline
    calibrator: SigmoidCalibrator
    feature_columns: tuple[str, ...]
    threshold: float
    model_kind: str
    threshold_strategy: ThresholdStrategy
    endpoint: str = ENDPOINT

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return calibrated OTIF-miss probabilities."""
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"missing model features: {missing}")
        raw = self.pipeline.predict_proba(frame.loc[:, self.feature_columns])[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "order_id" not in frame:
            raise ValueError("frame must contain order_id")
        return pd.DataFrame(
            {
                "order_id": frame["order_id"].to_numpy(),
                "risk_model_score": self.predict_proba(frame),
            }
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= self.threshold).astype(int)


@dataclass(frozen=True)
class TrainingResult:
    bundle: RiskBundle
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    threshold_selection: ThresholdSelection
    capacity_baseline_metrics: dict[str, Any]


def train_risk_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    planner_capacity_fraction: float = 0.15,
    threshold_strategy: ThresholdStrategy = "recall_floor",
    target_recall: float = 0.55,
    min_precision: float = 0.35,
    random_state: int = 42,
) -> TrainingResult:
    """Train one model, calibrate on validation, and evaluate validation/test."""
    _validate_splits(train, validation, test, planner_capacity_fraction)
    feature_columns = tuple(
        column for column in train.columns if column not in {*ID_COLUMNS, TARGET_COLUMN}
    )
    numeric = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])
    ]
    categorical = [column for column in feature_columns if column not in numeric]
    preprocessor = _make_preprocessor(numeric, categorical)

    pipeline, model_kind = _fit_pipeline(
        preprocessor,
        train.loc[:, feature_columns],
        train[TARGET_COLUMN].astype(int).to_numpy(),
        random_state,
    )
    validation_raw = pipeline.predict_proba(validation.loc[:, feature_columns])[:, 1]
    calibrator = SigmoidCalibrator().fit(
        validation_raw, validation[TARGET_COLUMN].astype(int).to_numpy()
    )
    validation_probabilities = calibrator.predict(validation_raw)
    validation_labels = validation[TARGET_COLUMN].astype(int).to_numpy()
    threshold_selection = select_threshold(
        validation_labels,
        validation_probabilities,
        strategy=threshold_strategy,
        capacity_fraction=planner_capacity_fraction,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    capacity_threshold_value = capacity_threshold(
        validation_probabilities, planner_capacity_fraction
    )
    bundle = RiskBundle(
        pipeline=pipeline,
        calibrator=calibrator,
        feature_columns=feature_columns,
        threshold=threshold_selection.threshold,
        model_kind=model_kind,
        threshold_strategy=threshold_strategy,
    )
    return TrainingResult(
        bundle=bundle,
        validation_metrics=threshold_selection.validation_metrics,
        test_metrics=evaluate_predictions(
            test[TARGET_COLUMN], bundle.predict_proba(test), bundle.threshold
        ),
        threshold_selection=threshold_selection,
        capacity_baseline_metrics=evaluate_predictions(
            validation_labels,
            validation_probabilities,
            capacity_threshold_value,
        ),
    )


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    planner_capacity_fraction: float = 0.15,
    random_state: int = 42,
) -> TrainingResult:
    """Convenience alias using a shorter name."""
    return train_risk_model(
        train, validation, test, planner_capacity_fraction, random_state=random_state
    )


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    strategy: ThresholdStrategy = "recall_floor",
    capacity_fraction: float = 0.15,
    target_recall: float = 0.55,
    min_precision: float = 0.35,
) -> ThresholdSelection:
    """Pick an operating point on the validation set."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if strategy == "capacity":
        threshold = capacity_threshold(probabilities, capacity_fraction)
    elif strategy == "f1_max":
        threshold = _f1_max_threshold(labels, probabilities)
    elif strategy == "recall_floor":
        threshold = _recall_floor_threshold(
            labels, probabilities, target_recall, min_precision
        )
    else:
        raise ValueError(f"unsupported threshold strategy: {strategy}")
    return ThresholdSelection(
        threshold=threshold,
        strategy=strategy,
        validation_metrics=evaluate_predictions(labels, probabilities, threshold),
    )


def capacity_threshold(probabilities: np.ndarray, capacity_fraction: float) -> float:
    """Set the cutoff to the score of the last order within planner capacity."""
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        raise ValueError("cannot tune a threshold on an empty validation set")
    capacity = max(1, int(np.ceil(values.size * capacity_fraction)))
    return float(np.sort(values)[::-1][capacity - 1])


def evaluate_predictions(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": _safe_auc(average_precision_score, y_true, probabilities),
        "roc_auc": _safe_auc(roc_auc_score, y_true, probabilities),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "flagged_orders": int(predicted.sum()),
    }


def _recall_floor_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
    min_precision: float,
) -> float:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not 0 <= min_precision <= 1:
        raise ValueError("min_precision must be in [0, 1]")
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.unique(probabilities):
        predicted = (probabilities >= threshold).astype(int)
        recall = float(recall_score(labels, predicted, zero_division=0))
        precision = float(precision_score(labels, predicted, zero_division=0))
        if recall >= target_recall and precision >= min_precision:
            candidates.append((float(threshold), precision, recall))
    if candidates:
        return max(candidates, key=lambda item: item[0])[0]
    return _f1_max_threshold(labels, probabilities)


def _f1_max_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.unique(probabilities):
        predicted = (probabilities >= threshold).astype(int)
        score = float(f1_score(labels, predicted, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _configure_openmp_runtime() -> None:
    if sys.platform != "darwin":
        return
    for prefix in ("/opt/homebrew", "/usr/local"):
        libomp = Path(prefix) / "opt/libomp/lib"
        if (libomp / "libomp.dylib").exists():
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            if str(libomp) not in current.split(":"):
                os.environ["DYLD_LIBRARY_PATH"] = (
                    f"{libomp}:{current}" if current else str(libomp)
                )
            break


def _make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = [
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ]
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _fit_pipeline(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
    labels: np.ndarray,
    random_state: int,
) -> tuple[Pipeline, str]:
    _configure_openmp_runtime()
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - platform-specific import failure
        raise RuntimeError(
            "XGBoost is required for this prototype. On macOS install OpenMP with "
            "`brew install libomp`, then rerun. Original error: "
            f"{exc}"
        ) from exc

    estimator: Any = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    try:
        pipeline.fit(features, labels)
    except Exception as exc:
        raise RuntimeError(
            "XGBoost training failed. On macOS ensure OpenMP is installed with "
            "`brew install libomp`. Original error: "
            f"{exc}"
        ) from exc
    return pipeline, "xgboost"


def _safe_auc(metric: Any, labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, probabilities))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    capacity: float,
) -> None:
    if not 0 < capacity <= 1:
        raise ValueError("planner_capacity_fraction must be in (0, 1]")
    required = {*ID_COLUMNS, TARGET_COLUMN}
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} split is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} split must not be empty")
    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("training labels must contain both classes")
    validation_differs = tuple(train.columns) != tuple(validation.columns)
    test_differs = tuple(train.columns) != tuple(test.columns)
    if validation_differs or test_differs:
        raise ValueError("train, validation, and test must have identical columns")
