"""Local, non-causal explanations for OTIF-miss risk scores."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from otif_risk.model import RiskBundle


def explain_predictions(
    bundle: RiskBundle,
    frame: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return ranked local associations, using SHAP when it is compatible."""
    if "order_id" not in frame:
        raise ValueError("frame must contain order_id")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    try:
        factors = _shap_factors(bundle, frame, background, top_n)
    except Exception:  # SHAP/backend compatibility varies across supported environments.
        factors = _perturbation_factors(bundle, frame, background, top_n)
    return pd.DataFrame(
        {
            "order_id": frame["order_id"].to_numpy(),
            "top_factors_json": [json.dumps(items, separators=(",", ":")) for items in factors],
        }
    )


def explain_risk(
    bundle: RiskBundle,
    frame: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compatibility alias for risk explanation."""
    return explain_predictions(bundle, frame, background=background, top_n=top_n)


def _shap_factors(
    bundle: RiskBundle,
    frame: pd.DataFrame,
    background: pd.DataFrame | None,
    top_n: int,
) -> list[list[dict[str, object]]]:
    import shap

    sample = frame.loc[:, bundle.feature_columns]
    reference = sample if background is None else background.loc[:, bundle.feature_columns]
    preprocessor = bundle.pipeline.named_steps["preprocess"]
    estimator = bundle.pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    transformed_reference = preprocessor.transform(reference.iloc[: min(len(reference), 100)])
    names = list(preprocessor.get_feature_names_out())
    explainer = shap.Explainer(estimator, transformed_reference, feature_names=names)
    values = np.asarray(explainer(transformed).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != transformed.shape:
        raise ValueError("unsupported SHAP output shape")
    return [_rank_factors(names, row, top_n, "shap_association") for row in values]


def _perturbation_factors(
    bundle: RiskBundle,
    frame: pd.DataFrame,
    background: pd.DataFrame | None,
    top_n: int,
) -> list[list[dict[str, object]]]:
    sample = frame.loc[:, bundle.feature_columns]
    reference = sample if background is None else background.loc[:, bundle.feature_columns]
    baselines = {column: _baseline_value(reference[column]) for column in bundle.feature_columns}
    original = bundle.predict_proba(frame)
    contributions = np.empty((len(frame), len(bundle.feature_columns)), dtype=float)
    for index, column in enumerate(bundle.feature_columns):
        perturbed = frame.copy()
        perturbed[column] = baselines[column]
        contributions[:, index] = original - bundle.predict_proba(perturbed)
    return [
        _rank_factors(
            list(bundle.feature_columns),
            contributions[row_index],
            top_n,
            "local_perturbation_association",
        )
        for row_index in range(len(frame))
    ]


def _baseline_value(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0 if pd.api.types.is_numeric_dtype(series) else "__MISSING__"
    if pd.api.types.is_numeric_dtype(series):
        return float(non_null.median())
    return non_null.mode().iloc[0]


def _rank_factors(
    names: list[str],
    contributions: np.ndarray,
    top_n: int,
    method: str,
) -> list[dict[str, object]]:
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {
            "factor": names[index],
            "contribution": round(float(contributions[index]), 6),
            "direction": "higher_risk" if contributions[index] >= 0 else "lower_risk",
            "interpretation": "association_not_causation",
            "method": method,
        }
        for index in order
    ]
