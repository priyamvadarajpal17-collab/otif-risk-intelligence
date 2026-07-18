from __future__ import annotations

import json

import pytest

from otif_risk.registry import (
    HELD,
    PROMOTED,
    REGISTERED,
    ROLLED_BACK,
    ModelMetrics,
    ModelRegistry,
    ModelVersion,
    PromotionTolerances,
    evaluate_promotion,
)


def _metrics(**overrides) -> ModelMetrics:
    base = dict(
        pr_auc=0.72,
        brier=0.15,
        calibration_error=0.05,
        recall=0.68,
        alert_rate=0.20,
        drift_regime_pr_auc=0.60,
        normal_regime_pr_auc=0.75,
        policy_value_50pct_capacity=8.0,
        schema_valid=True,
        leakage_gate_passed=True,
        manifest_verified=True,
    )
    base.update(overrides)
    return ModelMetrics(**base)


def _version(version_id: str, metrics: ModelMetrics, parent: str | None = None) -> ModelVersion:
    return ModelVersion(
        version_id=version_id,
        trained_at_utc="2024-01-01T00:00:00+00:00",
        manifest_content_id=f"content-{version_id}",
        metrics=metrics,
        artifact_paths={"xgboost": f"models/{version_id}/xgboost_risk.joblib"},
        parent_version_id=parent,
    )


def test_evaluate_promotion_passes_when_challenger_matches_or_improves():
    champion = _metrics()
    challenger = _metrics(pr_auc=0.73, policy_value_50pct_capacity=8.2)
    decision = evaluate_promotion(champion, challenger)
    assert decision.decision == PROMOTED
    assert decision.reasons == []
    assert all(result.passed for result in decision.gate_results.values())


def test_evaluate_promotion_holds_on_policy_value_regression():
    """The Bayesian-enhanced-challenger scenario: prediction metrics are fine
    but the measured policy value regresses beyond tolerance -> HELD."""
    champion = _metrics(policy_value_50pct_capacity=7.7729)
    challenger = _metrics(policy_value_50pct_capacity=6.4)  # ~ -17.7%, negative Bayesian ablation

    decision = evaluate_promotion(champion, challenger)

    assert decision.decision == HELD
    assert any("policy value" in reason for reason in decision.reasons)
    assert decision.gate_results["policy_value_50pct_capacity"].passed is False
    # Every other gate independently still passed -- only policy value failed.
    assert decision.gate_results["pr_auc"].passed is True


def test_evaluate_promotion_holds_on_pr_auc_regression():
    champion = _metrics(pr_auc=0.75)
    challenger = _metrics(pr_auc=0.60)
    decision = evaluate_promotion(champion, challenger)
    assert decision.decision == HELD
    assert decision.gate_results["pr_auc"].passed is False


def test_evaluate_promotion_holds_when_manifest_unverified():
    champion = _metrics()
    challenger = _metrics(manifest_verified=False)
    decision = evaluate_promotion(champion, challenger)
    assert decision.decision == HELD
    assert decision.gate_results["manifest_verified"].passed is False


def test_evaluate_promotion_reports_multiple_simultaneous_failures():
    champion = _metrics()
    challenger = _metrics(pr_auc=0.5, recall=0.4, schema_valid=False)
    decision = evaluate_promotion(champion, challenger)
    assert decision.decision == HELD
    assert len(decision.reasons) >= 3


def test_evaluate_promotion_respects_custom_tolerances():
    champion = _metrics(pr_auc=0.75)
    challenger = _metrics(pr_auc=0.74)
    strict = PromotionTolerances(max_pr_auc_regression=0.001)
    loose = PromotionTolerances(max_pr_auc_regression=0.05)
    assert evaluate_promotion(champion, challenger, strict).decision == HELD
    assert evaluate_promotion(champion, challenger, loose).decision == PROMOTED


def test_register_version_is_immutable(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register_version(_version("v1", _metrics()))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_version(_version("v1", _metrics()))


def test_register_version_appends_registered_event(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register_version(_version("v1", _metrics()))
    history = registry.history()
    assert history[0]["event"] == REGISTERED
    assert history[0]["version_id"] == "v1"


def test_promote_or_hold_sets_active_pointer_only_on_promotion(tmp_path):
    registry = ModelRegistry(tmp_path)
    champion_metrics = _metrics()
    registry.register_version(_version("v1", champion_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(champion_metrics, champion_metrics))
    assert registry.active_version() == "v1"

    challenger_metrics = _metrics(policy_value_50pct_capacity=1.0)  # heavy regression
    registry.register_version(_version("v2", challenger_metrics, parent="v1"))
    decision = evaluate_promotion(champion_metrics, challenger_metrics)
    assert decision.decision == HELD
    registry.promote_or_hold("v2", decision)

    # Held challenger must never move the active pointer.
    assert registry.active_version() == "v1"
    history = registry.history()
    assert any(event["event"] == HELD and event["version_id"] == "v2" for event in history)


def test_promote_or_hold_unknown_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path)
    champion_metrics = _metrics()
    with pytest.raises(ValueError, match="unknown challenger version"):
        registry.promote_or_hold(
            "nonexistent", evaluate_promotion(champion_metrics, champion_metrics)
        )


def test_rollback_to_verified_version_succeeds(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_metrics = _metrics()
    v2_metrics = _metrics(pr_auc=0.5)  # would fail promotion
    registry.register_version(_version("v1", v1_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(v1_metrics, v1_metrics))
    registry.register_version(_version("v2", v2_metrics, parent="v1"))
    registry.promote_or_hold("v2", evaluate_promotion(v1_metrics, v2_metrics))
    assert registry.active_version() == "v1"

    # Suppose v1 needs to be rolled back to itself (a verified prior version)
    # after some later, since-abandoned promotion attempt.
    result = registry.rollback("v1")
    assert result["rolled_back"] is True
    assert registry.active_version() == "v1"
    history = registry.history()
    assert any(event["event"] == ROLLED_BACK and event["rolled_back"] for event in history)


def test_rollback_to_unverified_version_fails_and_leaves_pointer_unchanged(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_metrics = _metrics()
    registry.register_version(_version("v1", v1_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(v1_metrics, v1_metrics))

    unverified_metrics = _metrics(manifest_verified=False)
    registry.register_version(_version("v2", unverified_metrics, parent="v1"))

    result = registry.rollback("v2")
    assert result["rolled_back"] is False
    assert "manifest" in result["reason"]
    assert registry.active_version() == "v1"


def test_rollback_to_unknown_version_fails(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_metrics = _metrics()
    registry.register_version(_version("v1", v1_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(v1_metrics, v1_metrics))

    result = registry.rollback("does-not-exist")
    assert result["rolled_back"] is False
    assert registry.active_version() == "v1"


def test_events_are_append_only_and_never_rewritten(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_metrics = _metrics()
    registry.register_version(_version("v1", v1_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(v1_metrics, v1_metrics))
    registry.rollback("v1")

    raw_lines = registry.events_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    # Every line must remain independently parseable JSON (never merged/rewritten).
    for line in raw_lines:
        json.loads(line)


def test_active_model_pointer_is_written_atomically(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_metrics = _metrics()
    registry.register_version(_version("v1", v1_metrics))
    registry.promote_or_hold("v1", evaluate_promotion(v1_metrics, v1_metrics))

    # No leftover temp files after a successful write.
    leftovers = list(tmp_path.glob(".*tmp*"))
    assert leftovers == []
    payload = json.loads(registry.active_pointer_path.read_text(encoding="utf-8"))
    assert payload["active_version_id"] == "v1"
