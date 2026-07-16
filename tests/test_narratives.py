from __future__ import annotations

from otif_pdf.narratives import order_narrative, parse_top_factors


def test_parse_top_factors_supports_pipeline_json_shapes() -> None:
    value = '[{"feature": "carrier_delay", "importance": 0.4}, {"factor": "dc_load"}]'

    assert parse_top_factors(value) == ["carrier delay", "dc load"]
    assert parse_top_factors('{"inventory_gap": -0.8, "vendor_rate": 0.2}') == [
        "inventory gap",
        "vendor rate",
    ]


def test_narrative_is_one_line_deterministic_and_complete() -> None:
    order = {
        "order_id": "O-12",
        "combined_risk_score": 0.73,
        "primary_cause": "TRANSPORT",
        "top_factors_json": '["carrier_delay", "weather_risk"]',
        "causal_pathway": "carrier capacity → dispatch delay → late delivery",
        "decision_status": "RECOMMENDED",
        "recommended_action": "Secure alternate transport capacity",
    }

    first = order_narrative(order)

    assert first == order_narrative(order)
    assert "\n" not in first
    assert "73% OTIF risk" in first
    assert "carrier delay, weather risk" in first
    assert "Secure alternate transport capacity" in first
