from __future__ import annotations

from otif_risk.narratives import order_narrative, parse_top_factors


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


def test_narrative_renders_pathway_route_affected_skus_and_resource_status() -> None:
    order = {
        "order_id": "O-99",
        "combined_risk_score": 0.61,
        "primary_cause": "VENDOR_FAILURE",
        "top_factors_json": '["vendor_ready_delay_hours"]',
        "causal_pathway": (
            '{"route": ["VENDOR_FAILURE", "INVENTORY_SHORTAGE", "WAREHOUSE_OPS", '
            '"TRANSPORT", "OTIF_MISS"], "posterior_risk": 0.6}'
        ),
        "affected_skus_json": '[{"sku_id": "SKU0042", "evidence_strength": 0.5}]',
        "decision_status": "CONTESTED",
        "resource_type": "vendor",
        "resource_id": "V001",
        "contested_with": "O-100",
        "recommended_action": "Escalate supplier recovery",
    }

    narrative = order_narrative(order)

    expected_route = (
        "VENDOR_FAILURE -> INVENTORY_SHORTAGE -> WAREHOUSE_OPS -> TRANSPORT -> OTIF_MISS"
    )
    assert expected_route in narrative
    assert "SKU0042" in narrative
    assert "contested" in narrative
    assert "O-100" in narrative
    assert "\n" not in narrative


def test_narrative_degrades_gracefully_without_pathway_or_sku_evidence() -> None:
    order = {
        "order_id": "O-1",
        "combined_risk_score": 0.2,
        "primary_cause": "ON_TIME",
        "decision_status": "MONITOR",
    }
    narrative = order_narrative(order)
    assert "no causal pathway" in narrative
    assert "no affected-SKU evidence" in narrative
    assert "resource status: monitor" in narrative
