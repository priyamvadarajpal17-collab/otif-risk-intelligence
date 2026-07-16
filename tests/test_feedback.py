from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from otif_pdf.feedback import append_feedback


def test_feedback_is_appended_without_rewriting_prior_rows(tmp_path) -> None:
    path = tmp_path / "feedback.csv"
    timestamp = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    append_feedback(
        path,
        order_id="O-1",
        feedback_action="accept",
        original_status="RECOMMENDED",
        original_recommendation="Expedite",
        timestamp=timestamp,
    )
    first_bytes = path.read_bytes()
    append_feedback(
        path,
        order_id="O-2",
        feedback_action="OVERRIDE",
        override_recommendation="Use alternate carrier",
        reason="Preferred lane is full",
        timestamp=timestamp,
    )

    rows = pd.read_csv(path)
    assert path.read_bytes().startswith(first_bytes)
    assert rows["feedback_action"].tolist() == ["ACCEPT", "OVERRIDE"]
    assert rows.loc[1, "override_recommendation"] == "Use alternate carrier"


@pytest.mark.parametrize(
    ("action", "reason", "override"),
    [
        ("UNKNOWN", "", ""),
        ("REJECT", "", ""),
        ("OVERRIDE", "capacity conflict", ""),
        ("ACCEPT", "", "unexpected replacement"),
    ],
)
def test_invalid_feedback_is_rejected(tmp_path, action, reason, override) -> None:
    with pytest.raises(ValueError):
        append_feedback(
            tmp_path / "feedback.csv",
            order_id="O-1",
            feedback_action=action,
            reason=reason,
            override_recommendation=override,
        )
