"""Synthetic screening decisions only; no checkpoint or public cases."""

import math

import pytest

from looking_glass import targeted_edit_screened_selection as selection


def cases():
    return [{"case_id": f"{group}-{index}", "role": "target", "stratum": group}
            for group in selection.STRATA for index in range(8)]


def test_round_robin_uses_frozen_order_not_margin_magnitude():
    margins = [-0.01 - i for i in range(32)]
    result = selection.select_screen(cases(), margins)
    assert result["selected_offsets"] == [0, 8, 16, 24, 1, 9, 17, 25]
    assert result["selected_counts_by_stratum"] == {s: 2 for s in selection.STRATA}
    assert result["eligible_count"] == 32 and result["formal_target_eligible"]
    assert sum(row["selected"] for row in result["rows"]) == 8


def test_skips_ineligible_and_exhausted_strata_without_replacement():
    margins = [0.0] * 32
    for index in (1, 2, 3, 4, 5, 6, 7, 9, 17):
        margins[index] = -1.0
    result = selection.select_screen(cases(), margins)
    assert result["selected_offsets"] == [1, 9, 17, 2, 3, 4, 5, 6]
    assert result["eligible_counts_by_stratum"] == {"s1": 7, "s2": 1, "s3": 1, "s4": 0}
    assert result["selected_counts_by_stratum"] == {"s1": 6, "s2": 1, "s3": 1, "s4": 0}


@pytest.mark.parametrize("count", range(8))
def test_underfilled_screen_stops_and_marks_both_formal_arms_ineligible(count):
    margins = [-1.0] * count + [0.0] * (32 - count)
    result = selection.select_screen(cases(), margins)
    assert result["selected_offsets"] == list(range(count))
    assert result["eligible_count"] == count
    assert result["selected_count"] == count
    assert result["formal_target_eligible"] is False


def test_zero_including_negative_zero_never_qualifies():
    result = selection.select_screen(cases(), [-0.0, 0.0] + [1.0] * 30)
    assert result["selected_ids"] == []
    assert not any(row["eligible"] for row in result["rows"])


@pytest.mark.parametrize("bad", [True, math.inf, -math.inf, math.nan, 10 ** 1000, "-1"])
def test_invalid_margin_rejected(bad):
    with pytest.raises(selection.SelectionError, match="finite"):
        selection.select_screen(cases(), [bad] + [0.0] * 31)


def test_rejects_duplicate_ids_wrong_stratum_role_and_wrong_lengths():
    for change in ("duplicate", "stratum", "role", "length"):
        rows = cases()
        margins = [0.0] * 32
        if change == "duplicate":
            rows[1]["case_id"] = rows[0]["case_id"]
        elif change == "stratum":
            rows[0]["stratum"] = "s2"
        elif change == "role":
            rows[0]["role"] = "locality"
        else:
            margins.pop()
        with pytest.raises(selection.SelectionError):
            selection.select_screen(rows, margins)
