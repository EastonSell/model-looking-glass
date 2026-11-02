"""The frozen potential schedule is independent of model results."""

from collections import Counter

import pytest

from looking_glass.flan_multitoken_schedule_v2 import potential_schedule, potential_component_bound


@pytest.mark.parametrize("subject,slots,expected,components", [
    ("story", (0, 1, 2, 3), 580, 3516),
    ("concert", (0, 1, 2, 3), 580, 3516),
    ("painting", (0, 2), 290, 1758),
])
def test_full_pass_count_and_component_budget(subject, slots, expected, components):
    schedule = potential_schedule(subject, slots)
    assert len(schedule) == expected
    assert potential_component_bound(schedule) == components
    counts = Counter(item["kind"] for item in schedule)
    assert counts == {"clean": len(slots) * 8, "passthrough": len(slots) * 8,
                      "self": len(slots) * 24, "full_swap": len(slots) * 8,
                      "matched": len(slots) * 80, "natural_final": len(slots) * 8,
                      "natural_span": len(slots) * 8, "repeat": len(slots)}


def test_one_group_order_and_reciprocal_donors():
    schedule = potential_schedule("painting", (2,))
    assert all(item["unit"] == 2 and item["subject"] == "painting" for item in schedule)
    first = schedule[:4]
    assert [(item["kind"], item["receiver"]) for item in first] == [
        ("clean", "P1"), ("clean", "N1"), ("clean", "P2"), ("clean", "N2")]
    actions = schedule[4:21]
    assert [item["kind"] for item in actions] == (
        ["passthrough", "self", "self", "self", "full_swap"] + ["matched"] * 10
        + ["natural_final", "natural_span"])
    assert [item["site"] for item in actions[1:4]] == ["adjective", "subject", "instruction"]
    assert [(item["dose"], item["action"]) for item in actions[5:15]] == [
        (dose, arm) for dose in ("low", "high") for arm in (
            "opposite_adjective", "same_adjective", "opposite_subject",
            "opposite_instruction", "orthogonal")]
    assert actions[4]["donor"] == "N1"
    assert actions[5]["donor"] == "N1"
    assert actions[6]["donor"] == "P2"
    assert actions[-1]["donor"] == "N1"
    assert schedule[72]["kind"] == "repeat" and schedule[72]["instruction"] == 0
    assert schedule[73]["kind"] == "clean" and schedule[73]["instruction"] == 1


def test_missing_slots_remain_uncompacted_and_no_outputs_are_scheduled_for_none():
    schedule = potential_schedule("story", (0, 3))
    assert len(schedule) == 290
    assert {item["unit"] for item in schedule} == {0, 3}
    assert potential_schedule("painting", ()) == ()


@pytest.mark.parametrize("subject,slots", [
    ("painting", (1,)), ("painting", (3,)), ("story", (3, 0)),
    ("story", (0, 0)), ("story", (True,)), ("other", (0,)),
])
def test_invalid_or_renumbered_slots_are_rejected(subject, slots):
    with pytest.raises(ValueError):
        potential_schedule(subject, slots)
