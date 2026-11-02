"""Potential action order for the frozen multi-subtoken protocol.

This is a model-free schedule, not an admission, execution, or custody gate.
The caller retains original slot IDs and records individually skipped actions
when a prespecified finite landing or group is unavailable.
"""

RECEIVERS = ("P1", "N1", "P2", "N2")
OPPOSITE = {"P1": "N1", "N1": "P1", "P2": "N2", "N2": "P2"}
SAME = {"P1": "P2", "P2": "P1", "N1": "N2", "N2": "N1"}
SELF_SITES = ("adjective", "subject", "instruction")
ARMS = ("opposite_adjective", "same_adjective", "opposite_subject",
        "opposite_instruction", "orthogonal")
SUBJECT_SLOTS = {"story": (0, 1, 2, 3), "concert": (0, 1, 2, 3), "painting": (0, 2)}


def potential_schedule(subject, slots):
    """Return the ordered maximum actions for the supplied original slots.

    This function deliberately does not decide which measured interventions
    are valid. Empty slots are omitted here but remain in the planned
    denominator maintained by the caller.
    """
    if (type(subject) is not str or subject not in SUBJECT_SLOTS
            or type(slots) not in (tuple, list)
            or any(type(slot) is not int or slot not in SUBJECT_SLOTS[subject] for slot in slots)
            or tuple(slots) != tuple(sorted(set(slots)))):
        raise ValueError("invalid subject or original slot order")
    schedule = []
    for unit in slots:
        for instruction in (0, 1):
            def row(kind, receiver, **fields):
                return {"kind": kind, "subject": subject, "unit": unit,
                        "instruction": instruction, "receiver": receiver, **fields}
            schedule.extend(row("clean", receiver) for receiver in RECEIVERS)
            for receiver in RECEIVERS:
                donor = OPPOSITE[receiver]
                schedule.append(row("passthrough", receiver))
                schedule.extend(row("self", receiver, site=site) for site in SELF_SITES)
                schedule.append(row("full_swap", receiver, donor=donor))
                for dose in ("low", "high"):
                    for arm in ARMS:
                        arm_donor = SAME[receiver] if arm == "same_adjective" else donor
                        if arm == "orthogonal":
                            arm_donor = None
                        schedule.append(row("matched", receiver, dose=dose,
                                            action=arm, donor=arm_donor))
                schedule.append(row("natural_final", receiver, donor=donor))
                schedule.append(row("natural_span", receiver, donor=donor))
            if instruction == 0:
                schedule.append(row("repeat", "P1"))
    return tuple(schedule)


def potential_component_bound(schedule):
    """Conservative bound: six decoder calls per output and clean encodes."""
    kinds = {"clean", "passthrough", "self", "full_swap", "matched",
             "natural_final", "natural_span", "repeat"}
    if (type(schedule) is not tuple or any(type(item) is not dict
            or item.get("kind") not in kinds for item in schedule)):
        raise ValueError("expected a potential schedule")
    return 6 * len(schedule) + sum(item["kind"] in ("clean", "repeat") for item in schedule)
