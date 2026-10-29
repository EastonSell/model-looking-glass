"""Explicit site-blind mapping from executor observations to selector values.

This is an in-process data adapter, not a custody, publication, or execution
authorization boundary. The caller must perform and preserve the protocol's
private controls separately before asserting them to the selector.
"""
from .pretrained_number_injection import DonorIdentity, InjectionObservation, PublicAction, PublicTraceBoundary
from .pretrained_number_selection import Action, Boundary, DiscoveryCase, Donor, Observation


def _boundary(value):
    if type(value) is not PublicTraceBoundary:
        raise TypeError("an executor public trace boundary is required")
    return Boundary(value.module_path, value.capture_order, value.invocation, value.batch_index,
                    value.token_index, value.output_selector, value.output_shape, value.selected_shape,
                    value.dtype, value.device)


def _donor(value):
    if value is None:
        return None
    if type(value) is not DonorIdentity:
        raise TypeError("an executor public donor identity is required")
    return Donor(value.request_id, value.input_sha256, value.module_path, value.invocation,
                 value.batch_index, value.token_index, value.output_selector, value.selected_shape,
                 value.dtype, value.device)


def _action(value):
    if value is None:
        return None
    if type(value) is not PublicAction:
        raise TypeError("an executor public action is required")
    return Action(value.operation, value.module_path, value.invocation, value.batch_index,
                  value.token_index, value.output_selector, value.output_shape, value.selected_shape,
                  value.dtype, value.device, _donor(value.donor_identity))


def to_selector_observation(value):
    """Copy only allowed public fields; keep immutable payload bytes by reference.

    The selector validates the resulting typed coordinates, condition, donor,
    payload sizes and values. This adapter must run inside the trusted
    custodian: it neither accepts nor exports private action evidence.
    """
    if type(value) is not InjectionObservation or type(value.trace) is not tuple:
        raise TypeError("an executor public observation is required")
    return Observation(value.request_id, value.input_sha256, value.condition, value.attention,
                       value.mlp, value.block, value.logits, tuple(_boundary(item) for item in value.trace),
                       _action(value.action))


def assemble_discovery_case(*, index, clean, repeat, injected, forward, reverse,
                            null, self_patch, upstream, downstream, verify_clean):
    """Build one selector case without accepting an injected-site identity."""
    if (type(index) is not int or not 0 <= index < 8 or type(forward) is not tuple or len(forward) != 4
            or type(reverse) is not tuple or len(reverse) != 4):
        raise ValueError("case index or candidate sequence is invalid")
    return DiscoveryCase(index, to_selector_observation(clean), to_selector_observation(repeat),
                         to_selector_observation(injected), tuple(map(to_selector_observation, forward)),
                         tuple(map(to_selector_observation, reverse)), to_selector_observation(null),
                         to_selector_observation(self_patch), to_selector_observation(upstream),
                         to_selector_observation(downstream), to_selector_observation(verify_clean))
