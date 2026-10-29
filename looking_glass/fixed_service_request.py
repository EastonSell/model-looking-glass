"""Pre-allocation validation for the fixed two-token service IPC factory.

Only trusted owner construction supplies the downstream tensor factory. This
module does not independently admit a workload or authenticate a client.
"""
import re


_IDENTIFIER = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z', re.ASCII)


class FixedServiceRequestFactory:
    """Reject out-of-profile wire input before the tensor factory is called."""

    def __init__(self, tensor_factory):
        if not callable(tensor_factory):
            raise TypeError('a trusted tensor request factory is required')
        self._tensor_factory = tensor_factory

    def __call__(self, request_id, inputs, seed):
        from .ipc import IPCInputRejected

        # Only exact builtins are inspected, without constructing a tensor,
        # taking an alias for retention, or invoking client-controlled code.
        if (type(request_id) is not str or _IDENTIFIER.fullmatch(request_id) is None
                or type(inputs) is not dict or len(inputs) != 1):
            raise IPCInputRejected('invalid fixed-profile request')
        key = next(iter(inputs))
        if type(key) is not str or key != 'input_ids':
            raise IPCInputRejected('invalid fixed-profile request')
        rows = inputs[key]
        if (type(rows) is not list or len(rows) != 1 or type(rows[0]) is not list
                or len(rows[0]) != 2 or type(rows[0][0]) is not int
                or type(rows[0][1]) is not int or rows[0][0] != 15496 or rows[0][1] != 995
                or type(seed) is not int or seed not in (29, 43, 71)):
            raise IPCInputRejected('invalid fixed-profile request')
        # The tensor factory receives a service-owned canonical value, never
        # the caller's mutable list or dict after the validation boundary.
        canonical_inputs = {'input_ids': [[15496, 995]]}
        try:
            return self._tensor_factory(request_id, canonical_inputs, seed)
        except IPCInputRejected:
            # A tensor-producing factory cannot certify that it retained no
            # allocation or alias when it raises. Keep that path uncertain.
            raise RuntimeError('tensor factory failed after pre-validation') from None
