"""Guard a single model parameter row and verify full-state restoration."""

from contextlib import contextmanager
import hashlib

import numpy as np
import torch


class RestorationError(RuntimeError):
    pass


class RestorationUnprobed(RestorationError):
    pass


def buffer_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_buffers(remove_duplicate=False), key=lambda item: item[0]):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("buffers must be strided CPU tensors")
        data = tensor.detach().contiguous().numpy()
        digest.update(repr((name, str(data.dtype), data.shape)).encode("utf-8") + b"\0")
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def state_hash(model, skip=None):
    digest = hashlib.sha256()
    named = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    for name, tensor in sorted(named, key=lambda item: item[0]):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("state tensors must be strided CPU tensors")
        data = tensor.detach().contiguous().numpy()
        digest.update(repr((name, str(data.dtype), data.shape)).encode("utf-8") + b"\0")
        if skip is not None and tensor is skip[0]:
            index = skip[1]
            if data.ndim != 2 or type(index) is not int or not 0 <= index < data.shape[0]:
                raise ValueError("invalid excluded row")
            digest.update(b"excluded-row\0" + str(index).encode("ascii") + b"\0")
            digest.update(data[:index].tobytes(order="C"))
            digest.update(data[index + 1:].tobytes(order="C"))
        else:
            digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def topology(model):
    named = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    return tuple((name, id(tensor), tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                  tuple(tensor.shape), tuple(tensor.stride()))
                 for name, tensor in sorted(named, key=lambda item: item[0]))


@contextmanager
def temporary_row(model, parameter, index, candidate, *, expected_state, expected_probe, restored_probe):
    if (not isinstance(parameter, torch.nn.Parameter) or parameter.device.type != "cpu"
            or parameter.dtype != torch.float32 or parameter.ndim != 2
            or type(index) is not int or not 0 <= index < parameter.shape[0]
            or not any(value is parameter for _, value in model.named_parameters(remove_duplicate=False))
            or not isinstance(candidate, np.ndarray) or candidate.dtype != np.float32
            or candidate.shape != (parameter.shape[1],) or not np.isfinite(candidate).all()
            or not isinstance(expected_probe, bytes) or not expected_probe or not callable(restored_probe)):
        raise ValueError("invalid row mutation")
    if state_hash(model) != expected_state:
        raise ValueError("checkpoint state does not match the expected baseline")
    unchanged_state = state_hash(model, skip=(parameter, index))
    original_topology = topology(model)
    original = parameter[index].detach().clone()
    body_error = None
    try:
        with torch.no_grad():
            parameter[index].copy_(torch.from_numpy(candidate))
        if (parameter[index].detach().numpy().tobytes() != candidate.tobytes()
                or state_hash(model, skip=(parameter, index)) != unchanged_state
                or topology(model) != original_topology):
            raise ValueError("mutation changed unexpected state or failed to install candidate")
        yield
    except BaseException as exc:
        body_error = exc
    finally:
        try:
            with torch.no_grad():
                parameter[index].copy_(original)
            if state_hash(model) != expected_state or topology(model) != original_topology:
                raise RestorationError("full model state or nonpersistent buffer changed")
            if body_error is not None:
                raise RestorationUnprobed("row restored, but no further model probe was run") from body_error
            if restored_probe() != expected_probe:
                raise RestorationError("clean probe logits differ")
            if state_hash(model) != expected_state or topology(model) != original_topology:
                raise RestorationError("clean probe changed model state or alias topology")
        except BaseException as exc:
            if isinstance(exc, RestorationError):
                raise
            raise RestorationError("row restoration or clean probe failed") from exc
