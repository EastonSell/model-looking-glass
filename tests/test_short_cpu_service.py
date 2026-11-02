"""CPU-only short-input service checks using a tiny untrained GPT-2."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.ipc import IPCClient, IPCInputRejected
from looking_glass.service import OwnerStatus


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model():
    config = GPT2Config(vocab_size=41, n_positions=32, n_embd=16, n_layer=2,
                        n_head=2, use_cache=False, eos_token_id=40,
                        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(207)
        return GPT2LMHeadModel(config).eval()


@pytest.mark.parametrize("bucket,lengths", [(16, (1, 15)), (16, (9, 16)),
                                             (32, (1, 31)), (32, (17, 32))])
def test_same_bucket_clients_get_last_active_token_logits_matching_reference(bucket, lengths):
    from looking_glass.short_cpu_service import make_short_input, serve_short_cpu_model

    model = tiny_model()
    rows = [[((i + offset) % 38) + 1 for i in range(length)]
            for length, offset in zip(lengths, (0, 7), strict=True)]
    references = []
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row], dtype=torch.long)
            references.append(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False).logits[0, -1].clone())
    stop, ready = threading.Event(), threading.Event()
    server_box, owner_results = [], []
    def announce(server):
        server_box.append(server)
        ready.set()
    owner = threading.Thread(target=lambda: owner_results.append(serve_short_cpu_model(
        model, model_id="tiny-gpt2", team_ids=("alpha", "beta"), bucket_length=bucket,
        stop_event=stop, announce=announce)), daemon=True)
    owner.start()
    try:
        assert ready.wait(10)
        server = server_box[0]
        barrier = threading.Barrier(2)
        def submit(team, row):
            barrier.wait(timeout=10)
            handle = IPCClient(server.address, server.team_tokens[team], io_timeout=10).submit(
                team_id=team, request_id="job-" + team, seed=17,
                inputs=make_short_input(row, bucket_length=bucket, vocab_size=41, pad_token_id=40))
            return handle.wait(timeout=20)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit, team, row)
                       for team, row in zip(("alpha", "beta"), rows, strict=True)]
            outcomes = [future.result(timeout=30) for future in futures]
        assert not torch.allclose(references[0], references[1], atol=1e-6, rtol=1e-5)
        for team, outcome, reference in zip(("alpha", "beta"), outcomes, references, strict=True):
            assert outcome["status"] == "succeeded" and outcome["team_id"] == team
            assert outcome["request_id"] == "job-" + team
            result = outcome["result"]
            assert result["shape"] == [1, 1, 41] and result["dtype"] == "torch.float32"
            assert torch.allclose(torch.tensor(result["logits"]), reference, atol=1e-6, rtol=1e-5)
    finally:
        stop.set()
        owner.join(15)
    assert not owner.is_alive()
    assert owner_results[0].status is OwnerStatus.CLEAN


def test_request_factory_rejects_invalid_masks_and_shapes_before_tensor_factory():
    from looking_glass.short_cpu_service import ShortRequestFactory, make_short_input

    calls = []
    factory = ShortRequestFactory(lambda *args: calls.append(args), bucket_length=16,
                                  vocab_size=41, pad_token_id=40)
    valid = make_short_input([1, 2, 3], bucket_length=16, vocab_size=41, pad_token_id=40)
    bad_mask = {"input_ids": [valid["input_ids"][0][:]],
                "attention_mask": [valid["attention_mask"][0][:]]}
    bad_mask["attention_mask"][0][4] = 1
    zero_mask = {"input_ids": [[40] * 16], "attention_mask": [[0] * 16]}
    wrong_padding = {"input_ids": [valid["input_ids"][0][:]],
                     "attention_mask": [valid["attention_mask"][0][:]]}
    wrong_padding["input_ids"][0][-1] = 3
    for inputs, seed in ((bad_mask, 0), (zero_mask, 0), (wrong_padding, 0),
                         ({"input_ids": [[1]], "attention_mask": [[1]]}, 0),
                         (valid, True)):
        with pytest.raises(IPCInputRejected):
            factory("request", inputs, seed)
    assert calls == []
    factory("request", valid, 0)
    assert len(calls) == 1 and calls[0][1] == valid and calls[0][1] is not valid


def test_preallocation_helper_rejects_truncation_and_invalid_ids():
    from looking_glass.short_cpu_service import make_short_input
    for values in ([], [True], [41], [-1], list(range(17))):
        with pytest.raises(ValueError):
            make_short_input(values, bucket_length=16, vocab_size=41, pad_token_id=40)


def test_helper_output_uses_the_validated_copy_even_if_caller_mutates():
    import inspect
    import sys
    from looking_glass.short_cpu_service import make_short_input
    original = [1, 2, 3]
    lines, start = inspect.getsourcelines(make_short_input)
    count_line = next(start + index for index, line in enumerate(lines)
                      if line.strip().startswith("count = len("))
    previous_trace = sys.gettrace()
    def mutate(frame, event, _arg):
        if frame.f_code is make_short_input.__code__ and event == "line" and frame.f_lineno == count_line:
            original[0] = True
        return mutate
    try:
        sys.settrace(mutate)
        result = make_short_input(original, bucket_length=16, vocab_size=41, pad_token_id=40)
    finally:
        sys.settrace(previous_trace)
    assert original[0] is True
    assert result["input_ids"][0][:3] == [1, 2, 3]


def test_factory_copies_validated_input_and_downgrades_late_clean_rejection():
    from looking_glass.short_cpu_service import ShortRequestFactory, make_short_input
    original = make_short_input([1, 2], bucket_length=16, vocab_size=41, pad_token_id=40)
    received = []
    def consume(_request_id, inputs, _seed):
        original["input_ids"][0][0] = 39
        received.append(inputs)
    factory = ShortRequestFactory(consume, bucket_length=16, vocab_size=41, pad_token_id=40)
    factory("valid", original, 7)
    assert received[0]["input_ids"][0][0] == 1
    def rejected(*_args):
        raise IPCInputRejected("late")
    factory = ShortRequestFactory(rejected, bucket_length=16, vocab_size=41, pad_token_id=40)
    with pytest.raises(RuntimeError, match="after pre-validation"):
        factory("valid", original, 7)


def test_factory_validates_a_private_snapshot_before_tensor_allocation(monkeypatch):
    from looking_glass import short_cpu_service as service
    original = service.make_short_input([1, 2], bucket_length=16, vocab_size=41, pad_token_id=40)
    valid_snapshot = [1, 1] + [0] * 14
    validate = service._wire_rows
    received = []
    def mutate_original_after_validation(inputs, *args):
        result = validate(inputs, *args)
        original["attention_mask"][0][4] = 1
        return result
    monkeypatch.setattr(service, "_wire_rows", mutate_original_after_validation)
    factory = service.ShortRequestFactory(lambda _id, inputs, _seed: received.append(inputs),
        bucket_length=16, vocab_size=41, pad_token_id=40)
    factory("valid", original, 7)
    assert received[0]["attention_mask"][0] == valid_snapshot
    assert original["attention_mask"][0][4] == 1


def test_cli_rejects_invalid_teams_before_loading_checkpoint(monkeypatch):
    from looking_glass import cpu_short_service
    loads = []
    monkeypatch.setattr(cpu_short_service, "load_distilgpt2", lambda **kw: loads.append(kw))
    with pytest.raises(SystemExit):
        cpu_short_service.main(["--cache-dir", "unused", "--bucket", "16",
                                "--team", "alpha", "--team", "alpha"])
    assert loads == []


def test_last_token_projection_has_exact_compact_independent_storage():
    from looking_glass.resident_worker import ForwardRequest, ResidentWorker
    from looking_glass.short_cpu_service import ShortLastTokenCPUWorker, make_short_input
    model = tiny_model()
    resident = ResidentWorker("tiny", model, max_batch_size=1, max_sequence_length=16,
                              max_input_bytes=256, max_output_bytes=16 * 41 * 4)
    worker = ShortLastTokenCPUWorker(resident, bucket_length=16, vocab_size=41, pad_token_id=40)
    values = make_short_input([1, 2, 3], bucket_length=16, vocab_size=41, pad_token_id=40)
    request = ForwardRequest("request", {name: torch.tensor(value) for name, value in values.items()}, 0)
    try:
        result = worker.execute(request)
        assert result.logits.shape == (1, 1, 41)
        assert result.logits.untyped_storage().nbytes() == 41 * 4
        assert result.logits.storage_offset() == 0 and not result.logits.is_inference()
    finally:
        worker.close()


def test_nonfinite_selected_logits_fail_worker_instead_of_returning_a_result():
    from types import SimpleNamespace
    from looking_glass.resident_worker import ForwardRequest, ResidentWorker, WorkerUnavailable
    from looking_glass.short_cpu_service import ShortLastTokenCPUWorker, make_short_input
    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
        def forward(self, input_ids, attention_mask):
            result = torch.zeros((1, 16, 41))
            result[0, 2, 0] = float("nan")
            return {"logits": result}
    resident = ResidentWorker("bad", BadModel().eval(), max_batch_size=1, max_sequence_length=16,
                              max_input_bytes=256, max_output_bytes=16 * 41 * 4)
    worker = ShortLastTokenCPUWorker(resident, bucket_length=16, vocab_size=41, pad_token_id=40)
    values = make_short_input([1, 2, 3], bucket_length=16, vocab_size=41, pad_token_id=40)
    request = ForwardRequest("request", {name: torch.tensor(value) for name, value in values.items()}, 0)
    try:
        with pytest.raises(RuntimeError, match="not finite"):
            worker.execute(request)
        assert worker.failed and worker.poisoned
        with pytest.raises(WorkerUnavailable):
            worker.execute(request)
    finally:
        worker.close()


def test_pre_execution_cancellation_does_not_poison_cpu_worker():
    from looking_glass.resident_worker import ForwardRequest, ResidentWorker, RequestCancelled
    from looking_glass.short_cpu_service import ShortLastTokenCPUWorker, make_short_input
    resident = ResidentWorker("tiny", tiny_model(), max_batch_size=1, max_sequence_length=16,
                              max_input_bytes=256, max_output_bytes=16 * 41 * 4)
    worker = ShortLastTokenCPUWorker(resident, bucket_length=16, vocab_size=41, pad_token_id=40)
    values = make_short_input([1, 2, 3], bucket_length=16, vocab_size=41, pad_token_id=40)
    request = ForwardRequest("request", {name: torch.tensor(value) for name, value in values.items()}, 0)
    cancelled = threading.Event()
    cancelled.set()
    try:
        with pytest.raises(RequestCancelled):
            worker.execute(request, cancelled)
        assert not worker.failed
        assert worker.execute(request).logits.shape == (1, 1, 41)
    finally:
        worker.close()
