"""CPU integration checks for the bounded, uncached T5 trace adapter."""

import importlib.util

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration


def tiny_t5():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=2,
                          num_decoder_layers=2, num_heads=2, dropout_rate=0.0,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        return T5ForConditionalGeneration(config).eval()


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_encoder_and_decoder_block_tokens_match_untraced_hidden_states():
    # A decoder-token/encoder-token mixup or a wrong tuple selector must fail.
    assert importlib.util.find_spec("looking_glass.flan_t5_trace") is not None
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    source = torch.tensor([[4, 5, 1, 0]], dtype=torch.long)
    source_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    decoder = torch.tensor([[0, 7, 1]], dtype=torch.long)
    decoder_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    with torch.inference_mode():
        baseline = model(input_ids=source, attention_mask=source_mask,
                         decoder_input_ids=decoder, decoder_attention_mask=decoder_mask,
                         use_cache=False, output_hidden_states=True, return_dict=True)
    output, records = trace_t5_forward(
        model, input_ids=source, attention_mask=source_mask, decoder_input_ids=decoder,
        decoder_attention_mask=decoder_mask, run_id="tiny",
        points=[T5TracePoint("encoder", 0, "block", 0, 1),
                T5TracePoint("decoder", 0, "block", 0, 2)])
    assert len(records) == 2
    torch.testing.assert_close(output.logits, baseline.logits, rtol=0, atol=0)
    torch.testing.assert_close(records[0].value, baseline.encoder_hidden_states[1][0, 1], rtol=0, atol=0)
    torch.testing.assert_close(records[1].value, baseline.decoder_hidden_states[1][0, 2], rtol=0, atol=0)
    assert [r.event.module_path for r in records] == ["encoder.block.0", "decoder.block.0"]
    assert [r.event.shape for r in records] == [(1, 4, 16), (1, 3, 16)]
    assert [r.event.token_positions for r in records] == [(1,), (2,)]
    assert all(r.event.output_path == (0,) and r.event.invocation == 0 for r in records)
    assert all(r.event.generation.stage == "forward" and r.event.generation.use_cache is False
               and r.event.generation.cache_position is None for r in records)


def test_attention_and_feed_forward_boundaries_keep_stack_and_selector_identity():
    # T5 self-attention wrappers are tuple outputs; FF wrappers return tensors.
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    source = torch.tensor([[4, 5, 1, 0]], dtype=torch.long)
    source_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    decoder = torch.tensor([[0, 7, 1]], dtype=torch.long)
    decoder_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    paths = ("encoder.block.0.layer.0", "encoder.block.0.layer.1",
             "decoder.block.0.layer.0", "decoder.block.0.layer.2")
    observed = {}
    handles = []
    for path in paths:
        def save(_module, _args, output, *, path=path):
            observed[path] = (output[0] if type(output) is tuple else output).detach().clone()
        handles.append(model.get_submodule(path).register_forward_hook(save))
    try:
        with torch.inference_mode():
            baseline = model(input_ids=source, attention_mask=source_mask, decoder_input_ids=decoder,
                             decoder_attention_mask=decoder_mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    output, records = trace_t5_forward(model, input_ids=source, attention_mask=source_mask,
        decoder_input_ids=decoder, decoder_attention_mask=decoder_mask, run_id="boundaries",
        points=[T5TracePoint("decoder", 0, "feed_forward", 0, 2),
                T5TracePoint("encoder", 0, "feed_forward", 0, 1),
                T5TracePoint("decoder", 0, "self_attention", 0, 2),
                T5TracePoint("encoder", 0, "self_attention", 0, 1)])
    assert [r.event.module_path for r in records] == list(paths)
    assert [r.event.output_path for r in records] == [(0,), (), (0,), ()]
    assert [r.event.module_kind for r in records] == [
        "t5_encoder_self_attention_residual", "t5_encoder_feed_forward_residual",
        "t5_decoder_self_attention_residual", "t5_decoder_feed_forward_residual"]
    for record, path, token in zip(records, paths, (1, 1, 2, 2), strict=True):
        torch.testing.assert_close(record.value, observed[path][0, token], rtol=0, atol=0)
    torch.testing.assert_close(output.logits, baseline.logits, rtol=0, atol=0)


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
def test_non_eager_stack_configuration_is_rejected(stack):
    # A stack-local backend change must not silently be described as eager.
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    getattr(model, stack).config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        trace_t5_forward(model, input_ids=torch.tensor([[4, 1]]), attention_mask=torch.tensor([[1, 1]]),
            decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 1]]),
            run_id="backend", points=[T5TracePoint("encoder", 0, "block", 0, 0)])


def test_masks_and_unsupported_generation_or_cross_attention_are_rejected():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    kwargs = dict(input_ids=torch.tensor([[4, 1, 0]]), attention_mask=torch.tensor([[1, 1, 0]]),
                  decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 0]]),
                  run_id="masked")
    for point in (T5TracePoint("encoder", 0, "block", 0, 2),
                  T5TracePoint("decoder", 0, "block", 0, 1)):
        with pytest.raises(ValueError, match="masked token"):
            trace_t5_forward(model, points=[point], **kwargs)
    with pytest.raises(ValueError, match="unsupported T5 boundary"):
        T5TracePoint("decoder", 0, "cross_attention", 0, 0)
    with pytest.raises(ValueError, match="first invocation"):
        T5TracePoint("decoder", 0, "block", 0, 0, invocation=1)
    with pytest.raises(TypeError):
        trace_t5_forward(model, points=[T5TracePoint("encoder", 0, "block", 0, 0)],
                         past_key_values=object(), **kwargs)


def test_failed_capture_budget_cleans_hooks_and_allows_a_later_forward():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    kwargs = dict(input_ids=torch.tensor([[4, 1]]), attention_mask=torch.tensor([[1, 1]]),
                  decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 1]]),
                  run_id="budget", points=[T5TracePoint("encoder", 0, "block", 0, 0)])
    with pytest.raises(ValueError, match="captures exceed"):
        trace_t5_forward(model, max_trace_bytes=1, **kwargs)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    output, records = trace_t5_forward(model, **kwargs)
    assert output.logits.shape == (1, 2, 32)
    assert records[0].event.saved_shape == (16,)


def test_input_and_logit_budgets_reject_oversized_requests():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    kwargs = dict(input_ids=torch.tensor([[4, 1]]), attention_mask=torch.tensor([[1, 1]]),
                  decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 1]]),
                  run_id="limits", points=[T5TracePoint("encoder", 0, "block", 0, 0)])
    with pytest.raises(ValueError, match="input token budget"):
        trace_t5_forward(model, max_input_tokens=3, **kwargs)
    with pytest.raises(ValueError, match="logit output budget"):
        trace_t5_forward(model, max_logits_bytes=255, **kwargs)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_autocast_is_rejected_before_an_uncached_float32_trace():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        trace_t5_forward(model, input_ids=torch.tensor([[4, 1]]), attention_mask=torch.tensor([[1, 1]]),
            decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 1]]),
            run_id="autocast", points=[T5TracePoint("encoder", 0, "block", 0, 0)])


def test_mixed_parameter_dtypes_are_rejected_before_forward():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    layer = model.encoder.block[0].layer[1].DenseReluDense.wi
    layer.weight.data = layer.weight.data.to(torch.float16)
    with pytest.raises(ValueError, match="parameters"):
        trace_t5_forward(model, input_ids=torch.tensor([[4, 1]]), attention_mask=torch.tensor([[1, 1]]),
            decoder_input_ids=torch.tensor([[0, 1]]), decoder_attention_mask=torch.tensor([[1, 1]]),
            run_id="mixed", points=[T5TracePoint("encoder", 0, "block", 0, 0)])


def test_nested_boundaries_execute_in_model_order_with_uncached_metadata():
    from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

    model = tiny_t5()
    model.config.use_cache = model.decoder.config.use_cache = True
    output, records = trace_t5_forward(model, input_ids=torch.tensor([[4, 1, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]), decoder_input_ids=torch.tensor([[0, 1]]),
        decoder_attention_mask=torch.tensor([[1, 1]]), run_id="ordered", points=[
            T5TracePoint("decoder", 0, "block", 0, 1),
            T5TracePoint("encoder", 0, "block", 0, 1),
            T5TracePoint("decoder", 0, "self_attention", 0, 1),
            T5TracePoint("encoder", 0, "feed_forward", 0, 1)])
    assert [r.event.module_path for r in records] == ["encoder.block.0.layer.1", "encoder.block.0",
        "decoder.block.0.layer.0", "decoder.block.0"]
    assert [r.event.sequence for r in records] == [0, 1, 2, 3]
    assert output.past_key_values is None
    assert all(r.event.generation == records[0].event.generation for r in records)
    assert records[0].event.generation.to_dict() == {"stage": "forward", "use_cache": False,
        "generation_step": None, "cache_position": None, "cache_length_before": None,
        "cache_length_after": None}
