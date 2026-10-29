"""Synthetic CPU hook oracle for the site-blind injection executor.

Uses no pretrained weights, real cases, held-out inputs, or private study site.
"""
import hashlib
import json
import struct
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from looking_glass.pretrained_number_injection import NumberInjectionExecutor


WIDTH, VOCAB = 768, 50257
IDS = (7, 13, 7, 19)


class Attention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        return (hidden * ((self.layer + 2) / 32) + (self.layer + 1) / 128,)


class MLP(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        return hidden * ((self.layer + 1) / 32) - (self.layer + 1) / 256


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attn, self.mlp = Attention(layer), MLP(layer)

    def forward(self, hidden):
        hidden = hidden + self.attn(hidden)[0]
        return (hidden + self.mlp(hidden),)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList(Block(i) for i in range(6))

    def forward(self, hidden):
        for block in self.h:
            hidden = block(hidden)[0]
        return hidden


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Backbone()
        self.config = SimpleNamespace(n_layer=6, n_embd=WIDTH, vocab_size=VOCAB,
                                      use_cache=False, _attn_implementation="eager")
        self.is_gradient_checkpointing = False
        self.register_buffer("indices", (torch.arange(VOCAB, dtype=torch.int64, device="cpu") * 13) % WIDTH)
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        assert not use_cache and bool(torch.all(attention_mask == 1))
        features = torch.arange(WIDTH, dtype=torch.float32, device="cpu").reshape(1, 1, WIDTH) / 2048
        hidden = input_ids.to(torch.float32).unsqueeze(-1) / 16 + features
        hidden = self.transformer(hidden)
        return SimpleNamespace(logits=hidden[:, :, self.indices])


def raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def vector(payload):
    return torch.frombuffer(bytearray(payload), dtype=torch.int32).clone().view(torch.float32)


def direction():
    bits = []
    for chunk in range(3):
        digest = hashlib.sha256(b"MLG-v1-direction\0" + (20261028).to_bytes(8, "big") +
                                chunk.to_bytes(4, "big")).digest()
        bits.extend(0x3c93cd3a if byte & (1 << bit) else 0xbc93cd3a
                    for byte in digest for bit in range(7, -1, -1))
    return vector(struct.pack("<768I", *bits))


def reference(site, condition, layer=None, donor=None):
    """Use plain PyTorch hooks, without ActivationPlan or its metadata."""
    model = Model()
    delta = direction()
    injected = condition in {"injected", "clean_to_injected", "upstream", "downstream", "null", "self"}
    patch_layer = (0 if condition == "upstream" else layer) if condition in {
        "upstream", "clean_to_injected", "injected_to_clean"} else None
    captured, counts, witnesses, handles = {}, {}, {}, []
    token = len(IDS) - 1

    for current, block in enumerate(model.transformer.h):
        def attention(_m, _args, output, current=current):
            key = f"attention_{current}"
            counts[key] = counts.get(key, 0) + 1
            assert type(output) is tuple
            captured[key] = raw(output[0][0, token])

        def mlp(_m, _args, output, current=current):
            key = f"mlp_{current}"
            counts[key] = counts.get(key, 0) + 1
            assert type(output) is torch.Tensor
            value = output
            if injected and current == site:
                before = raw(value)
                value = value.clone()
                value[0, token] = value[0, token] + delta
                witnesses["add"] = (f"transformer.h.{current}.mlp", before, raw(value), None)
            action = patch_layer == current or (condition in {"null", "self"} and current == 1)
            if action:
                before = raw(value)
                source = raw(value[0, token]) if condition == "self" else donor
                if condition != "null":
                    value = value.clone()
                    value[0, token] = vector(source)
                witnesses["null" if condition == "null" else "patch"] = (
                    f"transformer.h.{current}.mlp", before, raw(value), source)
            captured[key] = raw(value[0, token])
            return value

        handles.extend((block.attn.register_forward_hook(attention), block.mlp.register_forward_hook(mlp)))

    def final_block(_m, _args, output):
        counts["block_5"] = counts.get("block_5", 0) + 1
        assert type(output) is tuple
        value = output[0]
        if condition == "downstream":
            before = raw(value)
            value = value.clone()
            value[0, token] = vector(donor)
            witnesses["patch"] = ("transformer.h.5", before, raw(value), donor)
        captured["block_5"] = raw(value[0, token])
        return (value,) + output[1:]

    handles.append(model.transformer.h[5].register_forward_hook(final_block))
    try:
        with torch.inference_mode():
            ids = torch.tensor([IDS], dtype=torch.int64, device="cpu")
            output = model(ids, torch.ones_like(ids))
            logits = raw(output.logits[0, -1])
    finally:
        for handle in handles:
            handle.remove()
    assert len(counts) == 13 and set(counts.values()) == {1}
    return captured, witnesses, logits


class InjectionCoordinateOracle(unittest.TestCase):
    def test_all_hidden_candidate_orders_and_controls(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        tiny = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
        was_flushed = int((tiny * 1.0).view(torch.int32).item()) == 0
        if not torch.set_flush_denormal(False):
            self.skipTest("CPU denormal mode cannot be configured")
        self.addCleanup(torch.set_flush_denormal, was_flushed)
        token = len(IDS) - 1
        expected_paths = [name for layer in range(6)
                          for name in (f"transformer.h.{layer}.attn", f"transformer.h.{layer}.mlp")]
        expected_paths.append("transformer.h.5")
        action_signatures = {}

        for site in range(1, 5):
            with self.subTest(synthetic_hidden_layer=site):
                model = Model()
                with NumberInjectionExecutor(model, site=site) as executor:
                    seq = 0

                    def run(condition, *, layer=None, donor_obj=None, donor_data=None, donor_request=None):
                        nonlocal seq
                        seq += 1
                        request_id = f"{seq:032x}"
                        expected, witnesses, expected_logits = reference(site, condition, layer, donor_data)
                        observation = executor.forward(IDS, request_id=request_id, condition=condition,
                                                       layer=layer, donor=donor_obj)
                        self.assertEqual(observation.request_id, request_id)
                        self.assertEqual(observation.condition, condition)
                        expected_input_sha = hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(IDS)]) +
                            b"".join(value.to_bytes(4, "big") for value in IDS)).hexdigest()
                        self.assertEqual(observation.input_sha256, expected_input_sha)
                        for i in range(6):
                            self.assertEqual(observation.attention[i], expected[f"attention_{i}"])
                            self.assertEqual(observation.mlp[i], expected[f"mlp_{i}"])
                        self.assertEqual((observation.block, observation.logits),
                                         (expected["block_5"], expected_logits))
                        self.assertEqual([entry.module_path for entry in observation.trace], expected_paths)
                        for index, boundary in enumerate(observation.trace):
                            model.get_submodule(boundary.module_path)
                            selector = (0,) if boundary.module_path.endswith(".attn") or index == 12 else ()
                            self.assertEqual((boundary.capture_order, boundary.invocation, boundary.batch_index,
                                              boundary.token_index, boundary.output_selector, boundary.output_shape,
                                              boundary.selected_shape, boundary.dtype, boundary.device),
                                             (index, 0, 0, token, selector, (1, len(IDS), WIDTH),
                                              (WIDTH,), "torch.float32", "cpu"))
                        evidence = [item for item in executor.private_evidence() if item.request_id == request_id]
                        self.assertEqual({item.operation for item in evidence}, set(witnesses))
                        requested_action = witnesses.get("patch") or witnesses.get("null")
                        if requested_action is None:
                            self.assertIsNone(observation.action)
                        else:
                            path = requested_action[0]
                            operation = "null" if condition == "null" else "patch"
                            selector = (0,) if path == "transformer.h.5" else ()
                            action = observation.action
                            self.assertEqual((action.operation, action.module_path, action.invocation,
                                              action.batch_index, action.token_index, action.output_selector,
                                              action.output_shape, action.selected_shape, action.dtype, action.device),
                                             (operation, path, 0, 0, token, selector, (1, len(IDS), WIDTH), (WIDTH,),
                                              "torch.float32", "cpu"))
                            private_action = next(item for item in evidence if item.operation == operation)
                            self.assertEqual(action.donor_identity, private_action.donor_identity)
                        signature_key = (condition, layer)
                        signature = observation.action
                        if signature_key in action_signatures:
                            self.assertEqual(signature, action_signatures[signature_key])
                        else:
                            action_signatures[signature_key] = signature
                        for item in evidence:
                            path, before, after, source = witnesses[item.operation]
                            model.get_submodule(path)
                            self.assertEqual((item.input_sha256, item.condition, item.module_path, item.token_index,
                                              item.before, item.after, item.donor),
                                             (observation.input_sha256, condition, path, token, before, after, source))
                            start, end = token * WIDTH * 4, (token + 1) * WIDTH * 4
                            self.assertEqual((before[:start], before[end:]), (after[:start], after[end:]))
                            if item.operation == "patch":
                                identity = item.donor_identity
                                model.get_submodule(identity.module_path)
                                donor_id = request_id if condition == "self" else donor_request
                                selector = (0,) if path == "transformer.h.5" else ()
                                self.assertEqual((identity.request_id, identity.input_sha256, identity.module_path,
                                                  identity.invocation, identity.batch_index, identity.token_index,
                                                  identity.output_selector, identity.selected_shape, identity.dtype,
                                                  identity.device),
                                                 (donor_id, expected_input_sha, path, 0, 0, token, selector,
                                                  (WIDTH,), "torch.float32", "cpu"))
                            else:
                                self.assertIsNone(item.donor_identity)
                            proof = json.loads(item.metadata_json)
                            self.assertEqual(len(proof), 3)
                            self.assertEqual([x["module_path"] for x in proof], [path] * 3)
                            self.assertEqual([x["operation"] for x in proof], ["capture", item.operation, "capture"])
                            selector = [0] if path == "transformer.h.5" else []
                            for i, event in enumerate(proof):
                                self.assertEqual((event["invocation"], event["forward_index"], event["output_selector"],
                                                  event["output_shape"], event["dtype"], event["device"]),
                                                 (0, 0, selector, [1, len(IDS), WIDTH], "torch.float32", "cpu"))
                                self.assertEqual((event["batch_index"], event["token_index"], event["selected_shape"]),
                                                 (0, token, [WIDTH]) if i == 1 else
                                                 (None, None, [1, len(IDS), WIDTH]))
                        return observation, expected

                    clean, clean_ref = run("clean")
                    injected, injected_ref = run("injected")
                    for layer in range(site):
                        self.assertEqual(clean_ref[f"mlp_{layer}"], injected_ref[f"mlp_{layer}"])
                    self.assertNotEqual(clean_ref[f"mlp_{site}"], injected_ref[f"mlp_{site}"])
                    for layer in range(1, 5):
                        restored, _ = run("clean_to_injected", layer=layer, donor_obj=clean.mlp_donor(layer),
                                          donor_data=clean_ref[f"mlp_{layer}"], donor_request=clean.request_id)
                        reverse, _ = run("injected_to_clean", layer=layer, donor_obj=injected.mlp_donor(layer),
                                         donor_data=injected_ref[f"mlp_{layer}"], donor_request=injected.request_id)
                        if layer == site:
                            self.assertEqual(restored.logits, clean.logits)
                            self.assertEqual(reverse.logits, injected.logits)
                    upstream, _ = run("upstream", donor_obj=clean.mlp_donor(0), donor_data=clean_ref["mlp_0"],
                                      donor_request=clean.request_id)
                    downstream, _ = run("downstream", donor_obj=clean.block_donor(), donor_data=clean_ref["block_5"],
                                        donor_request=clean.request_id)
                    null, _ = run("null")
                    same, _ = run("self")
                    self.assertEqual(upstream.logits, injected.logits)
                    self.assertEqual(downstream.logits, clean.logits)
                    self.assertEqual(null.logits, injected.logits)
                    self.assertEqual(same.logits, injected.logits)
                    self.assertEqual(len(executor.completed()), 14)
                self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                    for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
