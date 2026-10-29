"""Fabricated discovery records only; no model or held-out execution."""
import functools
import hashlib
import json
import struct
import unittest
from dataclasses import replace

from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY

try:
    from looking_glass.pretrained_number_selection import (
        Action, Boundary, DiscoveryCase, DiscoveryInvalid, Donor, Observation, freeze_discovery,
    )
except ModuleNotFoundError as error:
    if error.name != "looking_glass.pretrained_number_selection":
        raise
    Action = Boundary = DiscoveryCase = DiscoveryInvalid = Donor = Observation = freeze_discovery = None


@functools.lru_cache(maxsize=None)
def vector(first):
    return struct.pack("<768f", first, *([0.0] * 767))


@functools.lru_cache(maxsize=None)
def logits(margin):
    value = bytearray(50257 * 4)
    struct.pack_into("<f", value, 318 * 4, margin)
    return bytes(value)


def input_hash(ids):
    return hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(ids)]) +
                          b"".join(value.to_bytes(4, "big") for value in ids)).hexdigest()


def fabricated_cases(changed=True):
    """Independent hand-sized effects; no model, private site, or generated split."""
    rows = []
    serial = 0
    for index in range(8):
        ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
        trace = tuple(Boundary(
            module_path=path, capture_order=order, invocation=0, batch_index=0,
            token_index=len(ids) - 1, output_selector=selector,
            output_shape=(1, len(ids), 768), selected_shape=(768,),
            dtype="torch.float32", device="cpu")
            for order, (path, selector) in enumerate(
                [(f"transformer.h.{layer}.{kind}", (0,) if kind == "attn" else ())
                 for layer in range(6) for kind in ("attn", "mlp")] +
                [("transformer.h.5", (0,))]))
        zero = vector(0.0)
        altered = (zero, zero, vector(0.5), vector(1.0), vector(2.0), zero) if changed else (zero,) * 6

        def obs(condition, *, margin=0.0, mlp=None, layer=None, donor=None):
            nonlocal serial
            serial += 1
            request_id = f"{serial:032x}"
            action = None
            if condition in ("clean_to_injected", "injected_to_clean", "null", "self", "upstream", "downstream"):
                path = "transformer.h.5" if condition == "downstream" else f"transformer.h.{layer}.mlp"
                selector = (0,) if condition == "downstream" else ()
                source = None if condition == "null" else Donor(
                    request_id=request_id if condition == "self" else donor.request_id,
                    input_sha256=input_hash(ids), module_path=path, invocation=0,
                    batch_index=0, token_index=len(ids) - 1, output_selector=selector,
                    selected_shape=(768,), dtype="torch.float32", device="cpu")
                action = Action(operation="null" if condition == "null" else "patch",
                                module_path=path, invocation=0, batch_index=0,
                                token_index=len(ids) - 1, output_selector=selector,
                                output_shape=(1, len(ids), 768), selected_shape=(768,),
                                dtype="torch.float32", device="cpu", donor=source)
            return Observation(request_id=f"{serial:032x}", input_sha256=input_hash(ids),
                               condition=condition, attention=(zero,) * 6,
                               mlp=altered if mlp == "injected" else (zero,) * 6 if mlp is None else mlp,
                               block=zero, logits=logits(margin), trace=trace, action=action)

        clean = obs("clean")
        repeat = obs("clean")
        injected = obs("injected", mlp="injected")
        forward = tuple(obs("clean_to_injected", margin=float(layer),
                            mlp=altered[:layer] + (zero,) + altered[layer + 1:], layer=layer, donor=clean)
                        for layer in range(1, 5))
        reverse = tuple(obs("injected_to_clean", margin=-float(layer),
                            mlp=(zero,) * layer + (altered[layer],) + (zero,) * (5 - layer),
                            layer=layer, donor=injected)
                        for layer in range(1, 5))
        rows.append(DiscoveryCase(index=index, clean=clean, repeat=repeat, injected=injected,
                                  forward=forward, reverse=reverse,
                                  null=obs("null", mlp="injected", layer=1),
                                  self_patch=obs("self", mlp="injected", layer=1),
                                  upstream=obs("upstream", mlp="injected", layer=0, donor=clean),
                                  downstream=obs("downstream", donor=clean), verify_clean=obs("clean")))
    return tuple(rows)


class NumberInjectionSelectionTests(unittest.TestCase):
    def test_votes_magnitude_and_forward_reverse_effects_are_distinct(self):
        # Wrong earliest-difference, reversed patch signs, or omitted averaging
        # must change one of these independently derived literal results.
        self.assertIsNotNone(freeze_discovery, "site-blind selector is not implemented")
        result = json.loads(freeze_discovery(fabricated_cases(), private_controls_verified=True,
                                             execution_device="cpu"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["first_changed"], 2)
        self.assertEqual(result["ranking"], [2, 1, 3, 4])
        self.assertEqual(result["votes"], {"1": 0, "2": 8, "3": 0, "4": 0})
        self.assertEqual(result["mean_l2"], {"1": 0.0, "2": 0.5, "3": 1.0, "4": 2.0})
        self.assertEqual(result["magnitude"], 4)
        self.assertEqual(result["earliest"], 1)
        self.assertEqual(result["predictions"], {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0})
        self.assertEqual(result["cases"][0]["reverse_effects"], [-1.0, -2.0, -3.0, -4.0])
        self.assertEqual(len(result["cases"]), 8)

    def test_zero_clean_margins_are_reported_as_incorrect_without_discarding_cases(self):
        result = json.loads(freeze_discovery(fabricated_cases(), private_controls_verified=True,
                                             execution_device="cpu"))
        self.assertEqual(result["clean_accuracy"], {"correct": 0, "incorrect": 8, "ties": 8})
        self.assertTrue(all(case["clean_correct"] is False for case in result["cases"]))

    def test_no_detectable_difference_abstains_but_keeps_baselines(self):
        result = json.loads(freeze_discovery(fabricated_cases(changed=False),
                                             private_controls_verified=True, execution_device="cpu"))
        self.assertIsNone(result["first_changed"])
        self.assertEqual(result["ranking"], [1, 2, 3, 4])
        self.assertEqual(result["votes"], {"1": 0, "2": 0, "3": 0, "4": 0})
        self.assertEqual(result["magnitude"], 1)
        self.assertEqual(result["random"], 1)
        self.assertTrue(all(case["first_changed"] is None for case in result["cases"]))

    def test_tied_votes_choose_lower_layer(self):
        cases = list(fabricated_cases(changed=False))
        for index, case in enumerate(cases):
            layer = 1 if index < 4 else 2
            altered_mlp = case.injected.mlp[:layer] + (vector(1.0),) + case.injected.mlp[layer + 1:]
            reverse = case.reverse[layer - 1]
            reverse_mlp = reverse.mlp[:layer] + (vector(1.0),) + reverse.mlp[layer + 1:]
            reverse = case.reverse[:layer - 1] + (replace(reverse, mlp=reverse_mlp),) + case.reverse[layer:]
            null, same = case.null, case.self_patch
            if layer == 1:
                null = replace(null, mlp=altered_mlp)
                same = replace(same, mlp=altered_mlp)
            cases[index] = replace(case, injected=replace(case.injected, mlp=altered_mlp),
                                   reverse=reverse, null=null, self_patch=same)
        result = json.loads(freeze_discovery(tuple(cases), private_controls_verified=True, execution_device="cpu"))
        self.assertEqual(result["votes"], {"1": 4, "2": 4, "3": 0, "4": 0})
        self.assertEqual(result["ranking"], [1, 2, 3, 4])
        self.assertEqual(result["first_changed"], 1)
        self.assertEqual(result["magnitude"], 1)

    def test_frozen_bytes_bind_all_observations_without_revealing_a_site_field(self):
        cases = fabricated_cases()
        first = freeze_discovery(cases, private_controls_verified=True, execution_device="cpu")
        self.assertEqual(first, freeze_discovery(cases, private_controls_verified=True, execution_device="cpu"))
        result = json.loads(first)
        self.assertEqual(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode(), first)
        self.assertNotIn("site", result)
        entries = result["cases"][0]["observations"]
        self.assertEqual(len(entries), 16)
        self.assertEqual(entries[0]["logits_sha256"], hashlib.sha256(cases[0].clean.logits).hexdigest())
        self.assertEqual(entries[2]["mlp_sha256"][2], hashlib.sha256(cases[0].injected.mlp[2]).hexdigest())
        self.assertEqual(entries[3]["action"]["donor"]["request_id"], cases[0].clean.request_id)

    def test_every_observation_reports_both_contrasts_from_raw_logits(self):
        cases = fabricated_cases()
        case = cases[0]
        raw = bytearray(case.forward[0].logits)
        struct.pack_into("<f", raw, 373 * 4, 2.5)
        struct.pack_into("<f", raw, 547 * 4, -1.0)
        forward = (replace(case.forward[0], logits=bytes(raw)),) + case.forward[1:]
        result = json.loads(freeze_discovery((replace(case, forward=forward),) + cases[1:],
                                             private_controls_verified=True, execution_device="cpu"))
        observation = result["cases"][0]["observations"][3]
        self.assertEqual(observation["role"], "forward_1")
        self.assertEqual(observation["present_margin"], 1.0)
        self.assertEqual(observation["past_margin"], 3.5)

    def test_rejects_wrong_candidate_source_and_payload_landing(self):
        cases = fabricated_cases()
        original = cases[0]
        patch = original.forward[1]
        wrong_layer = replace(patch, action=replace(patch.action, module_path="transformer.h.3.mlp"))
        wrong_donor = replace(patch, action=replace(patch.action,
                              donor=replace(patch.action.donor, request_id=original.injected.request_id)))
        wrong_bits = replace(patch, mlp=patch.mlp[:2] + (vector(2.0),) + patch.mlp[3:])
        for bad in (wrong_layer, wrong_donor, wrong_bits):
            altered = replace(original, forward=original.forward[:1] + (bad,) + original.forward[2:])
            with self.subTest(action=bad.action), self.assertRaises(DiscoveryInvalid):
                freeze_discovery((altered,) + cases[1:], private_controls_verified=True, execution_device="cpu")

    def test_rejects_bad_coordinates_duplicate_requests_and_failed_controls(self):
        cases = fabricated_cases()
        original = cases[0]
        boundary = original.injected.trace[0]
        trace = (replace(boundary, invocation=False),) + original.injected.trace[1:]
        malformed = replace(original.injected, trace=trace)
        duplicate = replace(original.injected, request_id=original.clean.request_id)
        failed_null = replace(original.null, logits=logits(1.0))
        failed_repeat = replace(original.repeat, mlp=(vector(1.0),) + original.repeat.mlp[1:])
        bad_cases = (replace(original, injected=malformed), replace(original, injected=duplicate),
                     replace(original, null=failed_null), replace(original, repeat=failed_repeat))
        for altered in bad_cases:
            with self.subTest(case=altered), self.assertRaises(DiscoveryInvalid):
                freeze_discovery((altered,) + cases[1:], private_controls_verified=True, execution_device="cpu")
        with self.assertRaises(DiscoveryInvalid):
            freeze_discovery(cases[:-1], private_controls_verified=True, execution_device="cpu")
        with self.assertRaises(DiscoveryInvalid):
            freeze_discovery(cases, private_controls_verified=False, execution_device="cpu")

    def test_nonfinite_full_vocabulary_is_rejected_even_outside_contrast(self):
        cases = fabricated_cases()
        raw = bytearray(cases[0].injected.logits)
        struct.pack_into("<I", raw, 50000 * 4, 0x7fc00000)
        altered = replace(cases[0], injected=replace(cases[0].injected, logits=bytes(raw)))
        with self.assertRaises(DiscoveryInvalid):
            freeze_discovery((altered,) + cases[1:], private_controls_verified=True, execution_device="cpu")

    def test_subnormal_difference_counts_in_norm_even_below_vote_tolerance(self):
        cases = fabricated_cases(changed=False)
        case = cases[0]
        tiny = struct.pack("<I", 1) + bytes(767 * 4)
        injected = replace(case.injected, mlp=case.injected.mlp[:2] + (tiny,) + case.injected.mlp[3:])
        reverse = case.reverse[1]
        reverse = replace(reverse, mlp=reverse.mlp[:2] + (tiny,) + reverse.mlp[3:])
        altered = replace(case, injected=injected, reverse=case.reverse[:1] + (reverse,) + case.reverse[2:])
        result = json.loads(freeze_discovery((altered,) + cases[1:],
                                             private_controls_verified=True, execution_device="cpu"))
        self.assertIsNone(result["first_changed"])
        self.assertEqual(result["mean_l2"]["2"], 2.0 ** -152)
        self.assertEqual(result["magnitude"], 2)


if __name__ == "__main__":
    unittest.main()
