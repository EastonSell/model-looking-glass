"""Fabricated records only; no model or actual held-out execution."""
import functools
import hashlib
import json
import math
import struct
import unittest
from dataclasses import fields, replace

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


def fabricated_cases(changed=True, ids_rows=None):
    """Independent hand-sized effects; no model, private site, or generated split."""
    rows = []
    serial = 0
    for index in range(8):
        ids = (PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
               if ids_rows is None else ids_rows[index])
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


def fabricated_evaluation():
    # These arbitrary token numbers are invented test data, not protocol inputs.
    from looking_glass.pretrained_number_selection import EvaluationCase
    ids_rows = tuple((1000 + index, 2000 + index, 3000 + index) for index in range(8))
    rows = fabricated_cases(ids_rows=ids_rows)
    return tuple(EvaluationCase(**{field.name: getattr(row, field.name)
                                   for field in fields(DiscoveryCase)},
                                text=f"fabricated case {index}", input_ids=ids_rows[index])
                 for index, row in enumerate(rows))


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


class NumberInjectionEvaluationTests(unittest.TestCase):
    def setUp(self):
        from looking_glass.pretrained_number_selection import score_evaluation
        self.score = score_evaluation
        self.discovery = freeze_discovery(fabricated_cases(), private_controls_verified=True,
                                          execution_device="cpu")
        self.options = {"discovery_bytes": self.discovery,
                        "prediction_sha256": hashlib.sha256(self.discovery).hexdigest(),
                        "prediction_freeze_sha256": "a" * 64,
                        "heldout_commitment_sha256": "b" * 64,
                        "custody_verified": True, "private_controls_verified": True,
                        "execution_device": "cpu"}

    @staticmethod
    def with_logits(case, injected, patches):
        return replace(case, injected=replace(case.injected, logits=injected),
                       null=replace(case.null, logits=injected),
                       self_patch=replace(case.self_patch, logits=injected),
                       upstream=replace(case.upstream, logits=injected),
                       forward=tuple(replace(obs, logits=raw)
                                     for obs, raw in zip(case.forward, patches)))

    def test_frozen_predictions_grouped_errors_and_full_vocabulary_restoration(self):
        # Site 1's effects are -2..5; its discovery prediction is 1.0.
        cases = tuple(self.with_logits(case, logits(2.0),
                                       (logits(float(i)), logits(2.0), logits(3.0), logits(0.0)))
                      for i, case in enumerate(fabricated_evaluation()))
        result = json.loads(self.score(cases, **self.options))
        self.assertEqual(result["selection"]["first_changed"], 2)
        self.assertEqual(result["selection"]["magnitude"], 4)
        self.assertEqual(result["prediction_sha256"], self.options["prediction_sha256"])
        site = result["sites"]["1"]
        self.assertEqual(site["effects"], [-2., -1., 0., 1., 2., 3., 4., 5.])
        self.assertEqual(site["errors"], [-3., -2., -1., 0., 1., 2., 3., 4.])
        self.assertEqual(site["mae"], 2.0)
        self.assertEqual(site["rmse"], math.sqrt(5.5))
        self.assertEqual(site["zero_mae"], 2.25)
        self.assertEqual(site["zero_rmse"], math.sqrt(7.5))
        self.assertEqual([family["errors"] for family in site["families"]],
                         [[-3., -2.], [-1., 0.], [1., 2.], [3., 4.]])
        self.assertEqual(site["restoration"], {"numerator": 140.0, "denominator": 32.0,
                                               "value": -3.375})
        self.assertEqual([result["sites"][str(s)]["restoration"]["value"] for s in range(2, 5)],
                         [0.0, -1.25, 1.0])

    def test_denominator_equal_to_threshold_is_undefined_and_nonzero_is_retained(self):
        # Each term is exactly representable in float32 and the binary64 fsum
        # is exactly 1e-8; odd exponents split into two squares.
        numerator, denominator = (1e-8).as_integer_ratio()
        denominator_power = denominator.bit_length() - 1
        terms = []
        for bit in range(numerator.bit_length()):
            if (numerator >> bit) & 1:
                exponent = bit - denominator_power
                terms.extend([2.0 ** (exponent // 2)] * (1 if exponent % 2 == 0 else 2))
        self.assertEqual(math.fsum(term * term for term in terms), 1e-8)
        raw = bytearray(logits(0.0))
        for index, term in enumerate(terms, start=1000):
            struct.pack_into("<f", raw, index * 4, term)
        cases = list(fabricated_evaluation())
        cases[0] = self.with_logits(cases[0], bytes(raw), (logits(0.0),) * 4)
        for index in range(1, 8):
            cases[index] = self.with_logits(cases[index], logits(0.0), (logits(0.0),) * 4)
        result = json.loads(self.score(tuple(cases), **self.options))
        restoration = result["sites"]["1"]["restoration"]
        self.assertEqual(restoration["denominator"], 1e-8)
        self.assertEqual(restoration["numerator"], 0.0)
        self.assertIsNone(restoration["value"])

    def test_no_site_abstention_is_frozen_and_not_reselected(self):
        discovery = freeze_discovery(fabricated_cases(changed=False),
                                     private_controls_verified=True, execution_device="cpu")
        options = {**self.options, "discovery_bytes": discovery,
                   "prediction_sha256": hashlib.sha256(discovery).hexdigest()}
        result = json.loads(self.score(fabricated_evaluation(), **options))
        self.assertIsNone(result["selection"]["first_changed"])
        self.assertEqual(result["selection"]["ranking"], [1, 2, 3, 4])
        self.assertEqual(result["selection"]["magnitude"], 1)

    def test_smallest_subnormal_restoration_denominator_is_retained(self):
        raw = bytearray(logits(0.0))
        struct.pack_into("<I", raw, 1000 * 4, 1)
        cases = list(fabricated_evaluation())
        cases[0] = self.with_logits(cases[0], bytes(raw), (logits(0.0),) * 4)
        for index in range(1, 8):
            cases[index] = self.with_logits(cases[index], logits(0.0), (logits(0.0),) * 4)
        result = json.loads(self.score(tuple(cases), **self.options))
        restoration = result["sites"]["1"]["restoration"]
        self.assertEqual(restoration["denominator"], 2.0 ** -298)
        self.assertIsNone(restoration["value"])

    def test_self_consistent_digest_cannot_hide_inconsistent_frozen_mean(self):
        from looking_glass.pretrained_number_selection import EvaluationInvalid
        parsed = json.loads(self.discovery)
        parsed["predictions"]["2"] = 123.0
        forged = json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode()
        options = {**self.options, "discovery_bytes": forged,
                   "prediction_sha256": hashlib.sha256(forged).hexdigest()}
        with self.assertRaises(EvaluationInvalid):
            self.score(fabricated_evaluation(), **options)

    def test_rejects_noncanonical_frozen_bytes_even_with_matching_digest(self):
        from looking_glass.pretrained_number_selection import EvaluationInvalid
        invalid = self.discovery + b"\n"
        options = {**self.options, "discovery_bytes": invalid,
                   "prediction_sha256": hashlib.sha256(invalid).hexdigest()}
        with self.assertRaises(EvaluationInvalid):
            self.score(fabricated_evaluation(), **options)

    def test_rejects_impossible_finite_predictions_without_overflow(self):
        from looking_glass.pretrained_number_selection import EvaluationInvalid
        parsed = json.loads(self.discovery)
        for case in parsed["cases"]:
            case["l2"][0] = 1e308
        forged = json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode()
        options = {**self.options, "discovery_bytes": forged,
                   "prediction_sha256": hashlib.sha256(forged).hexdigest()}
        with self.assertRaises(EvaluationInvalid):
            self.score(fabricated_evaluation(), **options)

    def test_rejects_excessively_nested_prediction_json(self):
        from looking_glass.pretrained_number_selection import EvaluationInvalid
        malformed = b"[" * 5000 + b"]" * 5000
        options = {**self.options, "discovery_bytes": malformed,
                   "prediction_sha256": hashlib.sha256(malformed).hexdigest()}
        with self.assertRaises(EvaluationInvalid):
            self.score(fabricated_evaluation(), **options)

    def test_rejects_tampering_failed_controls_nonfinite_and_bad_attestations(self):
        from looking_glass.pretrained_number_selection import EvaluationInvalid
        cases = fabricated_evaluation()
        altered = self.discovery.replace(b'"earliest":1', b'"earliest":2', 1)
        raw = bytearray(cases[0].forward[0].logits)
        struct.pack_into("<I", raw, 50000 * 4, 0x7fc00000)
        invalid = replace(cases[0], forward=(replace(cases[0].forward[0], logits=bytes(raw)),) +
                          cases[0].forward[1:])
        failed = replace(cases[0], null=replace(cases[0].null, logits=logits(2.0)))
        bad_options = ({**self.options, "discovery_bytes": altered},
                       {**self.options, "custody_verified": False},
                       {**self.options, "private_controls_verified": False},
                       {**self.options, "heldout_commitment_sha256": "c" * 63})
        for options in bad_options:
            with self.assertRaises(EvaluationInvalid):
                self.score(cases, **options)
        for first in (invalid, failed):
            with self.assertRaises(EvaluationInvalid):
                self.score((first,) + cases[1:], **self.options)


if __name__ == "__main__":
    unittest.main()
