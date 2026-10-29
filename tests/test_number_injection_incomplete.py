"""Private failure serialization preserves raw bits without model execution."""
import struct
import unittest

from looking_glass.pretrained_number_injection import PrivateActionEvidence, PrivateFailureEvidence
from looking_glass.pretrained_number_incomplete import decode_incomplete_forward, encode_incomplete_forward


class IncompleteForwardTests(unittest.TestCase):
    def test_nonfinite_bits_and_arbitrary_bounded_private_metadata_round_trip(self):
        raw = struct.pack("<6I", 0x7fc00001, 0x7f800000, 0xff800000, 0x80000000, 1, 0x3f800000)
        metadata = 'unparsed private metadata: \u2603\n{"site": "private"}\ud800'
        failure = PrivateFailureEvidence("a" * 32, "b" * 64, "injected", "RuntimeError",
                                         "private diagnostic", (("capture-0", raw),), raw, metadata)
        encoded = encode_incomplete_forward(failure=failure)
        result = decode_incomplete_forward(encoded.data)
        self.assertEqual(result.payloads["failure.captures.0.data"], raw)
        self.assertEqual(result.payloads["failure.logits"], raw)
        self.assertEqual(result.payloads["failure.metadata_json"].decode("utf-8", "surrogatepass"), metadata)
        self.assertEqual(result.payloads["failure.exception_message"], b"private diagnostic")
        self.assertTrue(result.manifest["failure"]["exception_details_may_be_truncated"])
        self.assertEqual(encoded.tensor_bytes, len(raw))  # Shared object counted once.
        self.assertEqual(encoded.omissions, ())

    def test_unavailable_and_oversize_fields_are_explicit(self):
        failure = PrivateFailureEvidence(None, "", "invalid", "Error", "message",
                                         (("oversize", b"x" * (1024 * 1024 + 1)),
                                          ("valid", b"y" * 16)), None, None)
        encoded = encode_incomplete_forward(failure=failure)
        result = decode_incomplete_forward(encoded.data)
        self.assertIn("failure.captures.0.data", encoded.omissions)
        self.assertEqual(result.payloads["failure.captures.1.data"], b"y" * 16)
        field = next(item for item in result.manifest["fields"] if item["path"] == "failure.captures.0.data")
        self.assertEqual(field["status"], "oversize")
        self.assertIsNone(result.manifest["observation"])

    def test_action_witnesses_are_preserved_even_without_an_observation(self):
        bits = struct.pack("<I", 0x7fa01234)
        action = PrivateActionEvidence("a" * 32, "b" * 64, "injected", "add", "transformer.h.2.mlp", 3,
                                       bits, bits, None, None, None, "private order data")
        encoded = encode_incomplete_forward(actions=(action,))
        result = decode_incomplete_forward(encoded.data)
        self.assertEqual(result.payloads["actions.0.before"], bits)
        self.assertEqual(result.payloads["actions.0.after"], bits)
        self.assertEqual(result.payloads["actions.0.metadata_json"], b"private order data")
        self.assertIsNone(result.manifest["observation"])
        self.assertEqual(encoded.tensor_bytes, len(bits))

    def test_maximum_raw_tensor_fits_even_with_metadata_overhead(self):
        payload = b"a" * (1024 * 1024)
        failure = PrivateFailureEvidence(None, "", "invalid", "Error", "message",
                                         (("maximum", payload),), None, "bounded metadata")
        encoded = encode_incomplete_forward(failure=failure)
        self.assertGreater(len(encoded.data), 1024 * 1024)
        self.assertEqual(encoded.tensor_bytes, len(payload))
        self.assertEqual(decode_incomplete_forward(encoded.data).payloads["failure.captures.0.data"], payload)
        self.assertEqual(encoded.omissions, ())

    def test_total_raw_cap_reports_every_dropped_field(self):
        first = b"a" * (700 * 1024)
        second = b"b" * (700 * 1024)
        failure = PrivateFailureEvidence(None, "", "invalid", "Error", "message",
                                         (("first", first), ("second", second)), None, None)
        encoded = encode_incomplete_forward(failure=failure)
        result = decode_incomplete_forward(encoded.data)
        self.assertEqual(result.payloads["failure.captures.0.data"], first)
        self.assertIn("failure.captures.1.data", encoded.omissions)
        field = next(x for x in result.manifest["fields"] if x["path"] == "failure.captures.1.data")
        self.assertEqual(field["status"], "raw-cap")

    def test_escaped_metadata_overflow_preserves_raw_captures(self):
        raw = b"\x01\x02\x03\x04"
        captures = tuple(("\u2603" * 500 + str(i), raw) for i in range(64))
        failure = PrivateFailureEvidence(None, "", "invalid", "Error", "message", captures, None, None)
        encoded = encode_incomplete_forward(failure=failure)
        result = decode_incomplete_forward(encoded.data)
        for i in range(64):
            self.assertEqual(result.payloads[f"failure.captures.{i}.data"], raw)
        self.assertTrue(any(path.endswith(".name") for path in encoded.omissions))

    def test_completed_toy_forward_preserves_observation_and_private_witnesses(self):
        import torch
        from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
        from looking_glass.pretrained_number_injection import NumberInjectionExecutor
        from tests.test_pretrained_number_injection import ToyModel
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with NumberInjectionExecutor(ToyModel(), site=2) as executor:
                observation = executor.forward(PUBLIC_DISCOVERY[0][1], request_id="a" * 32, condition="injected")
                actions = tuple(x for x in executor.private_evidence() if x.request_id == observation.request_id)
                encoded = encode_incomplete_forward(observation=observation, actions=actions)
                result = decode_incomplete_forward(encoded.data)
                self.assertEqual(result.payloads["observation.logits"], observation.logits)
                self.assertEqual(result.payloads["observation.mlp.2"], observation.mlp[2])
                self.assertEqual(result.payloads["actions.0.before"], actions[0].before)
                self.assertEqual(result.payloads["actions.0.after"], actions[0].after)
                self.assertEqual(result.manifest["observation"]["trace"][5]["module_path"], "transformer.h.2.mlp")
                self.assertEqual(result.manifest["observation"]["owner_token"], "excluded-process-local")
                self.assertEqual(encoded.omissions, ())
        finally:
            torch.set_num_threads(old_threads)

    def test_integrity_and_exact_size_are_checked(self):
        failure = PrivateFailureEvidence(None, "", "invalid", "Error", "message", (("capture", b"abcd"),), None, None)
        encoded = encode_incomplete_forward(failure=failure)
        for bad in (encoded.data + b"x", encoded.data[:-1], encoded.data[:20] + b"x" + encoded.data[21:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                decode_incomplete_forward(bad)


if __name__ == "__main__":
    unittest.main()
