"""CPU-only provenance audit using public discovery IDs and a synthetic toy model.

No pretrained model, private site commitment, hidden input or held-out case is used.
"""
import hashlib
import json
import unittest

import torch

from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
from looking_glass.pretrained_number_adapter import assemble_discovery_case, to_selector_observation
from looking_glass.pretrained_number_archive import _read, audit_discovery_archive, serialize_discovery_observation
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_selection import freeze_discovery
from tests.test_number_injection_coordinate_oracle import Model


class ArchiveAudit(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        tiny = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
        previous = int((tiny * 1.0).view(torch.int32).item()) == 0
        if not torch.set_flush_denormal(False):
            self.skipTest("CPU denormal mode unavailable")
        self.addCleanup(torch.set_flush_denormal, previous)

    def test_all_executor_records_round_trip_and_bind_sources(self):
        records, cases, source_ids, request = [], [], [], 0
        def next_id():
            nonlocal request
            request += 1
            return f"{request:032x}"

        with NumberInjectionExecutor(Model(), site=2) as executor:
            for index in range(8):
                ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
                clean = executor.forward(ids, request_id=next_id(), condition="clean")
                repeat = executor.forward(ids, request_id=next_id(), condition="clean")
                injected = executor.forward(ids, request_id=next_id(), condition="injected")
                forward = tuple(executor.forward(ids, request_id=next_id(), condition="clean_to_injected",
                                                layer=layer, donor=clean.mlp_donor(layer))
                                for layer in range(1, 5))
                reverse = tuple(executor.forward(ids, request_id=next_id(), condition="injected_to_clean",
                                                layer=layer, donor=injected.mlp_donor(layer))
                                for layer in range(1, 5))
                null = executor.forward(ids, request_id=next_id(), condition="null")
                self_patch = executor.forward(ids, request_id=next_id(), condition="self")
                upstream = executor.forward(ids, request_id=next_id(), condition="upstream", donor=clean.mlp_donor(0))
                downstream = executor.forward(ids, request_id=next_id(), condition="downstream", donor=clean.block_donor())
                verify = executor.forward(ids, request_id=next_id(), condition="clean")
                cases.append(assemble_discovery_case(index=index, clean=clean, repeat=repeat, injected=injected,
                             forward=forward, reverse=reverse, null=null, self_patch=self_patch,
                             upstream=upstream, downstream=downstream, verify_clean=verify))
                source_ids.append((clean.request_id, injected.request_id))
                entries = (("clean", clean), ("repeat", repeat), ("injected", injected),
                           *((f"forward_{i}", item) for i, item in enumerate(forward, 1)),
                           *((f"reverse_{i}", item) for i, item in enumerate(reverse, 1)),
                           ("null", null), ("self", self_patch), ("upstream", upstream),
                           ("downstream", downstream), ("verify_clean", verify))
                for role, observation in entries:
                    public = to_selector_observation(observation)
                    self.assertEqual((public.request_id, public.input_sha256, public.condition,
                                      public.attention, public.mlp, public.block, public.logits),
                                     (observation.request_id, observation.input_sha256, observation.condition,
                                      observation.attention, observation.mlp, observation.block, observation.logits))
                    boundary_fields = ("module_path", "capture_order", "invocation", "batch_index", "token_index",
                                       "output_selector", "output_shape", "selected_shape", "dtype", "device")
                    self.assertEqual(len(public.trace), len(observation.trace))
                    for converted, source in zip(public.trace, observation.trace):
                        self.assertEqual(tuple(getattr(converted, field) for field in boundary_fields),
                                         tuple(getattr(source, field) for field in boundary_fields))
                    if observation.action is None:
                        self.assertIsNone(public.action)
                    else:
                        action_fields = ("operation", "module_path", "invocation", "batch_index", "token_index",
                                         "output_selector", "output_shape", "selected_shape", "dtype", "device")
                        self.assertEqual(tuple(getattr(public.action, field) for field in action_fields),
                                         tuple(getattr(observation.action, field) for field in action_fields))
                        source_donor, converted_donor = observation.action.donor_identity, public.action.donor
                        if source_donor is None:
                            self.assertIsNone(converted_donor)
                        else:
                            donor_fields = ("request_id", "input_sha256", "module_path", "invocation", "batch_index",
                                            "token_index", "output_selector", "selected_shape", "dtype", "device")
                            self.assertEqual(tuple(getattr(converted_donor, field) for field in donor_fields),
                                             tuple(getattr(source_donor, field) for field in donor_fields))
                    raw = serialize_discovery_observation(index, role, public, execution_device="cpu")
                    got_index, got_role, restored = _read(raw, "cpu")
                    self.assertEqual((got_index, got_role, restored), (index, role, public))
                    self.assertNotIn(b"inject_before", raw)
                    self.assertNotIn(b"_owner_token", raw)
                    records.append(raw)
            self.assertEqual((len(executor.completed()), len(records)), (128, 128))
            self.assertEqual(len({item.request_id for item in executor.completed()}), 128)
            prediction = freeze_discovery(tuple(cases), private_controls_verified=True, execution_device="cpu")
            result = audit_discovery_archive(tuple(records), prediction, execution_device="cpu")
            expected = hashlib.sha256(b"MLG-number-injection-archive-v1\0" +
                                      b"".join(hashlib.sha256(raw).digest() for raw in records)).hexdigest()
            self.assertEqual(result, {"record_count": 128, "archive_sha256": expected,
                                      "prediction_sha256": hashlib.sha256(prediction).hexdigest()})
            self.assertEqual(audit_discovery_archive(tuple(reversed(records)), prediction,
                                                      execution_device="cpu"), result)
            # Each role is structurally valid by itself, but this forged cross-source
            # request must be rejected when the complete case is reconstructed.
            altered = json.loads(records[3])
            self.assertEqual((altered["case_index"], altered["role"]), (0, "forward_1"))
            altered["action"]["donor"]["request_id"] = source_ids[0][1]
            raw = json.dumps(altered, sort_keys=True, ensure_ascii=True, allow_nan=False,
                             separators=(",", ":")).encode()
            _read(raw, "cpu")
            with self.assertRaises(ValueError):
                audit_discovery_archive(tuple(records[:3] + [raw] + records[4:]), prediction,
                                        execution_device="cpu")


if __name__ == "__main__":
    unittest.main()
