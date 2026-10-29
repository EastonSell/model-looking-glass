"""Private-witness checks on toy CPU models, never a pretrained study."""
import json
import math
import struct
import unittest
from dataclasses import replace

import torch

from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
from looking_glass.pretrained_number_adapter import to_selector_observation
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_injection import add_float32_bits, injection_direction_bits
from looking_glass.agreement_numeric import decode_f32_le
from looking_glass.pretrained_number_witness import (
    audit_private_forward, audit_private_record, decode_private_forward, encode_private_forward,
)
from tests.test_pretrained_number_injection import ToyModel


class NumberInjectionWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_sites_patch_orders_and_controls_have_private_witnesses(self):
        ids = PUBLIC_DISCOVERY[0][1]
        for site in range(1, 5):
            with self.subTest(site=site), NumberInjectionExecutor(ToyModel(), site=site) as executor:
                ordinal = 0
                def run(role, condition, *, layer=None, donor=None):
                    nonlocal ordinal
                    ordinal += 1
                    obs = executor.forward(ids, request_id=f"{ordinal:032x}", condition=condition,
                                           layer=layer, donor=donor)
                    proof = tuple(item for item in executor.private_evidence() if item.request_id == obs.request_id)
                    return to_selector_observation(obs), proof

                clean, clean_evidence = run("clean", "clean")
                self.assertTrue(audit_private_forward(0, "clean", clean, clean_evidence, site=site)["verified"])
                injected, injected_evidence = run("injected", "injected")
                self.assertTrue(audit_private_forward(0, "injected", injected, injected_evidence,
                                                      site=site, clean=clean)["verified"])
                clean_source, injected_source = executor.completed()[:2]
                for layer in range(1, 5):
                    patched, evidence = run(f"forward_{layer}", "clean_to_injected", layer=layer,
                                            donor=clean_source.mlp_donor(layer))
                    self.assertTrue(audit_private_forward(0, f"forward_{layer}", patched, evidence,
                                                          site=site, clean=clean, injected=injected)["verified"])
                    raw = encode_private_forward(0, f"forward_{layer}", patched, evidence)
                    restored = decode_private_forward(raw)
                    self.assertEqual((restored.case_index, restored.role, restored.request_id),
                                     (0, f"forward_{layer}", patched.request_id))
                    self.assertEqual(restored.evidence, evidence)
                    self.assertTrue(audit_private_record(raw, patched, site=site,
                                                         clean=clean, injected=injected)["verified"])
                    reverse, evidence = run(f"reverse_{layer}", "injected_to_clean", layer=layer,
                                            donor=injected_source.mlp_donor(layer))
                    self.assertTrue(audit_private_forward(0, f"reverse_{layer}", reverse, evidence,
                                                          site=site, clean=clean, injected=injected)["verified"])
                for role, condition, donor in (("upstream", "upstream", clean_source.mlp_donor(0)),
                                               ("downstream", "downstream", clean_source.block_donor()),
                                               ("null", "null", None), ("self", "self", None)):
                    result, evidence = run(role, condition, donor=donor)
                    self.assertTrue(audit_private_forward(0, role, result, evidence, site=site,
                                                          clean=clean, injected=injected)["verified"])

    def test_malformed_witnesses_and_upstream_changes_are_rejected(self):
        ids = PUBLIC_DISCOVERY[0][1]
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            clean = executor.forward(ids, request_id=f"{1:032x}", condition="clean")
            injected = executor.forward(ids, request_id=f"{2:032x}", condition="injected")
            patch = executor.forward(ids, request_id=f"{3:032x}", condition="clean_to_injected",
                                     layer=2, donor=clean.mlp_donor(2))
            clean_public, injected_public, patch_public = map(to_selector_observation, (clean, injected, patch))
            evidence = tuple(item for item in executor.private_evidence() if item.request_id == patch.request_id)
            add = next(item for item in evidence if item.operation == "add")
            action = next(item for item in evidence if item.operation == "patch")

            def reject(items, *, public=patch_public, site=2):
                with self.assertRaises(ValueError):
                    audit_private_forward(0, "forward_2", public, items, site=site,
                                          clean=clean_public, injected=injected_public)

            reject((add,))
            reject((add, add))
            changed = bytearray(add.after)
            changed[0] ^= 1  # A different, nonselected token.
            reject((replace(add, after=bytes(changed)), action))
            changed = bytearray(add.realized_delta_f64_le)
            changed[0] ^= 1
            reject((replace(add, realized_delta_f64_le=bytes(changed)), action))
            reject((add, replace(action, donor_identity=replace(action.donor_identity,
                                                                 request_id=injected.request_id))))
            with self.assertRaises(ValueError):
                encode_private_forward(0, "forward_2", patch_public,
                                       (add, replace(action, module_path="x" * (1024 * 1024))))
            with self.assertRaises(ValueError):
                encode_private_forward(0, "forward_2", patch_public,
                                       (add, replace(action, donor_identity=replace(action.donor_identity,
                                                                                    request_id="x" * 100000))))
            metadata = json.loads(add.metadata_json)
            metadata[1]["order"] = False
            bad_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            reject((replace(add, metadata_json=bad_metadata), action))
            reject(evidence, site=3)
            inj_evidence = tuple(item for item in executor.private_evidence() if item.request_id == injected.request_id)
            inj = inj_evidence[0]
            pre, post = bytearray(inj.before), bytearray(inj.after)
            offset = (len(ids) - 1) * 768 * 4
            before_bits = 0x3f800000
            after_bits = add_float32_bits(before_bits, injection_direction_bits()[0])
            struct.pack_into("<I", pre, offset, before_bits)
            struct.pack_into("<I", post, offset, after_bits)
            delta = bytearray(inj.realized_delta_f64_le)
            a = float(decode_f32_le(bytes(post[offset:offset + 4]))[0])
            b = float(decode_f32_le(bytes(pre[offset:offset + 4]))[0])
            struct.pack_into("<d", delta, 0, math.fsum((a, -b)))
            forged = replace(inj, before=bytes(pre), after=bytes(post), realized_delta_f64_le=bytes(delta))
            visible = bytearray(injected_public.mlp[2])
            struct.pack_into("<I", visible, 0, after_bits)
            mlp = injected_public.mlp[:2] + (bytes(visible),) + injected_public.mlp[3:]
            forged_public = replace(injected_public, mlp=mlp)
            with self.assertRaises(ValueError):
                audit_private_forward(0, "injected", forged_public, (forged,), site=2, clean=clean_public)
            changed_attention = bytearray(injected_public.attention[0])
            changed_attention[:4] = b"\x00\x00\x00\x00"
            bad = replace(injected_public, attention=(bytes(changed_attention),) + injected_public.attention[1:])
            with self.assertRaises(ValueError):
                audit_private_forward(0, "injected", bad, inj_evidence, site=2, clean=clean_public)

    def test_private_encoding_rejects_unknown_fields_and_is_never_a_public_record(self):
        ids = PUBLIC_DISCOVERY[0][1]
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            clean = executor.forward(ids, request_id=f"{1:032x}", condition="clean")
            injected = executor.forward(ids, request_id=f"{2:032x}", condition="injected")
            public = to_selector_observation(injected)
            proof = tuple(item for item in executor.private_evidence() if item.request_id == injected.request_id)
            with self.assertRaises(ValueError):
                encode_private_forward(0, "injected", public, ())
            with self.assertRaises(ValueError):
                encode_private_forward(0, "injected", public, (proof[0], proof[0]))
            raw = encode_private_forward(0, "injected", public, proof)
            self.assertIn(b"transformer.h.2.mlp", raw)  # Site information must remain private.
            record = json.loads(raw)
            record["actions"] = []
            altered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            with self.assertRaises(ValueError):
                decode_private_forward(altered)
            record = json.loads(raw)
            record["unexpected"] = "private marker"
            altered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            with self.assertRaises(ValueError):
                decode_private_forward(altered)
            with self.assertRaises(ValueError):
                decode_private_forward(raw + b"\n")
            record = json.loads(raw)
            record["request_id"] = clean.request_id
            record["actions"][0]["request_id"] = clean.request_id
            altered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            with self.assertRaises(ValueError):
                audit_private_record(altered, public, site=2, clean=to_selector_observation(clean))
            clean_public = to_selector_observation(clean)
            clean_raw = encode_private_forward(0, "clean", clean_public, ())
            record = json.loads(clean_raw)
            record["input_sha256"] = "0" * 64
            altered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            with self.assertRaises(ValueError):
                audit_private_record(altered, clean_public, site=2)
            self.assertTrue(audit_private_forward(0, "injected", public, proof, site=2,
                                                  clean=to_selector_observation(clean))["verified"])


if __name__ == "__main__":
    unittest.main()
