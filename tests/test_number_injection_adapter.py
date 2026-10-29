"""Fabricated integration of the ordered executor and site-blind selector.

The sole model used here is a toy CPU module; no checkpoint, site nonce,
held-out split, or actual experiment execution is involved.
"""
import json
import unittest
from dataclasses import replace

import torch

from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
from looking_glass.pretrained_number_injection import NumberInjectionExecutor
from looking_glass.pretrained_number_selection import DiscoveryInvalid, freeze_discovery
from looking_glass.pretrained_number_adapter import assemble_discovery_case, to_selector_observation
from tests.test_pretrained_number_injection import ToyModel
from tests.test_pretrained_number_selection import fabricated_cases


class NumberInjectionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_toy_executor_maps_all_roles_into_site_blind_freeze(self):
        ids = PUBLIC_DISCOVERY[0][1]
        counter = iter(range(1, 17))
        def request():
            return f"{next(counter):032x}"

        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            clean = executor.forward(ids, request_id=request(), condition="clean")
            repeat = executor.forward(ids, request_id=request(), condition="clean")
            injected = executor.forward(ids, request_id=request(), condition="injected")
            forward = tuple(executor.forward(ids, request_id=request(), condition="clean_to_injected",
                                             layer=layer, donor=clean.mlp_donor(layer))
                            for layer in range(1, 5))
            reverse = tuple(executor.forward(ids, request_id=request(), condition="injected_to_clean",
                                             layer=layer, donor=injected.mlp_donor(layer))
                            for layer in range(1, 5))
            null = executor.forward(ids, request_id=request(), condition="null")
            self_patch = executor.forward(ids, request_id=request(), condition="self")
            upstream = executor.forward(ids, request_id=request(), condition="upstream", donor=clean.mlp_donor(0))
            downstream = executor.forward(ids, request_id=request(), condition="downstream", donor=clean.block_donor())
            verify_clean = executor.forward(ids, request_id=request(), condition="clean")

            case = assemble_discovery_case(index=0, clean=clean, repeat=repeat, injected=injected,
                                           forward=forward, reverse=reverse, null=null, self_patch=self_patch,
                                           upstream=upstream, downstream=downstream, verify_clean=verify_clean)
            self.assertFalse(hasattr(case.injected, "_owner_token"))
            self.assertFalse(hasattr(case.injected, "site"))
            self.assertIs(case.clean.logits, clean.logits)
            self.assertEqual(case.forward[1].action.module_path, "transformer.h.2.mlp")
            self.assertEqual(case.forward[1].action.donor.request_id, clean.request_id)
            self.assertEqual(case.reverse[1].action.donor.request_id, injected.request_id)
            self.assertEqual(case.self_patch.action.donor.request_id, self_patch.request_id)
            self.assertEqual(case.downstream.action.module_path, "transformer.h.5")
            self.assertEqual(case.downstream.action.output_selector, (0,))
            self.assertEqual(case.downstream.action.output_shape, (1, len(ids), 768))
            self.assertEqual(len(executor.private_evidence()), 21)

            # The other seven are selector test fixtures. The True flag is a
            # fabricated attestation solely for this synthetic integration.
            cases = (case,) + fabricated_cases()[1:]
            frozen = json.loads(freeze_discovery(cases, private_controls_verified=True, execution_device="cpu"))
            self.assertEqual(frozen["status"], "complete")
            self.assertEqual(len(frozen["cases"]), 8)
            self.assertEqual(frozen["cases"][0]["input_ids"], list(ids))
            self.assertEqual(len(frozen["cases"][0]["observations"]), 16)

            wrong_donor = replace(case.forward[1].action.donor, request_id=injected.request_id)
            wrong_action = replace(case.forward[1].action, donor=wrong_donor)
            bad = replace(case.forward[1], action=wrong_action)
            altered = replace(case, forward=(case.forward[0], bad, *case.forward[2:]))
            with self.assertRaises(DiscoveryInvalid):
                freeze_discovery((altered,) + cases[1:], private_controls_verified=True, execution_device="cpu")

    def test_private_evidence_and_nonobservations_are_rejected(self):
        with self.assertRaises(TypeError):
            to_selector_observation(object())
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            executor.forward((2, 3), request_id=f"{1:032x}", condition="injected")
            with self.assertRaises(TypeError):
                to_selector_observation(executor.private_evidence()[0])


if __name__ == "__main__":
    unittest.main()
