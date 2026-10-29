"""Independent CPU GPT-2 oracle for the natural agreement executor.

All inputs and weights here are synthetic development fixtures. This is not a
pretrained-model result, a held-out case, or an evaluation of a causal claim.
"""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@dataclass(frozen=True)
class _Case:
    case_id: str
    lemma: str
    number: str
    tokens: tuple[int, ...]
    subject_index: int = 1


_CASES = tuple(_Case(f"{lemma}-{number}", lemma, number, (1, token, 2, 3))
               for lemma, number, token in (("alpha", "singular", 11),
                                            ("alpha", "plural", 12),
                                            ("beta", "singular", 13),
                                            ("beta", "plural", 14)))
_SITES = tuple((layer, kind) for layer in range(1, 5) for kind in ("attn", "mlp"))
_CONTRAST = (318, 389, 373, 547)


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _model():
    config = GPT2Config(vocab_size=600, n_positions=16, n_ctx=16, n_embd=8,
                        n_layer=6, n_head=2, use_cache=False,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        return GPT2LMHeadModel(config).eval()


def _module(model, site):
    layer, kind = site
    block = model.transformer.h[layer]
    return block if kind == "block" else getattr(block, kind)


def _tensor(output):
    return output[0] if type(output) is tuple else output


def _run(model, case):
    ids = torch.tensor([case.tokens])
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]


def _input_sha(case):
    data = json.dumps({"ids": case.tokens, "subject_index": case.subject_index},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class AgreementCoordinateOracleTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def reference(self):
        """Capture/patch with ordinary hooks, independently of ActivationPlan."""
        model = _model()
        self.assertEqual(model.config._attn_implementation, "eager")
        self.assertIs(model.config.use_cache, False)
        clean, patches, controls = {}, {}, {}
        with torch.inference_mode():
            for case in _CASES:
                captured, counts, handles = {}, {}, []
                for site in _SITES + ((5, "block"),):
                    def capture(_module, _args, output, site=site):
                        counts[site] = counts.get(site, 0) + 1
                        self.assertEqual(type(output) is tuple, site[1] != "mlp")
                        value = _tensor(output)
                        self.assertEqual(tuple(value.shape), (1, len(case.tokens), 8))
                        captured[site] = {"last": value[0, -1].clone(),
                                          "subject": value[0, case.subject_index].clone(),
                                          "whole": value.clone()}

                    handles.append(_module(model, site).register_forward_hook(capture))
                try:
                    logits = _run(model, case).clone()
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(captured), 9)
                self.assertEqual(set(counts.values()), {1})
                clean[case.case_id] = {"sites": captured, "logits": logits}
            for site in _SITES:
                for target in _CASES:
                    for donor in _CASES:
                        source = clean[donor.case_id]["sites"][site]["last"]
                        saved, hits = {}, []

                        def patch(_module, _args, output):
                            hits.append(None)
                            self.assertEqual(type(output) is tuple, site[1] != "mlp")
                            value = _tensor(output)
                            saved["pre"] = _bits(value[0, -1])
                            saved["whole_pre"] = _bits(value)
                            changed = value.clone()
                            changed[0, -1] = source
                            saved["post"] = _bits(changed[0, -1])
                            saved["whole_post"] = _bits(changed)
                            return (changed,) + output[1:] if type(output) is tuple else changed

                        handle = _module(model, site).register_forward_hook(patch)
                        try:
                            logits = _run(model, target).clone()
                        finally:
                            handle.remove()
                        self.assertEqual(len(hits), 1)
                        self.assertEqual(saved["post"], _bits(source))
                        patches[(target.case_id, donor.case_id, site)] = {
                            "logits": _bits(logits[list(_CONTRAST)]),
                            "pre": saved["pre"], "post": saved["post"], "donor": _bits(source),
                            "whole_pre": saved["whole_pre"], "whole_post": saved["whole_post"]}
                        if target == donor:
                            self.assertEqual(_bits(logits), _bits(clean[target.case_id]["logits"]))
                for target in _CASES:
                    # This fixture makes every swapped donor observable at the
                    # four contrast logits for every target and site.
                    self.assertEqual(len({patches[(target.case_id, donor.case_id, site)]["logits"]
                                          for donor in _CASES}), 4)
            for target in _CASES:
                donor = next(case for case in _CASES
                             if case.lemma == target.lemma and case.number != target.number)
                for role in ("last", "subject"):
                    position = -1 if role == "last" else target.subject_index
                    source = clean[donor.case_id]["sites"][(5, "block")][role]
                    saved = {}

                    def patch(_module, _args, output):
                        value = _tensor(output)
                        saved["whole_pre"] = _bits(value)
                        changed = value.clone()
                        changed[0, position] = source
                        saved["whole_post"] = _bits(changed)
                        return (changed,) + output[1:] if type(output) is tuple else changed

                    handle = _module(model, (5, "block")).register_forward_hook(patch)
                    try:
                        logits = _run(model, target).clone()
                    finally:
                        handle.remove()
                    reference_id = donor.case_id if role == "last" else target.case_id
                    self.assertEqual(_bits(logits), _bits(clean[reference_id]["logits"]))
                    controls[(target.case_id, donor.case_id, role)] = {
                        "logits": _bits(logits), "whole_pre": saved["whole_pre"],
                        "whole_post": saved["whole_post"]}
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))
        return clean, patches, controls

    def check_step(self, step, case, layer, kind, role, index, action):
        path = f"transformer.h.{layer}" + ("" if kind == "block" else "." + kind)
        position = (None if role is None else
                    case.subject_index if role == "subject" else len(case.tokens) - 1)
        batch = None if role is None else 0
        selected_shape = [1, len(case.tokens), 8] if role is None else [8]
        self.assertEqual((step["order"], step["operation"], step["module_path"]),
                         (index, action, path))
        self.assertEqual(step["output_selector"], [0] if kind in ("attn", "block") else [])
        self.assertEqual((step["invocation"], step["forward_index"], step["batch_index"],
                          step["token_index"]), (0, 0, batch, position))
        self.assertEqual((step["output_shape"], step["selected_shape"], step["dtype"], step["device"]),
                         ([1, len(case.tokens), 8], selected_shape, "torch.float32", "cpu"))

    def check_action_steps(self, steps, target, layer, component, role, action):
        self.assertEqual([step["name"] for step in steps],
                         ["whole_before", "before", "action", "after", "whole_after"])
        for index, operation in enumerate(("capture", "capture", action, "capture", "capture")):
            selection = None if index in (0, 4) else role
            self.check_step(steps[index], target, layer, component, selection, index, operation)

    def check_whole_witness(self, row, raw, target, role, expected_before, expected_after):
        self.assertIs(row["event"]["untouched_output_bits_equal"], True)
        before, after = raw(row, "whole_before"), raw(row, "whole_after")
        self.assertEqual(before, expected_before)
        self.assertEqual(after, expected_after)
        for name in ("whole_before", "whole_after"):
            self.assertEqual((row["tensors"][name]["shape"], row["tensors"][name]["dtype"]),
                             ([1, len(target.tokens), 8], "float32-le"))
        position = target.subject_index if role == "subject" else len(target.tokens) - 1
        start, end = position * 8 * 4, (position + 1) * 8 * 4
        self.assertEqual((before[:start], before[end:]), (after[:start], after[end:]))
        self.assertEqual((before[start:end], after[start:end]), (raw(row, "before"), raw(row, "after")))

    def test_all_recorded_pairs_and_position_controls_match_manual_hooks(self):
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import AgreementCase, NaturalAgreementRunner

        clean, patches, controls = self.reference()
        model = _model()
        cases = tuple(AgreementCase(case.case_id, "synthetic-q", case.lemma, case.number,
                                    case.tokens, case.subject_index) for case in _CASES)
        by_id = {case.case_id: case for case in _CASES}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt"
            journal = AgreementJournal(path, {"protocol": "synthetic-development-only"})
            runner = NaturalAgreementRunner(model, journal, device="cpu", expected_quartets=1)
            result = runner.run(cases)
            self.assertEqual((result["status"], result["forwards"]), ("complete", 152))
            rows = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]

            def raw(row, name):
                ref = row["tensors"][name]
                self.assertEqual(ref["dtype"], "float32-le")
                data = (path / "tensors" / (ref["sha256"] + ".f32le")).read_bytes()
                self.assertEqual(len(data), ref["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), ref["sha256"])
                return data

            counts, seen_patches, seen_controls, seen_clean = {}, set(), set(), set()
            seen_repeat, seen_final, seen_null = set(), set(), set()
            seen_metric_sites = {"present_site_metrics": set(), "secondary_past_site_metrics": set()}
            capture_sites = _SITES + ((5, "block"), (5, "block"))
            for row in rows:
                event, kind = row["event"], row["event"]["kind"]
                counts[kind] = counts.get(kind, 0) + 1
                if kind not in ("summary", "run_contract", *seen_metric_sites):
                    self.assertEqual(event["quartet"], "synthetic-q")
                if kind == "run_contract":
                    self.assertEqual(event["cases"], [
                        {"case_id": case.case_id, "quartet": "synthetic-q", "lemma": case.lemma,
                         "number": case.number, "input_ids": list(case.tokens),
                         "attention_mask": [1] * len(case.tokens), "subject_index": case.subject_index}
                        for case in _CASES])
                    expected_sites = []
                    for layer, component, role in ([(layer, component, "last") for layer, component in _SITES] +
                                                   [(5, "block", "last"), (5, "block", "subject")]):
                        site_path = f"transformer.h.{layer}" + ("" if component == "block" else "." + component)
                        expected_sites.append({"id": f"{layer}:{component}:{role}", "module_path": site_path,
                                               "layer": layer, "component": component, "token_role": role,
                                               "output_selector": [0] if component in ("attn", "block") else []})
                    self.assertEqual(event["sites"], expected_sites)
                elif kind in seen_metric_sites:
                    site_id = event["site"]
                    self.assertNotIn(site_id, seen_metric_sites[kind])
                    seen_metric_sites[kind].add(site_id)
                    layer_text, component, role = site_id.split(":")
                    layer = int(layer_text)
                    self.assertEqual(role, "last")
                    self.assertIn((layer, component), _SITES)
                    self.assertEqual(event["metrics"]["site_summary"]["site"], site_id)
                    pairs = event["metrics"]["pairs"]
                    self.assertEqual(len(pairs), 16)
                    seen_pairs = set()
                    for pair in pairs:
                        target, donor = by_id[pair["target"]], by_id[pair["donor"]]
                        self.assertEqual((pair["site"], pair["quartet"]), (site_id, "synthetic-q"))
                        key = (target.case_id, donor.case_id)
                        self.assertNotIn(key, seen_pairs)
                        seen_pairs.add(key)
                        expected_cell = ("same_lemma" if target.lemma == donor.lemma else "different_lemma") + (
                            "_same_number" if target.number == donor.number else "_opposite_number")
                        self.assertEqual(pair["cell"], expected_cell)
                        scores = struct.unpack("<4f", patches[(target.case_id, donor.case_id, (layer, component))]["logits"])
                        offset = 0 if kind == "present_site_metrics" else 2
                        self.assertEqual((pair["patched_is"], pair["patched_are"]), scores[offset:offset + 2])
                    self.assertEqual(seen_pairs, {(target.case_id, donor.case_id)
                                                  for target in _CASES for donor in _CASES})
                elif kind == "clean_capture":
                    case = by_id[event["case"]]
                    self.assertNotIn(case.case_id, seen_clean)
                    seen_clean.add(case.case_id)
                    self.assertEqual(event["input_sha256"], _input_sha(case))
                    self.assertEqual(raw(row, "last_logits"), _bits(clean[case.case_id]["logits"]))
                    steps = event["plan_metadata"]["steps"]
                    self.assertEqual(len(steps), 10)
                    for index, (layer, component) in enumerate(capture_sites):
                        role = "subject" if index == 9 else "last"
                        self.assertEqual(steps[index]["name"], f"capture_{index}")
                        self.check_step(steps[index], case, layer, component, role, index, "capture")
                        self.assertEqual(raw(row, f"capture_{index}"),
                                         _bits(clean[case.case_id]["sites"][(layer, component)][role]))
                elif kind == "patch":
                    target, donor = by_id[event["target"]], by_id[event["donor"]]
                    site = event["site"]
                    layer, component = site["layer"], site["component"]
                    self.assertIn((layer, component), _SITES)
                    self.assertEqual(site["id"], f"{layer}:{component}:last")
                    self.assertEqual(site["module_path"], f"transformer.h.{layer}.{component}")
                    self.assertEqual(site["output_selector"], [0] if component == "attn" else [])
                    self.assertEqual(site["token_role"], "last")
                    self.assertEqual((event["target_input_sha256"], event["source_input_sha256"]),
                                     (_input_sha(target), _input_sha(donor)))
                    self.assertEqual(event["self_patch"], target == donor)
                    self.assertIs(event["post_action_matches_donor_bits"], True)
                    key = (target.case_id, donor.case_id, (layer, component))
                    self.assertNotIn(key, seen_patches)
                    seen_patches.add(key)
                    steps = event["plan_metadata"]["steps"]
                    self.check_action_steps(steps, target, layer, component, "last", "patch")
                    self.assertEqual(steps[2]["source_capture"], "before" if target == donor else None)
                    reference = patches[key]
                    self.assertEqual(raw(row, "before"), reference["pre"])
                    self.assertEqual(raw(row, "after"), reference["post"])
                    self.assertEqual(reference["post"], reference["donor"])
                    self.check_whole_witness(row, raw, target, "last", reference["whole_pre"], reference["whole_post"])
                    if target == donor:
                        full = raw(row, "last_logits")
                        logits = b"".join(full[4 * index:4 * index + 4] for index in _CONTRAST)
                    else:
                        logits = raw(row, "contrast_logits")
                    self.assertEqual(logits, reference["logits"])
                elif kind in ("delivery", "subject_position"):
                    target, donor = by_id[event["target"]], by_id[event["donor"]]
                    role = "last" if kind == "delivery" else "subject"
                    key = (target.case_id, donor.case_id, role)
                    self.assertNotIn(key, seen_controls)
                    seen_controls.add(key)
                    self.assertEqual(donor.lemma, target.lemma)
                    self.assertNotEqual(donor.number, target.number)
                    self.assertEqual(event["reference"], donor.case_id if role == "last" else target.case_id)
                    self.assertIs(event["post_action_matches_donor_bits"], True)
                    self.assertEqual((event["site"]["module_path"], event["site"]["output_selector"],
                                      event["site"]["token_role"], event["site"]["id"],
                                      event["site"]["layer"], event["site"]["component"]),
                                     ("transformer.h.5", [0], role, f"5:block:{role}", 5, "block"))
                    steps = event["plan_metadata"]["steps"]
                    self.check_action_steps(steps, target, 5, "block", role, "patch")
                    self.assertEqual(raw(row, "before"), _bits(clean[target.case_id]["sites"][(5, "block")][role]))
                    self.assertEqual(raw(row, "after"), _bits(clean[donor.case_id]["sites"][(5, "block")][role]))
                    self.check_whole_witness(row, raw, target, role, controls[key]["whole_pre"], controls[key]["whole_post"])
                    self.assertEqual(raw(row, "last_logits"), controls[key]["logits"])
                elif kind == "null":
                    target = by_id[event["target"]]
                    self.assertNotIn(target.case_id, seen_null)
                    seen_null.add(target.case_id)
                    self.assertEqual(event["site"]["module_path"], "transformer.h.1.attn")
                    self.assertEqual((event["site"]["id"], event["site"]["token_role"],
                                      event["site"]["output_selector"]), ("1:attn:last", "last", [0]))
                    steps = event["plan_metadata"]["steps"]
                    self.check_action_steps(steps, target, 1, "attn", "last", "null")
                    self.assertEqual(raw(row, "before"), raw(row, "after"))
                    self.assertEqual(raw(row, "before"), _bits(clean[target.case_id]["sites"][(1, "attn")]["last"]))
                    whole = _bits(clean[target.case_id]["sites"][(1, "attn")]["whole"])
                    self.check_whole_witness(row, raw, target, "last", whole, whole)
                    self.assertEqual(raw(row, "last_logits"), _bits(clean[target.case_id]["logits"]))
                elif kind == "clean_repeat":
                    case = by_id[event["case"]]
                    self.assertNotIn(case.case_id, seen_repeat)
                    seen_repeat.add(case.case_id)
                    self.assertEqual(raw(row, "last_logits"), _bits(clean[case.case_id]["logits"]))
                    steps = event["plan_metadata"]["steps"]
                    self.assertEqual(len(steps), 10)
                    for index, (layer, component) in enumerate(capture_sites):
                        role = "subject" if index == 9 else "last"
                        self.assertEqual(steps[index]["name"], f"capture_{index}")
                        self.check_step(steps[index], case, layer, component, role, index, "capture")
                        self.assertEqual(raw(row, f"capture_{index}"),
                                         _bits(clean[case.case_id]["sites"][(layer, component)][role]))
                    self.assertTrue(all(check["within_tolerance"] for check in event["checks"].values()))
                elif kind == "final_clean":
                    case = by_id[event["case"]]
                    self.assertNotIn(case.case_id, seen_final)
                    seen_final.add(case.case_id)
                    self.assertEqual(raw(row, "last_logits"), _bits(clean[case.case_id]["logits"]))
                    self.assertTrue(event["logits_check"]["within_tolerance"])
            self.assertEqual(seen_clean, set(by_id))
            self.assertEqual(seen_patches, set(patches))
            self.assertEqual(seen_controls, set(controls))
            self.assertEqual((seen_repeat, seen_final, seen_null), (set(by_id), set(by_id), set(by_id)))
            expected_site_ids = {f"{layer}:{kind}:last" for layer, kind in _SITES}
            self.assertTrue(all(sites == expected_site_ids for sites in seen_metric_sites.values()))
            self.assertEqual(counts["run_contract"], 1)
            self.assertEqual(counts["summary"], 1)
            self.assertEqual({kind: counts[kind] for kind in ("clean_capture", "patch", "delivery",
                                                              "subject_position", "null", "clean_repeat",
                                                              "final_clean")},
                             {"clean_capture": 4, "patch": 128, "delivery": 4,
                              "subject_position": 4, "null": 4, "clean_repeat": 4, "final_clean": 4})
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
