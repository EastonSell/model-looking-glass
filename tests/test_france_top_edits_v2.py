import pathlib, sys, unittest
import torch
_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src" if (_root / "src" / "looking_glass").is_dir() else _root))
from looking_glass import targeted_edit_screened_math as prior
from looking_glass import targeted_edit_france_top_edits_v2 as subject
from looking_glass import targeted_edit_full_top as protocol

def fixture():
    w = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(6, 4)
    x = torch.zeros((2,6), dtype=torch.float32)
    x[0,0] = 1
    x[1,1] = 1
    h = torch.arange(1,7,dtype=torch.float32)
    g = torch.tensor([1.0,-2.0,0.5,3.0],dtype=torch.float32)
    return w,x,h,g,torch.outer(h,g)

class EditTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.set_flush_denormal(False))

    def test_selected_dose_scales_all_five_controls_without_changing_global_rng(self):
        args=fixture()
        state=torch.get_rng_state().clone()
        prepared=subject.prepare(*args)
        result=prepared.conditions(0.004)
        self.assertEqual(set(result["updates"]), {"projected","unprojected","projected_random","opposite","nuisance"})
        self.assertEqual(result["dose"], 0.004)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(result["updates"]["opposite"],-result["updates"]["projected"]))
        for entry in result["attestations"].values():
            self.assertGreaterEqual(entry["realized_to_planned"],0.999)
            self.assertLessEqual(entry["realized_to_planned"],1.001)
            self.assertGreaterEqual(entry["realized_to_projected"],0.999)
            self.assertLessEqual(entry["realized_to_projected"],1.001)

    def test_fixed_prior_fraction_is_bitwise_identical_for_all_arms(self):
        args = fixture()
        expected = prior.prepare_projected_edits(*args)
        prepared = subject.prepare(*args)
        actual = prepared.conditions(0.001)
        for name in prior.CONDITIONS:
            self.assertTrue(torch.equal(actual["updates"][name], expected["updates"][name]), name)
            for key in ("planned_float64_sha256", "intended_delta_sha256",
                        "realized_weight_sha256", "realized_delta_float64_sha256"):
                self.assertEqual(actual["attestations"][name][key],
                                 expected["attestation"]["conditions"][name][key])
        self.assertTrue(torch.equal(prepared.unprojected(0.001)["updates"]["unprojected"],
                                    actual["updates"]["unprojected"]))
        self.assertTrue(torch.equal(prepared.projected(0.001)["updates"]["projected"],
                                    actual["updates"]["projected"]))

    def test_every_fixed_grid_fraction_is_constructible_and_scales_as_declared(self):
        prepared = subject.prepare(*fixture())
        for name, grid, fn in (("projected", protocol.PROJECTED_GRID, prepared.projected),
                               ("unprojected", protocol.UNPROJECTED_GRID, prepared.unprojected)):
            for dose in grid:
                result = fn(dose)
                self.assertEqual(list(result["updates"]), [name])
                self.assertEqual(result["dose"], dose)
                self.assertGreaterEqual(result["attestations"][name]["realized_to_planned"], 0.999)
                self.assertLessEqual(result["attestations"][name]["realized_to_planned"], 1.001)
        for dose in protocol.PROJECTED_GRID:
            self.assertEqual(set(prepared.conditions(dose)["updates"]), set(prior.CONDITIONS))

    def test_copy_inputs_and_do_not_reuse_mutated_returned_updates(self):
        args = fixture()
        prepared = subject.prepare(*args)
        initial = prepared.conditions(0.004)
        hashes = {key: value["intended_delta_sha256"] for key, value in initial["attestations"].items()}
        for tensor in args:
            tensor.zero_()
        for tensor in initial["updates"].values():
            tensor.zero_()
        again = prepared.conditions(0.004)
        self.assertEqual(hashes, {key: value["intended_delta_sha256"]
                                  for key, value in again["attestations"].items()})

    def test_internal_mutations_are_detected_before_construction(self):
        prepared = subject.prepare(*fixture())
        prepared._base["factors"]["projected_input"][0] += 1
        with self.assertRaises(prior.ProjectionError):
            prepared.projected(0.004)
        prepared = subject.prepare(*fixture())
        prepared._inputs[0][0, 0] += 1
        with self.assertRaises(prior.ProjectionError):
            prepared.unprojected(0.004)
        prepared = subject.prepare(*fixture())
        prepared._base["attestation"]["rank"] = -1
        with self.assertRaises(prior.ProjectionError):
            prepared.conditions(0.004)

    def test_wrong_doses_types_shapes_nonfinite_and_collapsed_projection_fail(self):
        prepared = subject.prepare(*fixture())
        for bad in (True, 0, 0.003, float("nan"), float("inf")):
            with self.subTest(dose=bad):
                with self.assertRaises(prior.ProjectionError):
                    prepared.projected(bad)
        with self.assertRaises(prior.ProjectionError):
            prepared.unprojected(0.0125)
        args = list(fixture())
        args[2] = args[2].double()
        with self.assertRaises(prior.ProjectionError):
            subject.prepare(*args)
        args = list(fixture())
        args[0][0, 0] = float("nan")
        with self.assertRaises(prior.ProjectionError):
            subject.prepare(*args)
        args = list(fixture())
        args[2] = torch.tensor([1., 0., 0., 0., 0., 0.])
        args[4] = torch.outer(args[2], args[3])
        with self.assertRaises(prior.ProjectionError):
            subject.prepare(*args)

    def test_flush_mode_is_rejected_and_never_silently_reset(self):
        prepared = subject.prepare(*fixture())
        try:
            self.assertTrue(torch.set_flush_denormal(True))
            with self.assertRaises((prior.ProjectionError, protocol.StudyError)):
                prepared.projected(0.004)
            with self.assertRaises((prior.ProjectionError, protocol.StudyError)):
                subject.prepare(*fixture())
            with self.assertRaises((prior.ProjectionError, protocol.StudyError)):
                subject._probe()
        finally:
            torch.set_flush_denormal(False)

if __name__ == "__main__":
    unittest.main()
