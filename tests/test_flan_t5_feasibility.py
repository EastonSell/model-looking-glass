"""Synthetic end-to-end exercise; does not load pretrained weights."""
import json
import os
import concurrent.futures
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent


def fake_child(output, scenario):
    sys.path.insert(0, str(REPO))
    import torch
    import transformers
    from types import SimpleNamespace
    from unittest.mock import patch
    from contextlib import nullcontext
    from looking_glass import flan_t5_feasibility as runner

    cases = json.loads(runner.CASES.read_bytes())
    rows = [cases["neutral"], *(x for p in cases["pairs"] for x in p["sources"] + p["targets"])]
    lookup = {r["text"]: r for r in rows}

    class T5TokenizerFast:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kwargs):
            row = lookup[text]
            return {"input_ids": row["input_ids"], "attention_mask": row.get("attention_mask", [1] * len(row["input_ids"])),
                    "special_tokens_mask": row.get("special_tokens_mask", [0] * (len(row["input_ids"]) - 1) + [1]),
                    "offset_mapping": [tuple(x) for x in row["offsets"]]}

    if scenario == "bounds":
        short = cases["pairs"][0]["sources"][0]
        short["input_ids"] = short["input_ids"][:5] + [1]
        short["attention_mask"] = [1] * 6
        short["special_tokens_mask"] = [0] * 5 + [1]
        short["offsets"] = short["offsets"][:5] + [[0, 0]]
        result = runner.check_tokenizer(T5TokenizerFast(), cases)
        assert result[0] == ["source-token-alignment"] and result[1] == []
        evidence = runner.Evidence(Path(output))
        class ShortWriter:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def write(self, value): return 1
        with patch.object(Path, "open", return_value=ShortWriter()):
            try:
                evidence.write("short.bin", b"abcd")
                raise AssertionError("short write was accepted")
            except OSError:
                pass
        print("synthetic eligibility/short-write PASS")
        return

    class T5ForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.encoder = torch.nn.Module()
            self.encoder.final_layer_norm = torch.nn.Identity()
            self.calls = 0
            self.config = SimpleNamespace(d_model=512, vocab_size=32128, decoder_start_token_id=0,
                                          eos_token_id=1, pad_token_id=0, use_cache=False)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

        def forward(self, input_ids, attention_mask, decoder_input_ids, **kwargs):
            self.calls += 1
            state = torch.zeros((1, input_ids.shape[1], 512), dtype=torch.float32)
            token = int(input_ids[0, 6])
            signal_value = 1.0 if token in (1712, 484) else -1.0 if token in (1782, 629) else 0.0
            if scenario == "unsupported":
                signal_value = 0.0
            state[0, 6, 0] = signal_value
            if scenario == "negative":
                state[0, 0, 0] = signal_value
            state = self.encoder.final_layer_norm(state)
            signal = state[0, 6, 0] + (state[0, 0, 0] if scenario == "negative" else 0)
            logits = torch.full((1, decoder_input_ids.shape[1], 32128), -2.0)
            for target in (16699, 4675):
                logits[0, 0, target] = signal * 20
            for target in (14587, 4598):
                logits[0, 0, target] = -signal * 20
            if scenario == "nonfinite" and self.calls == 1:
                logits[0, 0, 0] = float("nan")
            if scenario == "bad_control" and self.calls == 7:
                logits[0, 0, 0] += 1
            if scenario == "bad_capture" and self.calls == 7:
                state = state.clone()
                state[0, 0, 1] = 1
            if scenario == "non_tensor" and self.calls == 1:
                logits = "not-a-tensor"
            return SimpleNamespace(logits=logits, encoder_last_hidden_state=state)

    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    real_t5 = transformers.T5ForConditionalGeneration
    with patch.object(runner, "source_identity", return_value={"commit": "synthetic", "source_sha256": {}}), \
         patch.object(runner, "verify_assets", return_value={"synthetic": True}), \
         patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=T5TokenizerFast()), \
         patch.object(real_t5, "from_pretrained", return_value=(T5ForConditionalGeneration(), {})), \
         (patch.object(torch, "log_softmax", side_effect=RuntimeError("synthetic scoring failure"))
          if scenario == "bad_score" else nullcontext()):
        if scenario in ("bad_control", "bad_capture", "nonfinite", "non_tensor", "bad_score"):
            try:
                runner.run(Path(output).parent, Path(output))
                raise AssertionError("fault scenario did not fail")
            except (ValueError, RuntimeError):
                failed = json.loads((Path(output) / "failure.json").read_bytes())
                assert failed["status"] == "incomplete" and failed["failed_forward"]["name"]
                if scenario == "bad_score":
                    info = json.loads((Path(output) / "forward-000-scoring-failed.json").read_bytes())
                    assert failed["failed_forward"]["phase"] == "scoring" and (Path(output) / info["file"]).exists()
                elif scenario in ("nonfinite", "non_tensor"):
                    info = json.loads((Path(output) / "forward-000-failed.json").read_bytes())
                    assert info["encoder_available"] and (Path(output) / info["file"]).exists()
                    assert info["logits_available"] == (scenario == "nonfinite")
                else:
                    assert failed["failed_forward"]["phase"] == "control-check"
                    assert (Path(output) / "forward-006.bin").exists()
            print("synthetic failure path PASS")
            return
        result = runner.run(Path(output).parent, Path(output))
    assert result["status"] == "complete"
    if scenario == "unsupported":
        assert result["forwards"] == 12
        assert [x["status"] for x in result["pairs"]] == ["baseline-unsupported"] * 2
    else:
        assert result["forwards"] == 76
        expected = "negative" if scenario == "negative" else "positive"
        assert [x["status"] for x in result["pairs"]] == [expected] * 2
        if scenario == "positive":
            assert all(x["instruction_movements"] == [0.0, 0.0] for x in result["pairs"])
    assert sum(p.stat().st_size for p in Path(output).iterdir()) < runner.MAX_BYTES
    print("synthetic end-to-end PASS: 76 forwards, identity/swap/landing controls, raw evidence")


class FlanT5FeasibilityTests(unittest.TestCase):
    def test_full_synthetic_control_path(self):
        workspace = REPO.parent.parent
        scenarios = ("positive", "negative", "unsupported", "bad_control", "bad_capture", "nonfinite", "non_tensor", "bad_score", "bounds")
        def execute(scenario):
            with tempfile.TemporaryDirectory(prefix="t5-fake-test-", dir=workspace) as root:
                out = Path(root).resolve() / "output"
                if not out.is_relative_to(workspace.resolve()):
                    raise AssertionError("temporary output escaped workspace")
                result = subprocess.run([sys.executable, "-B", __file__, "--fake-child", str(out), scenario],
                                        cwd=REPO, capture_output=True, text=True, timeout=180)
                return result.returncode, result.stdout, result.stderr
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(execute, scenarios))
        for scenario, (code, stdout, stderr) in zip(scenarios, results):
            with self.subTest(scenario=scenario):
                self.assertEqual(code, 0, stdout + stderr)
                self.assertIn("PASS", stdout)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--fake-child":
        fake_child(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
