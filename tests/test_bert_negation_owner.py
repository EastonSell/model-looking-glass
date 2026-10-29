import importlib.util
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys
import zlib

import pytest


OWNER = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "bert_negation_cue_v1.py"


def load_owner():
    spec = importlib.util.spec_from_file_location("negation_owner_for_test", OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_load_torch_or_transformers():
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('owner', {str(OWNER)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_asset_preflight_rejects_wrong_bytes_and_missing_files(tmp_path, monkeypatch):
    owner = load_owner()
    payload = b"fixture"
    monkeypatch.setattr(owner, "ASSETS", {"asset.bin": (len(payload), hashlib.sha256(payload).hexdigest())})
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.verify_assets(tmp_path)
    (tmp_path / "asset.bin").write_bytes(b"different")
    with pytest.raises(ValueError):
        owner.verify_assets(tmp_path)
    (tmp_path / "asset.bin").write_bytes(payload)
    assert owner.verify_assets(tmp_path)["asset.bin"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_create_only_json_does_not_overwrite_existing_file(tmp_path):
    owner = load_owner()
    path = tmp_path / "result.json"
    owner.write_json_new(path, {"first": 1})
    with pytest.raises(FileExistsError):
        owner.write_json_new(path, {"second": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"first": 1}


def test_each_event_is_recoverable_before_gzip_footer(tmp_path):
    owner = load_owner()
    path = tmp_path / "partial.gz"
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            owner.write_event(zipped, raw, {"kind": "clean", "n": 1})
            decompressor = zlib.decompressobj(wbits=31)
            recovered = decompressor.decompress(path.read_bytes())
            assert json.loads(recovered) == {"kind": "clean", "n": 1}
            assert not decompressor.eof


def test_model_digest_covers_nonpersistent_buffers():
    import torch
    owner = load_owner()
    model = torch.nn.Module()
    model.register_buffer("position_ids", torch.tensor([0, 1]), persistent=False)
    before = owner.state_hash(model)
    model.position_ids[1] = 3
    assert owner.state_hash(model) != before


def test_no_flush_setting_must_be_accepted():
    owner = load_owner()
    class RejectingRuntime:
        def set_flush_denormal(self, enabled):
            assert enabled is False
            return False
    with pytest.raises(RuntimeError, match="denormal"):
        owner.enable_no_flush(RejectingRuntime())
