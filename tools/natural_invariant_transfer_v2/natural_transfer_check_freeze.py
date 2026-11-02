import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import natural_transfer_freeze as f
from natural_transfer_check_executor import prepared


def sha(raw): return hashlib.sha256(raw).hexdigest()
def gitblob(raw): return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


class FreezeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['NATURAL_FAKE_FIXTURE'])
        cls.mapping,cls.prior,_=prepared()
        cls.meta=json.loads((cls.root/'fixture.json').read_bytes())
        cls.raw=(cls.root/'freeze.json').read_bytes()
        cls.binding={k:'1'*64 for k in ('pre_run_sha256','prior_receipt_sha256','latest_exposure_sha256')}
        cls.binding['mapping_sha256']=cls.meta['mapping_sha256']

    def kwargs(self):
        return dict(mapping_path=self.root/'toy-mapping.json', expected_mapping_sha256=self.meta['mapping_sha256'],
                    prior=self.prior,prior_events_sha256=self.meta['prior_events_sha256'],
                    discovery_directory=self.root/'discovery',
                    discovery_events_sha256=self.meta['phases']['discovery']['events_sha256'],
                    discovery_terminal_sha256=self.meta['phases']['discovery']['terminal_sha256'],
                    discovery_binding=dict(self.binding),freeze_path=self.root/'freeze.json',
                    expected_freeze_sha256=sha(self.raw),observed_public_commit='a'*40,
                    observed_public_blob=gitblob(self.raw),width=8,vocabulary=1500)

    def test_recomputes_the_exact_frozen_predictions(self):
        result=f.verify_before_evaluation(**self.kwargs())
        self.assertEqual(result['freeze_sha256'],self.meta['freeze_sha256'])

    def test_rehashed_wrong_selection_still_rejects(self):
        data=json.loads(self.raw);data['selected']='Q4'
        raw=(json.dumps(data,sort_keys=True,indent=2)+'\n').encode()
        with tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP']) as tmp:
            path=Path(tmp)/'altered.json';path.write_bytes(raw)
            args=self.kwargs();args.update(freeze_path=path,expected_freeze_sha256=sha(raw),observed_public_blob=gitblob(raw))
            with self.assertRaises(ValueError): f.verify_before_evaluation(**args)

    def test_wrong_public_blob_or_prior_binding_rejects(self):
        args=self.kwargs();args['observed_public_blob']='b'*40
        with self.assertRaises(ValueError):f.verify_before_evaluation(**args)
        args=self.kwargs();args['discovery_binding']['mapping_sha256']='2'*64
        with self.assertRaises(ValueError):f.verify_before_evaluation(**args)

    def test_module_import_is_model_free(self):
        src="import sys; import natural_transfer_freeze; assert 'torch' not in sys.modules"
        proc=subprocess.run([sys.executable,'-B','-c',src],cwd=str(Path(f.__file__).parent),capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)


if __name__=='__main__':unittest.main()
