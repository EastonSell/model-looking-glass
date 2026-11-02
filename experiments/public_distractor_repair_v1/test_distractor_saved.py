"""Model-free saved-record and coherent-tamper checks on random data."""
import hashlib
import os
import subprocess
import sys
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
import distractor_saved as saved

ROOT=Path(tempfile.gettempdir())
FIXTURE=Path(os.environ.get('PUBLIC_DISTRACTOR_FIXTURE', str(Path(__file__).resolve().parent/'fake-fixture-v3')))
CASES=json.loads((FIXTURE/'cases.json').read_bytes())['cases']

class SavedTests(unittest.TestCase):
    def test_fixture_and_fixed_decisions(self):
        records,evidence=saved.load_phase(FIXTURE/'discovery',CASES[:8],vocab=400,hidden=8)
        self.assertEqual(evidence['actions'],224)
        frozen=saved.discovery(records,CASES[:8])
        self.assertEqual([frozen[str(x)]['random'] for x in (1,2,3)],['Q1','Q1','Q4'])
        held,he=saved.load_phase(FIXTURE/'held-out',CASES[8:],vocab=400,hidden=8)
        self.assertEqual(he['actions'],224)
        self.assertIsInstance(saved.heldout(held,CASES[8:],frozen,vocab=400)['primary']['supported'],bool)

    def test_changed_hook_or_false_control_is_rejected(self):
        for mutation in ('hook','check','raw_identity','complete_identity'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT,prefix='saved-fake-') as tmp:
                run=Path(tmp)/'run';shutil.copytree(FIXTURE/'discovery',run)
                path=run/'events.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
                if mutation=='hook':
                    next(x for x in rows if x['type']=='raw-recorded' and x['action']=='combined')['hook_events'][0]['module_path']='wrong.module'
                elif mutation=='check':
                    next(x for x in rows if x['type']=='complete' and x['action']=='decoy')['checks']['scored_logits_equal_clean']=False
                elif mutation=='raw_identity':
                    next(x for x in rows if x['type']=='raw-recorded' and x['action']=='decoy')['case_id']='distractor-repair-99'
                else:
                    next(x for x in rows if x['type']=='complete' and x['action']=='decoy')['action']='target'
                path.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows),encoding='utf-8')
                with self.assertRaises(ValueError):saved.load_phase(run,CASES[:8],vocab=400,hidden=8)

    def test_tensor_after_raw_recorded_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='saved-fake-') as tmp:
            run=Path(tmp)/'run';shutil.copytree(FIXTURE/'discovery',run)
            path=run/'events.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            index=next(i for i,x in enumerate(rows) if x['type']=='raw-recorded' and x['action']=='clean')
            moved=rows.pop(index-1);index-=1
            rows[index]['tensors'].pop()
            rows.insert(index+1,moved)
            path.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows),encoding='utf-8')
            with self.assertRaises(ValueError):saved.load_phase(run,CASES[:8],vocab=400,hidden=8)

    @unittest.skipUnless(os.name=='nt', 'Windows FP-mode regression')
    def test_reader_rejects_all_non_nearest_modes(self):
        code = r"""
import ctypes,json,os
from pathlib import Path
import distractor_saved as saved
root=Path(os.environ.get('PUBLIC_DISTRACTOR_FIXTURE', str(Path(saved.__file__).resolve().parent/'fake-fixture-v3')))
cases=json.loads((root/'cases.json').read_bytes())['cases'][:8]
control=ctypes.CDLL('msvcrt')._controlfp
control.argtypes=[ctypes.c_uint,ctypes.c_uint];control.restype=ctypes.c_uint
old=control(0,0)
try:
 for mode in (0x100,0x200,0x300):
  control(mode,0x300)
  try:saved.load_phase(root/'discovery',cases,vocab=400,hidden=8)
  except RuntimeError as error:assert 'round' in str(error).lower()
  else:raise AssertionError('reader accepted non-nearest mode')
finally:control(old,0x300)
"""
        child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(child.returncode,0,child.stderr)

    def test_coherently_rehashed_wrong_rotation_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='saved-fake-') as tmp:
            run=Path(tmp)/'run';shutil.copytree(FIXTURE/'discovery',run)
            path=run/'events.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            event=next(x for x in rows if x['type']=='tensor' and x['tensor']['name']=='rotation_planned')
            d=event['tensor'];rawpath=run/d['file'];data=bytearray(rawpath.read_bytes())
            value=struct.unpack_from('<f',data,d['offset'])[0]
            struct.pack_into('<f',data,d['offset'],value+0.125)
            rawpath.write_bytes(data)
            newhash=hashlib.sha256(data[d['offset']:d['offset']+d['length']]).hexdigest()
            d['sha256']=newhash
            recorded=next(x for x in rows if x['type']=='raw-recorded' and x['ordinal']==event['ordinal'])
            next(x for x in recorded['tensors'] if x['name']=='rotation_planned')['sha256']=newhash
            path.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows),encoding='utf-8')
            with self.assertRaises(ValueError):saved.load_phase(run,CASES[:8],vocab=400,hidden=8)

if __name__=='__main__':unittest.main()
