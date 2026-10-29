"""Observe one annotation-only occlusion preparation attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK / 'teams/vision_recognized_v4_result'
RUN = Path('E:/Codex/model-looking-glass/work/vision-occlusion-preparation-01')
LOCAL_REVISION = '403c3545b7f490843f9936895b45e4e2f5330f03'
PUBLIC_REVISION = 'ce111d175cb89bfa44f07d89011b079b6dd368ff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if shutil.disk_usage(WORK).free < 2 * 1024**3 or shutil.disk_usage(RUN.parent).free < 1024**3:
        raise RuntimeError('insufficient fresh source/cache or output headroom')
    RUN.mkdir(parents=False, exist_ok=False)
    cache, temporary = RUN / 'empty-bytecode-cache', RUN / 'temporary'
    cache.mkdir()
    temporary.mkdir()
    output = RUN / 'manifest.json'
    bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2],sys.argv[3]]); '
                 'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [str(Path(sys._base_executable)), '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
               '-c', bootstrap, str(WORK / '.venv/Lib/site-packages'), str(WORK / 'vision-coco-deps'),
               str(ROOT), str(ROOT / 'tools/prepare_vision_coco_occlusion.py'),
               '--assets', str(WORK / 'vision-coco-assets'), '--annotation-archive',
               str(WORK / 'vision-recognized-assets/annotations_trainval2017.zip'),
               '--output', str(output), '--selection-revision', LOCAL_REVISION]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TEMP=str(temporary), TMP=str(temporary), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    code, timed_out, error = None, False, None
    with (RUN / 'stdout.txt').open('xb') as stdout, (RUN / 'stderr.txt').open('xb') as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1800)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except Exception as exc:
            error = type(exc).__name__
    receipt = {'schema': 'vision-occlusion-v1-preparation-receipt', 'local_source_revision': LOCAL_REVISION,
               'public_source_revision': PUBLIC_REVISION, 'exit_code': code, 'timed_out': timed_out,
               'error_type': error, 'duration_seconds': time.monotonic() - start,
               'stdout_sha256': sha(RUN / 'stdout.txt'), 'stderr_sha256': sha(RUN / 'stderr.txt'),
               'manifest_sha256': sha(output) if output.exists() else None,
               'status': 'complete' if code == 0 and not timed_out and output.is_file() else 'incomplete',
               'model_forwards': 0, 'scope': 'cooperative preparation; local and public relevant source '
               'blobs were independently compared, not complete checkout or execution attestation'}
    raw = (json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (RUN / 'receipt.json').open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt))
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
