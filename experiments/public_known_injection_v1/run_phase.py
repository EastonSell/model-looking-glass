"""One offline phase of the public known-injection study (direct child only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


CASE_SHA = 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789'
PROTOCOL_SHA = '9eeca112d06f998037bea9d5a7a6b7097307f76634565441c62b24e5f2855cfd'
ADDENDUM_SHA = '68e12b08516a97c40c4084ac0ff581b8312afa507daf49bb29fc2928622fa8fa'
RESERVATION_SHA = 'e5ae202c1e3449a67e92788a83cd0dc62d404a3fa1e052716a11770ee92cc00c'
SOURCE_PATHS = (
    'experiments/public_known_injection_v1/run_phase.py',
    'experiments/public_known_injection_v1/executor.py',
    'experiments/public_known_injection_v1/hook_core.py',
    'experiments/public_known_injection_v1/journal.py',
    'experiments/public_known_injection_v1/selector.py',
    'experiments/public_known_injection_v1/analyze.py',
    'experiments/public_known_injection_v1/admission.py',
    'looking_glass/__init__.py', 'looking_glass/pretrained.py',
    'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
    'looking_glass/model_ownership.py', 'looking_glass/transformer_trace.py',
    'looking_glass/trace_schema.py', 'looking_glass/probes.py',
    'experiments/public_known_injection_cases_v1.json',
    'docs/PUBLIC_KNOWN_INJECTION_LOCALIZATION_V1.md',
    'docs/PUBLIC_KNOWN_INJECTION_DELIVERY_CLARIFICATION_V1.md',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('a fixed file digest differs')
    return data


def validate_startup(prefix_hash):
    if not sys.dont_write_bytecode or not sys.pycache_prefix:
        raise ValueError('requires -B and a fresh -X pycache_prefix')
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or prefix.is_symlink() or not prefix.is_dir() or any(prefix.iterdir()):
        raise ValueError('bytecode-cache prefix must be a fresh empty directory')
    identity = hashlib.sha256(os.path.normcase(str(prefix.resolve(strict=True))).encode()).hexdigest()
    if identity != prefix_hash:
        raise ValueError('bytecode-cache prefix differs from pre-run')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('discovery', 'held-out'), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--pre-run', type=Path, required=True)
    parser.add_argument('--pre-run-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--discovery-dir', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    pre = json.loads(checked_bytes(args.pre_run, args.pre_run_sha256))
    if (pre.get('schema') != 'public-known-injection-pre-run-v1'
            or pre.get('case_sha256') != CASE_SHA or pre.get('protocol_sha256') != PROTOCOL_SHA
            or pre.get('delivery_addendum_sha256') != ADDENDUM_SHA
            or pre.get('reservation_sha256') != RESERVATION_SHA):
        raise ValueError('unexpected pre-run pins')
    prefix = validate_startup(pre['pycache_prefix_sha256'][args.phase])
    root = args.source_root.resolve(strict=True)
    if Path(__file__).resolve() != root / SOURCE_PATHS[0]:
        raise ValueError('runner is not in the supplied source tree')
    hashes = pre['source_sha256']
    if set(hashes) != set(SOURCE_PATHS):
        raise ValueError('source path inventory differs')
    for name in SOURCE_PATHS:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != hashes[name]:
            raise ValueError('source bytes differ')
    case_path = root / 'experiments/public_known_injection_cases_v1.json'
    cases = json.loads(checked_bytes(case_path, CASE_SHA))['cases']
    checked_bytes(root / 'docs/PUBLIC_KNOWN_INJECTION_LOCALIZATION_V1.md', PROTOCOL_SHA)
    checked_bytes(root / 'docs/PUBLIC_KNOWN_INJECTION_DELIVERY_CLARIFICATION_V1.md', ADDENDUM_SHA)
    if (len(cases) != 16 or [c['case_id'] for c in cases] != [f'known-injection-{i:02d}' for i in range(1, 17)]
            or any(c['split'] != ('discovery' if i < 8 else 'held-out') or c['candidate_token_ids'] != [318, 389]
                   for i, c in enumerate(cases))):
        raise ValueError('fixed case order or split differs')
    if args.output_dir.exists() or not args.output_dir.parent.is_dir():
        raise ValueError('output must be a fresh path under an existing attempt')
    if shutil.disk_usage(args.output_dir.parent).free < 2 * 1024 ** 3:
        raise ValueError('insufficient disk for checkpoint staging and bounded evidence')
    if not args.cache_dir.is_dir() or not args.inventory.is_file():
        raise ValueError('offline checkpoint cache or inventory is unavailable')
    # Only after checking a fresh prefix and all source bytes, import study code.
    sys.path.insert(0, str(root))
    import admission as admission_module
    if Path(admission_module.__file__).resolve() != root / 'experiments/public_known_injection_v1/admission.py':
        raise ValueError('admission imported from an unexpected path')
    admission = admission_module.verify_inventory(args.inventory, RESERVATION_SHA, cases)
    if args.phase == 'held-out':
        if args.discovery_dir is None or args.freeze is None or not args.freeze_sha256:
            raise ValueError('held-out phase requires the separately published discovery freeze')
        frozen = json.loads(checked_bytes(args.freeze, args.freeze_sha256))
        import analyze as analysis_module
        if Path(analysis_module.__file__).resolve() != root / 'experiments/public_known_injection_v1/analyze.py':
            raise ValueError('analysis imported from an unexpected path')
        records, evidence = analysis_module.load_phase(args.discovery_dir, cases[:8])
        if (frozen.get('schema') != 'public-known-injection-discovery-freeze-v1'
                or frozen.get('case_sha256') != CASE_SHA or frozen.get('pre_run_sha256') != args.pre_run_sha256
                or frozen.get('reservation_sha256') != RESERVATION_SHA
                or frozen.get('evidence') != evidence
                or frozen.get('discovery') != analysis_module.discovery(records, cases[:8])):
            raise ValueError('held-out discovery freeze does not match saved evidence')
    elif args.freeze is not None or args.discovery_dir is not None or args.freeze_sha256 is not None:
        raise ValueError('discovery phase cannot consume held-out or freeze data')
    import torch
    import looking_glass.pretrained as pretrained_module
    import executor as execution_module
    if (Path(pretrained_module.__file__).resolve() != root / 'looking_glass/pretrained.py'
            or Path(execution_module.__file__).resolve() != root / 'experiments/public_known_injection_v1/executor.py'):
        raise ValueError('model loader or executor imported from an unexpected path')
    for name in ('admission', 'analyze', 'executor', 'hook_core', 'journal', 'selector'):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != root / f'experiments/public_known_injection_v1/{name}.py':
            raise ValueError('study dependency imported from an unexpected path')
    for name in ('pretrained', 'intervention_plan', 'interventions', 'model_ownership',
                 'transformer_trace', 'trace_schema', 'probes'):
        module = sys.modules.get('looking_glass.' + name)
        if module is not None and Path(module.__file__).resolve() != root / f'looking_glass/{name}.py':
            raise ValueError('model dependency imported from an unexpected path')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('could not disable CPU subnormal flushing')
    torch.manual_seed(0)
    loaded = pretrained_module.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
    if (loaded.metadata['model_revision'] != '2290a62682d06624634c1f46a6ad5be0f47f38aa'
            or loaded.metadata['weights_sha256'] != 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
            or loaded.metadata['asset_sha256']['tokenizer.json'] != '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'):
        raise ValueError('loaded checkpoint identity differs')
    for case in cases:
        ids, text = case['token_ids'], case['public_text']
        if loaded.tokenizer.encode(text, add_special_tokens=False) != ids:
            raise ValueError('loaded tokenizer disagrees with fixed cases')
        if any(loaded.tokenizer.encode(text + suffix, add_special_tokens=False) != ids + [token]
               for token, suffix in ((318, ' is'), (389, ' are'))):
            raise ValueError('loaded tokenizer continuation differs')
    chosen = cases[:8] if args.phase == 'discovery' else cases[8:]
    result = execution_module.execute_phase(loaded.model, chosen, args.output_dir, phase=args.phase)
    if result['completed_actions'] != 120 or any(prefix.iterdir()):
        raise RuntimeError('phase action count or bytecode prefix differs')
    print(json.dumps({'event': 'phase-complete', 'phase': args.phase,
                      'actions': result['completed_actions'], 'raw_bytes': result['raw_bytes'],
                      'admission': admission}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
