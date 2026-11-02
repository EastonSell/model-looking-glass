"""Exercise the saved schema with invented tensors; never load a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from scripts import copying_head_output_dose_v1 as owner
from looking_glass.copy_head_factorial import _values, head_slice
from looking_glass.copy_head_output_dose import HEADS, project, weights


def pack(values):
    return struct.pack('<%df' % len(values), *values)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--underfill', action='store_true')
    args = parser.parse_args()
    assert args.cases.is_absolute() and args.output.is_absolute() and not args.output.exists()
    case_raw = args.cases.read_bytes()
    assert sha(case_raw) == owner.CASES_SHA
    cases = json.loads(case_raw)['cases']
    by_input = {tuple(case['token_ids']): case for case in cases}
    assert len(by_input) == 32
    w = [0.0] * (64 * 768)
    w[0], w[1], w[768] = 1.0, -2.0, 17.0
    one_weight = pack(w)
    weight_raw = {h: one_weight for h in HEADS}
    weight_bytes = b''.join(weight_raw[h] for h in HEADS)
    matrix = weights(weight_raw)
    args.output.mkdir()
    (args.output / 'projection-weights.f32').write_bytes(weight_bytes)
    head_hashes = {str(h): sha(weight_raw[h]) for h in HEADS}
    counts = {'submitted': 0, 'model_returns': 0, 'returned': 0, 'retained': 0}
    activity, completed, families = {}, [], []
    runtime = {'fabricated': True}

    def fake_forward(_model, ids, action, *, expected_before=None, deliveries=None, observation=None):
        case = by_input[tuple(ids.tolist()[0])]
        source = [0.0] * (11 * 768)
        for h, value in ((2, 1.0), (10, 2.0), (7, 4.0)):
            index = 2 if args.underfill and case['group'] <= 3 and h == 2 else 0
            source[10 * 768 + 64 * h + index] = value
        before = pack(source)
        effect = 0.0
        after = before
        if action.startswith('h'):
            head = int(action[1:])
            assert expected_before == before and set(deliveries) == {head}
            after_array = bytearray(before)
            start = (10 * 768 + 64 * head) * 4
            after_array[start:start + 256] = deliveries[head]
            after = bytes(after_array)
            effect = {2: -1.0, 10: -0.5, 7: -0.1}[head] * (1.0 if case['variant'] == 0 else 0.1)
        logits = bytearray(50_257 * 4)
        struct.pack_into('<f', logits, case['target_id'] * 4, 1.0 + effect)
        payload = {'logits': bytes(logits)}
        if action not in ('clean', 'clean_after'):
            a = _values(after)
            contributions = [project(a[10 * 768 + 64*h:10 * 768 + 64*(h+1)], matrix[h]) for h in HEADS]
            final = [math.fsum(row[j] for row in contributions) for j in range(768)]
            payload.update({'cproj_before': before, 'cproj_after': after,
                            'cproj_output': pack([0.0] * (10 * 768) + final)})
        observation.update({'model_call_return_observed': True, 'payload': payload})
        return payload

    owner.prior.runtime = lambda _model: runtime
    owner.prior.run_forward = fake_forward
    with (args.output / 'attempt.jsonl').open('xb') as stream:
        log = owner.BoundedJournal(stream)
        owner.base._append_event(log, {'event': 'runtime_ready', 'actual_runtime': runtime,
            'weight_order': list(HEADS), 'weight_sha256': sha(weight_bytes), 'weight_head_sha256': head_hashes})
        def perform(case, kind, cache, head=None, delivery=None):
            return owner.perform_action(case, kind, cache, head, delivery, model=None, runtime_before=runtime,
                matrix=matrix, output=args.output, log=log, counts=counts, activity=activity, completed=completed)
        for group in range(8):
            families.append(owner.execute_family(cases[4*group:4*group+4], perform, lambda: None, log, weight_raw))
    dummy_hash = '0' * 64
    result = {'schema': 'distilgpt2-copy-output-dose-result-v1', 'status': 'complete',
        'synthetic_fixture': True, 'manifest_sha256': None, 'checkpoint': None,
        'weight_sha256': sha(weight_bytes), 'weight_head_sha256': head_hashes,
        'state_sha256_before': dummy_hash, 'state_sha256_after': dummy_hash,
        'buffers_sha256_before': dummy_hash, 'buffers_sha256_after': dummy_hash,
        'runtime': runtime, 'guard_safe_queries': {}, 'submitted_forwards': counts['submitted'],
        'model_call_returns_observed': counts['model_returns'], 'returned_forwards': counts['returned'],
        'raw_payload_bytes': counts['retained'], 'records': completed, 'families': families,
        'decisions': {label: owner.old.decision([f['summaries'][label] for f in families])
                      for label in ('output', 'input')}}
    owner.bounded_json(args.output / 'result.json', result, owner.MAX_RESULT_BYTES)
    print(json.dumps({'synthetic_fixture': True, 'journal_sha256': sha((args.output/'attempt.jsonl').read_bytes()),
        'result_sha256': sha((args.output/'result.json').read_bytes()), 'weight_sha256': sha(weight_bytes),
        'counts': counts, 'decisions': result['decisions']}))


if __name__ == '__main__':
    main()
