"""Exercise the BERT eight-family journal auditor without model execution.

This constructs synthetic evaluation-shaped records by relabeling an existing
development journal. They are test data, never held-out observations or
evidence that the evaluation was run.
"""
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import importlib.util

repo = Path(__file__).resolve().parents[1]

from looking_glass.bert_agreement_cases import build_manifest
from looking_glass.bert_agreement_study import audit_event_journal


def key(row):
    return row['subject_source'], row['subject_number'], row['distractor_number']


def run():
    manifest = build_manifest()
    dev = [r for r in manifest['rows'] if r['split'] == 'development' and r['family'] == 1]
    dev_cases = [r for r in manifest['targets'] if r['split'] == 'development' and r['family'] == 1]
    journal = repo / 'results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz'
    with gzip.open(journal, 'rb') as stream:
        raw = stream.read()
    expected = 'ff14574bf1b35058b3cff51e4752180ec589982cf3a547baba318a9591af8b67'
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('development source journal digest differs')
    events = [json.loads(line) for line in raw.splitlines()]
    families = []
    for number in range(1, 9):
        rows = [r for r in manifest['rows'] if r['split'] == 'evaluation' and r['family'] == number]
        targets = [r for r in manifest['targets'] if r['split'] == 'evaluation' and r['family'] == number]
        by_key = {key(r): r for r in rows}
        row_map = {r['row_id']: by_key[key(r)]['row_id'] for r in dev}
        target_map = {key(next(r for r in rows if r['row_id'] == c['target_row_id'])): c for c in targets}
        case_map = {c['case_id']: target_map[key(next(r for r in dev if r['row_id'] == c['target_row_id']))]['case_id']
                    for c in dev_cases}
        by_id = {r['row_id']: r for r in rows}
        group = []
        for original in events:
            event = dict(original, split='evaluation', family=number)
            for field in ('row_id', 'donor_row_id'):
                if event.get(field) is not None:
                    event[field] = row_map[event[field]]
            if event.get('case_id') is not None:
                event['case_id'] = row_map.get(event['case_id'], case_map.get(event['case_id'], event['case_id']))
            if 'model_input_ids' in event:
                row = by_id[event['row_id']]
                event['model_input_ids'], event['model_attention_mask'] = row['input_ids'], row['attention_mask']
            group.append(event)
        families.append(group)

    def write(path, groups, mutate=None):
        changed = False
        with path.open('wb') as stream:
            for group in groups:
                for original in group:
                    event = dict(original)
                    if mutate and not changed and mutate(event):
                        changed = True
                    if not event.pop('_skip_test_record', False):
                        stream.write(json.dumps(event, sort_keys=True, separators=(',', ':'),
                                                ensure_ascii=True, allow_nan=False).encode('ascii') + b'\n')
        if mutate and not changed:
            raise AssertionError('the synthetic corruption did not occur')

    def audit(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return audit_event_journal(manifest, path, split='evaluation', family=None, mask_id=103,
                                   hidden_size=768, expected_sha256=digest)

    def skip_last(event):
        if event['family'] == 8 and event.get('forward_index') == 140 and event['status'] == 'completed':
            event['_skip_test_record'] = True
            return True
        return False

    def wrong_landing(event):
        if event['family'] == 8 and event['status'] == 'completed' and event.get('kind') == 'patch_B_opposite':
            event['after_float32_le_base64'] = base64.b64encode(bytes(768 * 4)).decode('ascii')
            return True
        return False

    with tempfile.TemporaryDirectory(prefix='bert-synthetic-audit-') as temporary:
        path = Path(temporary) / 'synthetic-events.jsonl'
        write(path, families)
        observed = audit(path)
        assert len(observed['families']) == 8
        assert all(f['forward_count'] == 140 for f in observed['families'])
        assert observed['summary']['baseline']['n'] == 32
        assert sum(c['n'] for c in observed['summary']['baseline']['cells'].values()) == 32

        def decode(bits):
            return struct.unpack('<f', struct.pack('<I', bits))[0]

        cells = {}
        for event in families[0]:
            if event['status'] == 'completed' and event['site'] == 'candidate' and event['kind'].startswith('patch_'):
                margin = math.fsum((decode(event['is_bits']), -decode(event['are_bits'])))
                cells.setdefault(event['row_id'], {})[event['kind'][6:]] = margin
        rows = {r['row_id']: r for r in manifest['rows']}
        effects = []
        for result in observed['families'][0]['cases']:
            cell = cells[result['target_row_id']]
            sign = 1 if rows[result['target_row_id']]['subject_number'] == 'singular' else -1
            delta_a = math.fsum((cell['A_opposite'], -cell['A_same']))
            delta_b = math.fsum((cell['B_opposite'], -cell['B_same']))
            effect = -sign * (math.fsum((delta_a, delta_b)) / 2)
            assert effect == result['effects']['candidate']
            effects.append(effect)
        assert math.fsum(effects) / 4 == observed['summary']['family_means']['candidate'][0]
        print('Synthetic journal accepted: eight families, 1,120 forward records, 32 targets.')

        corruptions = (
            ('missing final forward', families, skip_last),
            ('wrong last-family donor landing', families, wrong_landing),
        )
        for name, groups, mutation in corruptions:
            write(path, groups, mutation)
            try:
                audit(path)
            except (ValueError, RuntimeError, KeyError):
                print('Rejected:', name)
            else:
                raise AssertionError('synthetic corruption was accepted: ' + name)


class SyntheticEightFamilyJournalTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is optional')
    def test_full_schedule_and_late_corruptions(self):
        run()


if __name__ == '__main__':
    unittest.main()
