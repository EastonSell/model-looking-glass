"""Compare a model-free full-vocabulary screen replay with the owner's summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    raw = path.read_bytes()
    return raw, json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def compare(independent, owner, *, fabricated=False):
    if (type(independent) is not dict or type(owner) is not dict or
            owner.get('case_sha256') != independent.get('manifest_sha256') or
            owner.get('events_sha256') != independent.get('journal_sha256') or
            type(owner.get('starts')) is not int or owner['starts'] != 9):
        raise ValueError('summary source, journal or count differs')
    if fabricated:
        if owner.get('fabricated') is not True:
            raise ValueError('expected explicitly fabricated fixture')
    elif (owner.get('schema') != 'distilbert-color-stage-result-v1' or
          owner.get('status') != 'complete' or owner.get('fabricated') is not None):
        raise ValueError('expected a complete non-fabricated stage result')
    summary = owner.get('summary')
    if (type(summary) is not dict or summary.get('status') != independent.get('status') or
            summary.get('selected_object') != independent.get('selected_object') or
            type(summary.get('candidate_eligible')) is not dict or
            type(independent.get('candidate_eligible')) is not dict or
            set(summary['candidate_eligible']) != set(independent['candidate_eligible']) or
            any(type(value) is not bool or value is not independent['candidate_eligible'].get(key)
                for key, value in summary['candidate_eligible'].items())):
        raise ValueError('screen eligibility summary differs')
    owner_rows, independent_rows = summary.get('rows'), independent.get('rows')
    if type(owner_rows) is not list or type(independent_rows) is not list or len(owner_rows) != 9 or len(independent_rows) != 9:
        raise ValueError('screen must contain exactly nine summary rows')
    for saved, expected in zip(owner_rows, independent_rows):
        if (type(saved) is not dict or type(expected) is not dict or
                any(saved.get(key) != expected.get(key) for key in ('case_id', 'object')) or
                type(saved.get('eligible')) is not bool or saved['eligible'] is not expected.get('eligible') or
                (saved.get('unique_top') is not None and type(saved.get('unique_top')) is not int) or
                saved.get('unique_top') != expected.get('unique_top') or
                type(saved.get('top_tie_count')) is not int or saved['top_tie_count'] != len(expected['top_ids']) or
                type(saved.get('top_representative')) is not int or
                saved['top_representative'] != expected['top_representative'] or
                type(saved.get('ranks')) is not dict or type(saved.get('logits')) is not dict):
            raise ValueError('per-row owner ranking or eligibility differs')
        for label in ('factual', 'counterfactual', 'wrong_target'):
            token = expected['tokens'][label]
            token_id = str(token['id'])
            value = saved['logits'].get(token_id)
            if (type(saved['ranks'].get(token_id)) is not int or saved['ranks'][token_id] != token['rank'] or
                    type(value) is not float or not math.isfinite(value) or
                    struct.pack('<f', value) != struct.pack('<f', token['logit'])):
                raise ValueError('selected saved-logit bits or rank differ')
        if type(saved.get('factual_rank')) is not int or saved['factual_rank'] != expected['tokens']['factual']['rank']:
            raise ValueError('factual rank differs')
    return {'rows': 9, 'status': independent['status'], 'selected_object': independent['selected_object'],
            'candidate_eligible': independent['candidate_eligible'],
            'journal_sha256': independent['journal_sha256'],
            'scope': 'Saved full-vocabulary rank/selected-logit consistency, not execution attestation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fabricated-fixture', action='store_true')
    args = parser.parse_args()
    report_raw, report = read(args.independent)
    result_raw, result = read(args.result)
    checked = compare(report, result, fabricated=args.fabricated_fixture)
    checked['independent_report_sha256'] = hashlib.sha256(report_raw).hexdigest()
    checked['owner_result_sha256'] = hashlib.sha256(result_raw).hexdigest()
    payload = (json.dumps(checked, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as output:
        if output.write(payload) != len(payload):
            raise OSError('short review write')
    print('summary review SHA-256', hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    main()
