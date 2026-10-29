"""Independent, model-free review of the closed copying-study publication.

Streams original and staged bytes; checks both derived CSVs against saved source
records.  Writes only a compact report when an output path is supplied.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

BLOCK = 1024 * 1024
INDEX_SHA = 'ddb47b3daed177ac749db3ea40e5f87577c5e7c24486d117579ea1b8c2fe178c'
RESULT_SHA = 'd598b48d32ac7fbdaf8d60c2b09f48b17471aa356408eb8a2ff5891acbcb1f88'
AUDIT_SHA = '6fa551521aa2293a672e6319742806195a13762efb24bbf20c89488a41b808fb'
FAMILY_SHA = '7c0ac89910e0ffe584c2d436f03c1c77f48d69b39dbe864502f119afe0a8920b'
CASES_SHA = '96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(BLOCK):
            h.update(b)
    return h.hexdigest()


def getjson(path, expected=None):
    raw = path.read_bytes()
    if expected:
        require(hashlib.sha256(raw).hexdigest() == expected, f'hash mismatch: {path.name}')
    return json.loads(raw)


def audit_stream(package, original, index):
    stream = index['tensor_stream']
    chunks, files = stream['chunks'], stream['files']
    require(len(chunks) == 33 and len(files) == 96, 'unexpected stream inventory')
    require(len({x['path'] for x in chunks}) == 33 and len({x['path'] for x in files}) == 96,
            'duplicate stream members')
    require(stream['bytes'] == 154720512, 'unexpected stream size')
    position = 0
    for j, chunk in enumerate(chunks):
        require(chunk['path'] == f'raw/part-{j:03}.bin', 'unexpected part name')
        require(chunk['offset'] == position and type(chunk['bytes']) is int and chunk['bytes'] > 0,
                'invalid part layout')
        path = package / chunk['path']
        require(path.is_file() and path.stat().st_size == chunk['bytes'], 'part size')
        h, g = hashlib.sha256(), hashlib.sha1(b'blob ' + str(chunk['bytes']).encode() + b'\0')
        with path.open('rb') as f:
            while b := f.read(BLOCK):
                h.update(b)
                g.update(b)
        require(h.hexdigest() == chunk['sha256'] and g.hexdigest() == chunk['git_blob_sha1'],
                'part hash mismatch')
        position += chunk['bytes']
    require(position == stream['bytes'], 'chunk total mismatch')

    full = hashlib.sha256()
    file_offset = part_index = part_position = 0
    source_part = None
    try:
        for member in files:
            name = member['path']
            require(Path(name).name == name and name.endswith('.f32'), 'unsafe file name')
            require(member['offset'] == file_offset and member['bytes'] == 1611672, 'file layout')
            h = hashlib.sha256()
            left = member['bytes']
            with (original / name).open('rb') as orig:
                while left:
                    if source_part is None:
                        source_part = (package / chunks[part_index]['path']).open('rb')
                        part_position = 0
                    take = min(left, BLOCK, chunks[part_index]['bytes'] - part_position)
                    public, saved = source_part.read(take), orig.read(take)
                    require(len(public) == take and public == saved, f'raw bytes differ: {name}')
                    h.update(public)
                    full.update(public)
                    part_position += take
                    left -= take
                    if part_position == chunks[part_index]['bytes']:
                        require(not source_part.read(1), 'extra part bytes')
                        source_part.close()
                        source_part = None
                        part_index += 1
                require(not orig.read(1), f'extra source bytes: {name}')
            require(h.hexdigest() == member['sha256'], f'member hash: {name}')
            file_offset += member['bytes']
    finally:
        if source_part:
            source_part.close()
    require(file_offset == stream['bytes'] and part_index == len(chunks), 'stream end mismatch')
    require(full.hexdigest() == stream['sha256'], 'stream digest mismatch')
    return {'parts': len(chunks), 'members': len(files), 'bytes': file_offset, 'sha256': full.hexdigest()}


def audit_records(package, original, independent, records):
    require(len(records) == 5 and {r['path'] for r in records} == {
        'manifest.json','attempt.jsonl','result.json','projection-weights.f32','independent-audit.json'},
        'record inventory mismatch')
    for record in records:
        path = package / record['gzip_path']
        require(path.stat().st_size == record['gzip_bytes'] and sha(path) == record['gzip_sha256'],
                'compressed record mismatch')
        source = independent if record['path'] == 'independent-audit.json' else original / record['path']
        count, h = 0, hashlib.sha256()
        with gzip.open(path, 'rb') as packed, source.open('rb') as orig:
            while b := packed.read(BLOCK):
                require(b == orig.read(len(b)), f'record bytes differ: {record["path"]}')
                h.update(b)
                count += len(b)
            require(not orig.read(1), 'extra original record bytes')
        require(count == record['bytes'] and h.hexdigest() == record['sha256'], 'record digest mismatch')
    return len(records)


def same_cell(row, key, expected):
    actual = row[key]
    if expected is None:
        require(actual == '', f'{key}: expected blank')
    elif isinstance(expected, bool):
        require(actual == str(expected), f'{key}: boolean mismatch')
    elif isinstance(expected, (int, float)):
        require(actual != '' and math.isfinite(float(actual)) and float(actual) == expected,
                f'{key}: number mismatch')
    else:
        require(actual == str(expected), f'{key}: text mismatch')


def audit_tables(package, result, audit, cases, family_review):
    families = {f['group']: f for f in audit['families']}
    family_ref = {f['group']: f for f in family_review['families']}
    by_case = {c['case_id']: c for c in cases['cases']}
    clean = {r['case_id']: r for r in result['records'] if r['action'] == 'clean'}
    actions = {(a['case_id'], a['action']): a for a in audit['actions']}
    require(len(families) == len(family_ref) == 12 and len(clean) == 96 and len(actions) == 288,
            'unexpected saved summary counts')
    with (package / 'families.csv').open(newline='', encoding='utf-8') as f:
        family_rows = list(csv.DictReader(f))
    require(len(family_rows) == 12 and len({r['group'] for r in family_rows}) == 12, 'family rows')
    for row in family_rows:
        group = int(row['group'])
        item, independent = families[group], family_ref[group]
        s, p, assessment = item['summary'], item['output_plan'], item['output_assessment']
        expected = {'baseline_status': item['baseline_status'], 'dose_status': s['dose_status'],
                    'dose_reason': assessment.get('reason') or p.get('reason'), 'q_out': s['q_out'],
                    'actual_range': assessment.get('actual_range'), 'contrast': s.get('contrast'),
                    'contrast_sign': s.get('contrast_sign'), 'h2_minus_h7': s.get('h2_minus_h7'),
                    'h10_minus_h7': s.get('h10_minus_h7')}
        require(json.loads(row['minimum_variant_heads']) == p['minima'], 'minimum plan mismatch')
        for h in ('h2','h10','h7'):
            expected['D_'+h] = s['D'][h]
            require(s['D'][h] == independent['D'][h], 'independent head D mismatch')
        for rotation, v in enumerate(s['rotations']):
            for h in ('h2','h10','h7'):
                expected[f'rotation{rotation}_{h}'] = v[h]['target_log_probability_D']
        for k, v in expected.items():
            same_cell(row, k, v)
        require(s['contrast'] == independent['contrast'] and s['contrast_sign'] == independent['contrast_sign']
                and item['baseline_status'] == independent['baseline_status'], 'independent family mismatch')

    with (package / 'cases.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 288 and len({(r['case_id'], r['action']) for r in rows}) == 288, 'case rows')
    totals = {h: {'n': 0, 'changed': 0, 'tv': [], 'kl': [], 'ties': 0} for h in ('out_h2','out_h10','out_h7')}
    repeat_tops = set()
    for row in rows:
        cid, action = row['case_id'], row['action']
        c, r, arm = by_case[cid], clean[cid], actions[(cid, action)]
        f = families[r['group']]
        z, score, comp, dose = r['score'], arm['score'], arm['comparison'], arm['dose']
        cosine = f['output_assessment']['pair_cosines'][r['variant']]
        expected = {'group': r['group'], 'variant': r['variant'], 'rotation': c['rotation'],
                    'shuffle': c['shuffle'], 'text': c['text'], 'target_id': r['target_id'],
                    'baseline_status': f['baseline_status'], 'dose_status': f['summary']['dose_status'],
                    'q_out': f['summary']['q_out'],
                    'clean_target_logit': z['target_logit'], 'clean_target_log_probability': z['target_log_probability'],
                    'clean_target_probability': math.exp(z['target_log_probability']), 'clean_target_rank': z['target_rank'],
                    'clean_unique_top_id': z['top_unique_id'], 'clean_top_representative_id': z['top_representative_id'],
                    'clean_top_tie_count': z['top_tie_count'], 'arm_target_logit': score['target_logit'],
                    'arm_target_log_probability': score['target_log_probability'],
                    'arm_target_probability': math.exp(score['target_log_probability']), 'arm_target_rank': score['target_rank'],
                    'arm_unique_top_id': score['top_unique_id'], 'arm_top_representative_id': score['top_representative_id'],
                    'arm_top_tie_count': score['top_tie_count'], 'cosine_h2_h10': cosine['2-10'],
                    'cosine_h2_h7': cosine['2-7'], 'cosine_h10_h7': cosine['10-7']}
        expected.update(comp)
        expected.update(dose)
        for k, v in expected.items():
            same_cell(row, k, v)
        require(z['top_tie_count'] == score['top_tie_count'] == 1, 'unexpected top tie')
        direct_change = 'unchanged' if z['top_unique_id'] == score['top_unique_id'] else 'changed'
        require(comp['top_change'] == direct_change, 'top change mismatch')
        t = totals[action]
        t['n'] += 1
        t['changed'] += direct_change == 'changed'
        t['tv'].append(comp['total_variation'])
        t['kl'].append(comp['kl_clean_to_arm'])
        if c['shuffle'] == 0 and z['top_unique_id'] == r['target_id']:
            repeat_tops.add(cid)
    out = {}
    for action, t in totals.items():
        require(t['n'] == 96, 'arm count mismatch')
        out[action] = {'count':t['n'], 'top_changes':t['changed'],
                       'mean_tv':math.fsum(t['tv'])/t['n'], 'mean_kl':math.fsum(t['kl'])/t['n']}
    return {'family_rows':len(family_rows), 'case_rows':len(rows), 'arm_summaries':out,
            'repeat_clean_target_top':len(repeat_tops)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--cases', required=True, type=Path)
    p.add_argument('--family-review', required=True, type=Path)
    a = p.parse_args()
    index = getjson(a.package / 'stream-index.json', INDEX_SHA)
    result = getjson(a.run / 'result.json', RESULT_SHA)
    audit = getjson(a.audit, AUDIT_SHA)
    cases = getjson(a.cases, CASES_SHA)
    family_review = getjson(a.family_review, FAMILY_SHA)
    require(index['source_result_sha256'] == RESULT_SHA and index['independent_audit_sha256'] == AUDIT_SHA,
            'index source binding mismatch')
    require(audit['result_sha256'] == RESULT_SHA and audit['cases_sha256'] == CASES_SHA, 'audit binding')
    require(sha(a.package / 'family-summary-review.json') == FAMILY_SHA, 'family review not packaged')
    report = {'status':'pass', 'index_sha256':INDEX_SHA,
              'stream':audit_stream(a.package,a.run,index),
              'records':audit_records(a.package,a.run,a.audit,index['records']),
              'tables':audit_tables(a.package,result,audit,cases,family_review),
              'scope':'Saved staged bytes and table metadata only; no model, GPU, execution or independent raw-logit analysis.'}
    print(json.dumps(report,sort_keys=True,separators=(',',':')))


if __name__ == '__main__':
    main()
