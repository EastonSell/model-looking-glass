"""Post-run review of public BERT process receipts and ordered journals; no model."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK / 'teams/bert_taste_propagation_selection'
ATTEMPT = WORK / 'bert-propagation-study-attempt-v1'
COMMIT = 'a27b34533c3da5f34a541ef2b8b57631bf42e819'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, description):
    if not condition:
        raise AssertionError(description)


result = read(ATTEMPT / 'result.json')
exit_path = WORK / 'bert-propagation-study-operator-exit-v1.json'
outer_path = WORK / 'bert-propagation-study-outer-observation-v1.json'
start = read(WORK / 'bert-propagation-study-operator-start-v1.json')
spawn = read(WORK / 'bert-propagation-study-operator-spawn-v1.json')
inner = read(exit_path)
outer = read(outer_path)
case_path = ROOT / 'experiments/bert_taste_propagation_public_cases_v1.json'
cases = read(case_path)

require(result['status'] == 'complete' and result['completed_forwards'] == 960, 'result status/count')
require(not (ATTEMPT / 'incomplete.json').exists(), 'incomplete marker')
require(result['source']['commit'] == start['source_commit'] == inner['source_commit']
        == outer['source_commit'] == COMMIT, 'source identities')
require(result['source']['selected_source_sha256'] == start['selected_source_sha256'], 'source map')
require(sha(case_path) == start['cases_sha256'] == inner['cases_sha256'], 'case file')
head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
require(head == COMMIT, 'checkout HEAD')
for name, digest in result['source']['selected_source_sha256'].items():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', COMMIT + ':' + name])
    require(hashlib.sha256(raw).hexdigest() == digest == sha(ROOT / name), 'source blob ' + name)

require(inner['direct_child_exit_code'] == 0 and inner['direct_child_reaped'] is True
        and inner['timed_out'] is False and inner['operator_exception_type'] is None
        and inner['saved_result_and_journal_hashes_consistent'] is True
        and inner['prefix_empty_after'] is True and inner['incomplete_marker_present'] is False,
        'model child receipt')
require(inner['process_id'] == spawn['process_id'] and spawn['source_commit'] == COMMIT, 'spawn binding')
require(inner['result_sha256'] == sha(ATTEMPT / 'result.json'), 'result hash binding')
require(outer['operator_exit_code'] == 0 and outer['operator_reaped'] is True
        and outer['timed_out'] is False and outer['observer_exception_type'] is None,
        'operator child receipt')
require(outer['operator_exit_record_sha256'] == sha(exit_path), 'inner receipt binding')
require(outer['operator_sha256'] == sha(WORK / 'run_bert_propagation_cpu_v1.py'), 'operator source binding')
for record, prefix in ((inner, 'bert-propagation-study-'), (outer, 'bert-propagation-study-outer-')):
    for kind in ('stdout', 'stderr'):
        require(record[kind + '_sha256'] == sha(WORK / (prefix + kind + '-v1.log')), prefix + kind)

rows = {row['id']: row for row in cases['rows']}
require(len(rows) == 24 and len(cases['selected_stems']) == 6, 'case count')
require(set(result['journals']) == {stem + '.jsonl' for stem in cases['selected_stems']}, 'journal inventory')
reports = {}
sites = ((0, 7), (3, 7), (3, 1), (3, 3))
actions = tuple(axis + '_' + sign for axis in ('lex', 'context', 'orth1', 'orth2')
                for sign in ('plus', 'minus')) + ('self',)
for stem in cases['selected_stems']:
    path = ATTEMPT / (stem + '.jsonl')
    meta = result['journals'][path.name]
    require(path.stat().st_size == meta['size'] and sha(path) == meta['sha256'], 'journal hash ' + stem)
    ordered_rows = [row for row in cases['rows'] if row['stem'] == stem]
    require(len(ordered_rows) == 4, 'four crossed rows ' + stem)
    expected = [(r['id'], 0, 7, 'clean_capture') for r in ordered_rows]
    expected += [(r['id'], layer, position, action) for r in ordered_rows
                 for layer, position in sites for action in actions]
    expected += [(r['id'], 0, 7, 'fresh_clean') for r in ordered_rows]
    expected += [(r['id'], 5, 7, 'final_capture') for r in ordered_rows]
    expected += [(r['id'], 5, 7, 'final_donor') for r in ordered_rows]
    actual, pending, events = [], None, collections.Counter()
    retained = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            events[item['event']] += 1
            if item['event'] == 'forward_start':
                require(pending is None, 'overlapping start ' + stem)
                pending = item
            elif item['event'] == 'forward_complete':
                require(pending is not None, 'completion without start ' + stem)
                for key in ('row_id', 'stem', 'template', 'ingredient', 'layer', 'position', 'mode', 'donor'):
                    require(pending[key] == item[key], 'start/end identity ' + key)
                row = rows[item['row_id']]
                require(item['model_entry'] == {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']},
                        'recorded entry ' + item['row_id'])
                actual.append((item['row_id'], item['layer'], item['position'], item['mode']))
                retained += item['mask_logits_float32_le_base64'] is not None
                pending = None
            else:
                require(item['event'] == 'dose_plan' and pending is None, 'unexpected event ' + stem)
    require(actual == expected and pending is None, 'ordered schedule ' + stem)
    require(events == {'forward_start': 160, 'forward_complete': 160, 'dose_plan': 1}
            and retained == 132, 'event inventory ' + stem)
    reports[stem] = {'forwards_started': 160, 'forwards_completed': 160,
                     'full_logit_rows_retained': retained, 'journal_sha256': meta['sha256']}

report = {'status': 'passed', 'scope': 'Post-run saved-source, ordered-schedule and process-receipt consistency only; not proof of actual execution, chronology, whole-tree cleanup or numerical/scientific validity.',
          'source_commit': COMMIT, 'cases_sha256': sha(case_path), 'source_file_count': len(result['source']['selected_source_sha256']),
          'result_sha256': sha(ATTEMPT / 'result.json'), 'inner_receipt_sha256': sha(exit_path),
          'outer_receipt_sha256': sha(outer_path), 'stems': reports, 'forwards_started': 960, 'forwards_completed': 960}
out = WORK / 'reviews/bert_propagation_process_audit.json'
with out.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'status': 'passed', 'report_sha256': sha(out), 'sources': report['source_file_count'],
                  'forwards_started': 960, 'forwards_completed': 960}))
