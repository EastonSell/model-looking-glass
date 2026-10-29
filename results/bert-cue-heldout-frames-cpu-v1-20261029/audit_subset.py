"""Replay an operator-derived target-logit subset of the fixed cue-frame study.

Checks retained target-logit bytes, exact event/input order, baseline eligibility,
all same-recipient specificity comparisons and the frame-level decision. This
does not verify dose vectors, full-logit controls, model execution or chronology.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
import re
import struct
from pathlib import Path

CUES = ('not', 'hardly', 'really', 'certainly')
LOW, HIGH = CUES[:2], CUES[2:]
SITES = ('cue2', 'object2', 'mask2', 'cue5')
FRAMES = ('subject_relative', 'causal_adjunct')
TRANSFERS = tuple((d, t) for t in CUES for d in CUES if d != t)
MANIFEST_HASH = '7a10dbcb598e48c2e8f56e2b75d8c03188ec04ca7a21ac071aea4b1e8294162c'
MANIFEST_PATH = 'experiments/bert_cue_heldout_frames_cases_v1.json'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def margin(event, any_id, some_id, vocab):
    raw = base64.b64decode(event['mask_logits_pair_f32le'], validate=True)
    if len(raw) != 8:
        raise ValueError('malformed saved target-logit pair')
    a, b = struct.unpack('<ff', raw)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite target logit')
    return math.fsum((a, -b))

def replay(manifest, result, stream, synthetic=False):
    cases = manifest['cases']
    if (len(cases) != 6 or [c['split'] for c in cases] != ['evaluation']*6
            or len({c['id'] for c in cases}) != 6 or result['status'] != 'complete'
            or len(result['cases']) != 6):
        raise ValueError('incomplete or mismatched case schedule')
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    vocab = None if synthetic else 30522
    if any_id == some_id or min(any_id, some_id) < 0:
        raise ValueError('invalid scored-token identities')
    count = 0
    def take(kind, ident, **keys):
        nonlocal count
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line or len(line) > 2 * 1024 * 1024:
            raise ValueError('missing or oversized event')
        event = json.loads(line)
        count += 1
        if event.get('kind') != kind or event.get('case_id') != ident or any(event.get(k) != v for k,v in keys.items()):
            raise ValueError('event order, identity or input mismatch')
        return event
    rows = []
    for case, saved in zip(cases, result['cases']):
        ident = case['id']
        if (saved['id'], saved['split'], saved['frame']) != (ident, case['split'], case['frame']):
            raise ValueError('result case identity mismatch')
        def ids(cue): return case['rows'][cue]['input_ids']
        clean = {}
        for cue in CUES:
            event = take('clean', ident, cue=cue, input_ids=ids(cue))
            clean[cue] = margin(event, any_id, some_id, vocab)
        gaps = [{'negative':n, 'affirmative':a, 'gap':math.fsum((clean[n],-clean[a]))}
                for n in LOW for a in HIGH]
        baseline = all(g['gap'] > 0.5 for g in gaps)
        baseline_event = take('baseline', ident)
        if (baseline_event['eligible'] != baseline or baseline_event['gaps'] != gaps
                or saved['baseline_eligible'] != baseline or saved['baseline_gaps'] != gaps
                or saved['clean_margins'] != clean):
            raise ValueError('baseline screen or clean margins differ')
        radius_event = take('radius', ident)
        r = radius_event['radius']
        if not math.isfinite(r) or r != saved['radius']:
            raise ValueError('invalid reported common radius')
        dose = r >= 0.001
        if radius_event['dose_eligible'] != dose or saved['dose_eligible'] != dose:
            raise ValueError('dose-eligibility disagreement')
        site_summaries = {}
        if not dose:
            take('dose_ineligible', ident, radius=r)
            if saved['effects'] or saved['specificity'] or saved['specificity_passed'] is not None:
                raise ValueError('dose-ineligible case reports interventions')
        else:
            for site in SITES:
                for cue in CUES:
                    e = take('self', ident, site=site, target=cue, input_ids=ids(cue))
                    if margin(e, any_id, some_id, vocab) != clean[cue]:
                        raise ValueError('self target margin differs')
                effects = {}
                for donor, target in TRANSFERS:
                    e = take('transfer', ident, site=site, donor=donor, target=target, input_ids=ids(target))
                    effects[(donor,target)] = math.fsum((margin(e, any_id, some_id, vocab), -clean[target]))
                if saved['effects'][site] != [{'donor':d,'target':t,'delta':effects[(d,t)]} for d,t in TRANSFERS]:
                    raise ValueError('saved effect differs from target logits')
                targets = {}
                for target in CUES:
                    own, opposite, sign = (LOW,HIGH,-1) if target in LOW else (HIGH,LOW,1)
                    same = next(c for c in own if c != target)
                    cross = {c:sign*effects[(c,target)] for c in opposite}
                    minimum = min(cross.values())
                    same_delta = effects[(same,target)]
                    value = math.fsum((minimum,-abs(same_delta)))
                    targets[target] = {'cross_signed':cross, 'cross_min':minimum, 'same_donor':same,
                                       'same_delta':same_delta, 'same_abs':abs(same_delta),
                                       'margin':value, 'passed':value > 0.05}
                detail = {'targets':targets, 'passed':all(x['passed'] for x in targets.values())}
                if saved['specificity'][site] != detail:
                    raise ValueError('specificity rule mismatch')
                site_summaries[site] = detail
            if saved['specificity_passed'] != site_summaries['cue2']['passed']:
                raise ValueError('cue specificity mismatch')
        for cue in CUES:
            e = take('repeat', ident, cue=cue, input_ids=ids(cue))
            if margin(e, any_id, some_id, vocab) != clean[cue]:
                raise ValueError('repeat target margin differs')
        forwards = 72 if dose else 8
        if saved['forwards'] != forwards:
            raise ValueError('case forward count mismatch')
        rows.append({'id':ident, 'split':case['split'], 'frame':case['frame'], 'forwards':forwards,
                     'clean_margins':clean, 'baseline_gaps':gaps, 'baseline_eligible':baseline,
                     'dose_eligible':dose, 'radius':r, 'site_specificity':site_summaries,
                     'specificity_passed':site_summaries['cue2']['passed'] if dose else None})
    if stream.readline(1):
        raise ValueError('unexpected extra event')
    if result['forwards'] != sum(row['forwards'] for row in rows):
        raise ValueError('total forward count mismatch')
    frames = {}
    for frame in FRAMES:
        members = [row for row in rows if row['split']=='evaluation' and row['frame']==frame]
        if len(members) != 3:
            raise ValueError('expected three evaluation cases in every frame')
        eligible = [row for row in members if row['baseline_eligible'] and row['dose_eligible']]
        frames[frame] = {'total':3, 'eligible':len(eligible),
                         'passing':sum(row['specificity_passed'] is True for row in eligible)}
    status = ('unevaluable' if any(x['eligible'] < 2 for x in frames.values()) else
              'passed' if all(x['passing'] >= 2 for x in frames.values()) else 'failed')
    evaluation = {'status':status, 'frames':frames}
    if result['evaluation'] != evaluation:
        raise ValueError('frame-level decision mismatch')
    return {'status':'passed', 'scope':'operator-derived target-logit subset and decisions; not original-journal authentication, full logits, dose, landing or execution',
            'forwards':result['forwards'], 'events':count, 'cases':rows, 'evaluation':evaluation}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('events', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--expected-source')
    p.add_argument('--expected-result-sha256')
    p.add_argument('--expected-subset-sha256')
    p.add_argument('--expected-original-journal-sha256')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    manifest_sha, event_sha, result_sha = sha(a.manifest), sha(a.events), sha(a.result)
    manifest, result = json.loads(a.manifest.read_bytes()), json.loads(a.result.read_bytes())
    if not a.synthetic:
        if (not re.fullmatch('[0-9a-f]{40}', a.expected_source or '') or
                not re.fullmatch('[0-9a-f]{64}', a.expected_result_sha256 or '') or
                result_sha != a.expected_result_sha256 or manifest_sha != MANIFEST_HASH or
                result.get('source_commit') != a.expected_source or
                result.get('source_sha256', {}).get(MANIFEST_PATH) != manifest_sha or
                not re.fullmatch('[0-9a-f]{64}', a.expected_subset_sha256 or '') or
                not re.fullmatch('[0-9a-f]{64}', a.expected_original_journal_sha256 or '') or
                event_sha != a.expected_subset_sha256 or
                result.get('events_sha256') != a.expected_original_journal_sha256 or a.events.suffix != '.gz'):
            raise ValueError('frozen source, manifest, result or journal binding mismatch')
    opener = gzip.open if a.events.suffix == '.gz' else open
    with opener(a.events, 'rb') as stream:
        report = replay(manifest, result, stream, synthetic=a.synthetic)
    report.update({'synthetic':a.synthetic,'source_commit':result.get('source_commit'),
                   'manifest_sha256':manifest_sha,'events_sha256':event_sha,'result_sha256':result_sha})
    print(json.dumps(report,sort_keys=True,allow_nan=False))

if __name__ == '__main__':
    main()
