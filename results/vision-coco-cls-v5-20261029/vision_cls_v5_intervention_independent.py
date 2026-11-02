"""Replay saved v5 CLS/spatial deliveries, controls, and decisions without Torch.

The companion screen reader checks the 64-image selection and initial clean
linkage. Inputs must remain quiescent. These checks establish saved-byte and
arithmetic consistency, not execution, source/model identity, publication
chronology, independent sampling, or supervision. Exact rounding-mode checks
sample the host at audit boundaries rather than throughout execution.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

COMPANION_SHA = 'a2c3e8fcdc102b9f4a14a4959c5542788bf23ec915853135b810e2554288c6d2'
companion_path = Path(__file__).with_name('vision_cls_v5_screen_independent.py')
if hashlib.sha256(companion_path.read_bytes()).hexdigest() != COMPANION_SHA:
    raise ValueError('screen reader source hash differs')
spec = importlib.util.spec_from_file_location('v5_screen_replay', companion_path)
screen_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen_reader)
s = screen_reader
TOKEN_BYTES = 192 * 4
STATE_BYTES = 197 * TOKEN_BYTES


def _round_shift_even(n, shift):
    if shift >= 0:
        return n << shift
    d = 1 << -shift
    quotient, remainder = divmod(n, d)
    return quotient + (2 * remainder > d or (2 * remainder == d and quotient % 2 == 1))


def f32_word(value):
    """Round the exact binary64 value to nearest-even binary32 using integers."""
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    sign = (bits >> 32) & 0x80000000
    exponent, fraction = (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    s.require(exponent != 0x7ff, 'nonfinite requested value')
    mantissa = fraction if exponent == 0 else (1 << 52) | fraction
    if not mantissa:
        return sign
    power = -1074 if exponent == 0 else exponent - 1075
    high = mantissa.bit_length() - 1 + power
    if high < -126:
        return sign | _round_shift_even(mantissa, power + 149)
    significand = _round_shift_even(mantissa, power + 23 - high)
    if significand == (1 << 24):
        significand >>= 1
        high += 1
    s.require(high <= 127, 'requested binary32 overflow')
    return sign | ((high + 127) << 23) | (significand - (1 << 23))


def encode(values):
    return b''.join(struct.pack('<I', f32_word(value)) for value in values)


def block(state, tokens):
    s.require(len(state) == STATE_BYTES and type(tokens) is list and
              len(tokens) in (1, 4) and len(set(tokens)) == len(tokens) and
              all(type(t) is int and 0 <= t < 197 for t in tokens), 'invalid state or tokens')
    return b''.join(state[t * TOKEN_BYTES:(t + 1) * TOKEN_BYTES] for t in tokens)


def distance(a, b):
    x, y = s.values(a), s.values(b)
    s.require(len(x) == len(y), 'distance shape differs')
    return math.sqrt(math.fsum((u - v) ** 2 for u, v in zip(x, y)))


def compare_nested(actual, expected, message):
    if type(expected) is dict:
        s.require(type(actual) is dict and set(actual) == set(expected), message)
        for key in expected:
            compare_nested(actual[key], expected[key], message + '/' + key)
    elif type(expected) is list:
        s.require(type(actual) is list and len(actual) == len(expected), message)
        for a, b in zip(actual, expected):
            compare_nested(a, b, message)
    elif type(expected) in (int, float) and type(expected) is not bool:
        s.close(actual, expected, message)
    else:
        s.require(s.typed_equal(actual, expected), message)


def _dose_reasons(norms, distances):
    reasons = []
    for site in ('cls', 'spatial'):
        if norms[site] <= 1e-6:
            reasons.append(site + '-receiver-norm')
        for kind in ('opposite', 'same_species'):
            if distances[site][kind] <= 1e-6:
                reasons.append(site + '-' + kind.replace('_species', '') + '-distance')
    return reasons


def _matchability(q, tolerance, norms, deliveries):
    reference = deliveries['cls']['opposite']
    return all(value > 0 and abs(value - q) <= tolerance and
               value <= .10 * norms[site] and abs(value - reference) <= .01 * reference
               for site, group in deliveries.items() for value in group.values())


def prepare_dose(own, other, same, spatial, pair_id, side, saved):
    sites = {'cls': [0], 'spatial': spatial}
    receiver = {site: block(own, ids) for site, ids in sites.items()}
    donor = {site: {'opposite': block(other, ids), 'same_species': block(same, ids)}
             for site, ids in sites.items()}
    norms = {site: distance(raw, bytes(len(raw))) for site, raw in receiver.items()}
    distances = {site: {kind: distance(receiver[site], raw) for kind, raw in group.items()}
                 for site, group in donor.items()}
    compare_nested(saved.get('receiver_norms'), norms, 'receiver norms')
    compare_nested(saved.get('distances'), distances, 'donor distances')
    s.require(all(value >= 0 for value in saved['receiver_norms'].values()) and
              all(value >= 0 for group in saved['distances'].values() for value in group.values()),
              'reported norm or distance is negative')
    computed_reasons = _dose_reasons(norms, distances)
    s.require(_dose_reasons(saved['receiver_norms'], saved['distances']) == computed_reasons,
              'reported distances change the eligibility decision')
    if computed_reasons:
        s.require(set(saved) == {'receiver_norms', 'distances', 'status', 'reasons'} and
                  saved['status'] == 'dose-ineligible' and saved['reasons'] == computed_reasons,
                  'distance eligibility differs')
        return saved['status'], {}
    q = min(.099 * norms['cls'], .099 * norms['spatial'],
            *(value for group in distances.values() for value in group.values()))
    reported_q = min(.099 * saved['receiver_norms']['cls'], .099 * saved['receiver_norms']['spatial'],
                     *(value for group in saved['distances'].values() for value in group.values()))
    s.close(saved.get('q'), q, 'q differs')
    s.require(saved['q'] == reported_q and (q <= 1e-6) == (reported_q <= 1e-6),
              'reported q changes the fixed bound or eligibility')
    if q <= 1e-6:
        s.require(set(saved) == {'receiver_norms', 'distances', 'status', 'reasons', 'q'} and
                  saved['status'] == 'dose-ineligible' and saved['reasons'] == ['q'], 'q eligibility differs')
        return saved['status'], {}
    s.require(set(saved) == {'receiver_norms', 'distances', 'status', 'reasons', 'q', 'tolerance',
                             'planned_norms', 'donor_scales'}, 'dose fields differ')
    tolerance = min(max(1e-5, 1e-5 * reported_q), .01 * reported_q)
    s.require(saved['tolerance'] == tolerance, 'dose tolerance differs')
    requested, scales = {}, {}
    for site in ('cls', 'spatial'):
        x = s.values(receiver[site])
        requested[site], scales[site] = {}, {}
        for kind in ('opposite', 'same_species'):
            y = s.values(donor[site][kind])
            ratio = reported_q / saved['distances'][site][kind]
            requested[site][kind] = encode([a + ratio * (b - a) for a, b in zip(x, y)])
            scales[site][kind] = ratio
        scale = reported_q / math.sqrt(len(x))
        signs = [1 if hashlib.sha256(f'vision-coco-cls-v5:{pair_id}:{side}:{site}:{i}'.encode('ascii')).digest()[0] & 1
                 else -1 for i in range(len(x))]
        requested[site]['random'] = encode([value + scale * sign for value, sign in zip(x, signs)])
    planned = {site: {kind: distance(receiver[site], raw) for kind, raw in group.items()}
               for site, group in requested.items()}
    compare_nested(saved['donor_scales'], scales, 'donor scales')
    compare_nested(saved['planned_norms'], planned, 'planned dose')
    s.require(all(value >= 0 for group in saved['planned_norms'].values() for value in group.values()),
              'reported planned dose is negative')
    actual_ok = _matchability(q, min(max(1e-5, 1e-5*q), .01*q), norms, planned)
    recorded_ok = _matchability(reported_q, tolerance, saved['receiver_norms'], saved['planned_norms'])
    s.require(actual_ok == recorded_ok, 'reported planned doses change matchability')
    status = 'matchable' if actual_ok else 'dose-unmatchable'
    s.require(saved['status'] == status and saved['reasons'] == ([] if actual_ok else ['rounded-delivery']),
              'dose matchability differs')
    return status, requested


def distribution(clean, patched):
    left, right = s.values(clean), s.values(patched)
    s.require(len(left) == len(right) == 1000, 'invalid logits')
    lm, rm = max(left), max(right)
    ls = math.fsum(math.exp(x-lm) for x in left)
    rs = math.fsum(math.exp(x-rm) for x in right)
    lp = [math.exp(x-lm)/ls for x in left]
    rp = [math.exp(x-rm)/rs for x in right]
    tv = .5 * math.fsum(abs(x-y) for x, y in zip(lp, rp))
    kl = math.fsum(p * ((x-y)-(lm-rm)-(math.log(ls)-math.log(rs)))
                   for p, x, y in zip(lp, left, right))
    s.require(math.isfinite(tv) and math.isfinite(kl) and kl >= -1e-12, 'invalid distribution')
    return {'tv': tv, 'kl_clean_to_patched': max(0., kl),
            'top1_changed': left.index(lm) != right.index(rm)}


def side_comparison(movements):
    c, p = movements['cls'], movements['spatial']
    ac = c['opposite'] - max(abs(c['same_species']), abs(c['random']))
    ap = p['opposite'] - max(abs(p['same_species']), abs(p['random']))
    return {'raw_opposite_difference': c['opposite']-p['opposite'], 'adjusted_cls': ac,
            'adjusted_spatial': ap, 'adjusted_difference': ac-ap,
            'side_pass': c['opposite'] >= .05 and ac >= .025 and ac-ap >= .025}


def baseline_status(cat, dog):
    gap = dog['margin'] - cat['margin']
    reasons = []
    if cat['margin'] >= 0: reasons.append('cat-relative-margin')
    if dog['margin'] <= 0: reasons.append('dog-relative-margin')
    if gap < 1: reasons.append('gap-below-one')
    if cat['cat_mass'] < .2: reasons.append('cat-mass-below-0.2')
    if dog['dog_mass'] < .2: reasons.append('dog-mass-below-0.2')
    return {'eligible': not reasons, 'reasons': reasons, 'gap': gap}


def clean_record(records, pair_id, side, name, baseline):
    row, tensors = records.take(f'{pair_id}:{side}:{name}')
    s.expect_shapes(row, tensors, {'logits': [1,1000], 'early': s.STATE_SHAPE, 'late': s.STATE_SHAPE})
    s.require(all(tensors[k] == baseline[k] for k in ('logits','early','late')), name+' clean differs')
    score, _ = s.checked_score(row, tensors, baseline['conditional'])
    return score, tensors['logits']


def patch_record(records, pair_id, side, name, path, tokens, before, replacement, baseline, expected_logits):
    action = 'null' if name.startswith('null_') else 'patch'
    row, tensors = records.take(f'{pair_id}:{side}:{name}', path, tokens, action)
    shapes = {'logits':[1,1000], 'before_all':s.STATE_SHAPE, 'after_all':s.STATE_SHAPE}
    if tokens is not None:
        shapes.update({f'{which}_{i}':[192] for i in range(len(tokens)) for which in ('before','after')})
    if action == 'patch':
        shapes['requested'] = s.STATE_SHAPE if tokens is None else [len(tokens), 192]
    s.expect_shapes(row, tensors, shapes)
    s.require(tensors['before_all'] == before, 'same-forward pre-state differs')
    if action == 'patch':
        s.require(tensors['requested'] == replacement, name+' requested bytes differ')
    if tokens is None:
        s.require(tensors['after_all'] == (before if action == 'null' else replacement), 'full landing differs')
    else:
        expected = block(before,tokens) if action == 'null' else replacement
        s.require(len(expected) == len(tokens)*TOKEN_BYTES, 'replacement size differs')
        after = bytearray(before)
        for i, token in enumerate(tokens):
            lo, hi = token*TOKEN_BYTES, (token+1)*TOKEN_BYTES
            delivered = expected[i*TOKEN_BYTES:(i+1)*TOKEN_BYTES]
            s.require(tensors[f'before_{i}'] == before[lo:hi] and tensors[f'after_{i}'] == delivered,
                      'per-token landing differs')
            after[lo:hi] = delivered
        s.require(tensors['after_all'] == after, 'untouched state or landing differs')
    if expected_logits is not None:
        s.require(tensors['logits'] == expected_logits, name+' control logits differ')
    score, _ = s.checked_score(row, tensors, baseline['conditional'])
    return score, tensors


def check_condition(saved, score, clean, logits):
    s.check_score(saved, score)
    effect = distribution(clean['logits'], logits)
    compare_nested({key:saved.get(key) for key in effect}, effect, 'distribution')
    s.require(0 <= saved['tv'] <= 1 and saved['kl_clean_to_patched'] >= 0 and
              type(saved['top1_changed']) is bool, 'reported distribution invalid')


def aggregate(statuses):
    s.require(len(statuses) <= 4 and all(x in {'positive','negative','dose-ineligible','dose-unmatchable'}
                                        for x in statuses), 'pair statuses differ')
    evaluable = sum(x in ('positive','negative') for x in statuses)
    positive = statuses.count('positive')
    status = 'inconclusive' if evaluable < 2 else 'positive' if 4*positive >= 3*evaluable else 'negative'
    return {'status':status, 'selected':len(statuses), 'evaluable':evaluable, 'positive':positive}


def audit_interventions(data, screen, synthetic=False):
    s.require_nearest_rounding()
    cross = s.cross_phase_audit(data, screen, synthetic)
    _, started, terminal_raw, terminal = s.validate_start_terminal(
        data, 'intervention', screen['manifest_sha256'], synthetic)
    pairs = screen['pairs']
    count = terminal['submissions']
    s.require(type(count) is int and 0 < count <= 34*len(pairs), 'forward count exceeds fixed plan')
    files = {f'forward-{i:03d}.{ext}' for i in range(1,count+1) for ext in ('json','bin')}
    expected_files = files | {'started.json','terminal.json'} | {p['pair_id']+'.json' for p in pairs}
    s.require({p.name for p in data.iterdir()} == expected_files, 'intervention file inventory differs')
    targets = [p[side]['image_id'] for p in pairs for side in ('cat','dog')]
    donors = [p[side+'_donor']['image_id'] for p in pairs for side in ('cat','dog')]
    s.require(len(set(targets)) == len(targets) and not set(targets).intersection(donors),
              'target/donor roles differ')
    records, clean = s.Records(data,count), {}
    for pair in pairs:
        for role in ('cat','dog','cat_donor','dog_donor'):
            ident = pair[role]['image_id']
            if ident not in clean:
                clean[ident] = s.initial(records,pair['pair_id'],role)
                s.require(clean[ident]['logits'] == screen['raw_by_id'][ident], 'initial screen logits differ')
    terminal_pairs = terminal.get('pairs')
    s.require(type(terminal_pairs) is list and
              [row.get('pair_id') for row in terminal_pairs] == [p['pair_id'] for p in pairs],
              'terminal pair inventory differs')
    statuses, outcomes, pair_hashes = [], [], {}
    for pair, terminal_pair in zip(pairs,terminal_pairs):
        pair_id, spatial = pair['pair_id'], pair['box_tokens']
        raw, saved = s.read_json(data/(pair_id+'.json'), 1_000_000)
        pair_hashes[pair_id] = s.sha(raw)
        s.require(s.typed_equal(saved,terminal_pair) and saved.get('split') == pair['split'] and
                  type(saved.get('sides')) is dict and set(saved['sides']) == {'cat','dog'},
                  'pair or terminal record differs')
        cat, dog = clean[pair['cat']['image_id']], clean[pair['dog']['image_id']]
        base = baseline_status(cat['score'],dog['score'])
        reported_base = baseline_status(saved['sides']['cat']['baseline'],saved['sides']['dog']['baseline'])
        s.require(base['eligible'] and reported_base['eligible'] and
                  reported_base['reasons'] == base['reasons'], 'screened baseline is contradicted')
        compare_nested(saved['baseline_eligibility'],base,'baseline eligibility')
        s.require(saved['baseline_eligibility']['gap'] == reported_base['gap'],
                  'reported baseline gap contradicts margins')
        statuses_by_side, raw_comparisons, saved_comparisons = {}, {}, {}
        side_reports = {}
        for side, other_side in (('cat','dog'),('dog','cat')):
            own = clean[pair[side]['image_id']]
            other = clean[pair[other_side]['image_id']]
            same = clean[pair[side+'_donor']['image_id']]
            info = saved['sides'][side]
            s.require(type(info.get('image_id')) is int and info['image_id'] == pair[side]['image_id'] and
                      type(info.get('same_species_donor_id')) is int and
                      info['same_species_donor_id'] == pair[side+'_donor']['image_id'], 'side identity differs')
            s.check_score(info['baseline'],own['score'])
            s.check_score(info['same_species_baseline'],same['score'])
            category = 17 if side == 'cat' else 18
            s.require(s.reported_screen_gate(info['baseline'],category)[0] and
                      s.reported_screen_gate(info['same_species_baseline'],category)[0],
                      'reported target/donor recognition differs')
            status, requested = prepare_dose(own['early'],other['early'],same['early'],spatial,
                                             pair_id,side,info['dose'])
            statuses_by_side[side] = status
            conditions = info['conditions']
            repeat, repeat_logits = clean_record(records,pair_id,side,'repeat',own)
            check_condition(conditions['repeat'],repeat,own,repeat_logits)
            plans = [('null_cls',s.EARLY,[0],None,own['logits']),
                     ('self_cls',s.EARLY,[0],block(own['early'],[0]),own['logits']),
                     ('null_spatial',s.EARLY,spatial,None,own['logits']),
                     ('self_spatial',s.EARLY,spatial,block(own['early'],spatial),own['logits']),
                     ('full_self',s.EARLY,None,own['early'],own['logits']),
                     ('full_swap',s.EARLY,None,other['early'],other['logits'])]
            if status == 'matchable':
                for site, tokens in (('cls',[0]),('spatial',spatial)):
                    for kind in ('opposite','same_species','random'):
                        plans.append((site+'_'+kind,s.EARLY,tokens,requested[site][kind],None))
            plans.append(('final_spatial',s.LATE,spatial,block(other['late'],spatial),own['logits']))
            s.require(type(conditions) is dict and set(conditions) == {'repeat','post_clean'} |
                      {item[0] for item in plans}, 'condition inventory differs')
            moves = {'cls':{}, 'spatial':{}}
            reported_moves = {'cls':{}, 'spatial':{}}
            ref_actual = ref_reported = None
            for name,path,tokens,replacement,expected_logits in plans:
                before = own['early'] if path == s.EARLY else own['late']
                score, tensors = patch_record(records,pair_id,side,name,path,tokens,before,replacement,
                                              own,expected_logits)
                condition = conditions[name]
                check_condition(condition,score,own,tensors['logits'])
                delta = score['margin']-own['score']['margin']
                oriented = delta * (1 if side == 'cat' else -1)
                s.close(condition.get('raw_margin_change'),delta,'margin change differs')
                s.close(condition.get('annotation_oriented_change'),oriented,'oriented movement differs')
                reported_delta = condition['margin']-info['baseline']['margin']
                s.require(condition['raw_margin_change'] == reported_delta and
                          condition['annotation_oriented_change'] == reported_delta*(1 if side == 'cat' else -1),
                          'reported movement contradicts reported margins')
                if name.startswith(('cls_','spatial_')):
                    site, kind = name.split('_',1)
                    pre, post = block(tensors['before_all'],tokens), block(tensors['after_all'],tokens)
                    actual, receiver = distance(pre,post), distance(pre,bytes(len(pre)))
                    q, tol = info['dose']['q'], info['dose']['tolerance']
                    s.require(actual > 0 and receiver > 0 and abs(actual-q) <= tol and actual <= .10*receiver,
                              'actual dose or receiver cap differs')
                    if name == 'cls_opposite': ref_actual = actual
                    else:
                        s.require(ref_actual is not None and abs(actual-ref_actual) <= .01*ref_actual,
                                  'actual cross-site dose match differs')
                    s.close(condition.get('actual_dose'),actual,'actual dose field differs')
                    s.close(condition.get('receiver_norm'),receiver,'actual receiver norm differs')
                    s.close(condition.get('realized_over_receiver_norm'),actual/receiver,'relative dose differs')
                    recorded_actual, recorded_receiver = condition['actual_dose'], condition['receiver_norm']
                    s.require(recorded_actual > 0 and recorded_receiver > 0 and
                              abs(recorded_actual-q) <= tol and recorded_actual <= .10*recorded_receiver and
                              condition['realized_over_receiver_norm'] == recorded_actual/recorded_receiver,
                              'reported actual dose contradicts its bounds')
                    if name == 'cls_opposite': ref_reported = recorded_actual
                    else:
                        s.require(ref_reported is not None and
                                  abs(recorded_actual-ref_reported) <= .01*ref_reported,
                                  'reported cross-site dose match differs')
                    per_token = [distance(pre[i*TOKEN_BYTES:(i+1)*TOKEN_BYTES],
                                          post[i*TOKEN_BYTES:(i+1)*TOKEN_BYTES]) for i in range(len(tokens))]
                    compare_nested(condition.get('token_doses'),per_token,'per-token dose')
                    moves[site][kind] = oriented
                    reported_moves[site][kind] = condition['annotation_oriented_change']
            post, post_logits = clean_record(records,pair_id,side,'post_clean',own)
            check_condition(conditions['post_clean'],post,own,post_logits)
            if status == 'matchable':
                compare_nested(info['movements'],moves,'movements')
                s.require(s.typed_equal(info['movements'],reported_moves),
                          'reported movements contradict conditions')
                comparison, comparison_reported = side_comparison(moves), side_comparison(reported_moves)
                compare_nested(info['comparison'],comparison,'comparison')
                s.require(s.typed_equal(info['comparison'],comparison_reported) and
                          comparison['side_pass'] == comparison_reported['side_pass'],
                          'reported movements change the side decision')
                raw_comparisons[side], saved_comparisons[side] = comparison, comparison_reported
                side_reports[side] = {'movements':moves, 'comparison':comparison}
            else:
                s.require('movements' not in info and 'comparison' not in info,
                          'unexpected movements for ineligible side')
                side_reports[side] = {'dose_status':status}
        s.require(s.typed_equal(saved['dose_matchability'],statuses_by_side), 'pair dose statuses differ')
        if 'dose-ineligible' in statuses_by_side.values(): pair_status = 'dose-ineligible'
        elif 'dose-unmatchable' in statuses_by_side.values(): pair_status = 'dose-unmatchable'
        else:
            passed = all(raw_comparisons[side]['side_pass'] for side in ('cat','dog'))
            s.require(passed == all(saved_comparisons[side]['side_pass'] for side in ('cat','dog')),
                      'reported pair decision differs')
            pair_status = 'positive' if passed else 'negative'
        s.require(saved['status'] == pair_status, 'pair status differs')
        statuses.append(pair_status)
        outcomes.append({'pair_id':pair_id, 'status':pair_status, 'dose_statuses':statuses_by_side,
                         'sides':side_reports})
    s.require(records.next == count+1, 'unexpected trailing forward records')
    result = aggregate(statuses)
    s.require(s.typed_equal(terminal.get('aggregate'),result), 'aggregate differs')
    s.require_nearest_rounding()
    return {'terminal_sha256':s.sha(terminal_raw), 'start_sha256':cross['start_sha256'],
            'forwards':count, 'forward_hashes':records.hashes, 'pair_hashes':pair_hashes,
            'pairs':outcomes, 'aggregate':result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--screen-data',type=Path,required=True)
    parser.add_argument('--public-screen-result',type=Path,required=True)
    parser.add_argument('--public-screen-archive',type=Path,required=True)
    parser.add_argument('--public-screen-custody',type=Path,required=True)
    parser.add_argument('--public-screen-observer',type=Path,required=True)
    parser.add_argument('--screen-supervisor',type=Path)
    parser.add_argument('--intervention-data',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    screen = s.screen_audit(args.manifest,args.screen_data,args.public_screen_result,args.public_screen_archive,
                            args.public_screen_custody,args.public_screen_observer,args.screen_supervisor,args.synthetic)
    result = audit_interventions(args.intervention_data,screen,args.synthetic)
    report = {'schema':'independent-vision-cls-v5-intervention-audit-v1',
              'scope':'Saved-byte, dose, landing, controls and numerical consistency only; no source/model, '
                      'execution, supervision, chronology or independent-sampling attestation. Inputs must remain '
                      'quiescent. Top-one IDs are checked but optional class-label strings are not.',
              'synthetic':args.synthetic, 'screen_reader_sha256':COMPANION_SHA,
              'manifest_sha256':screen['manifest_sha256'], 'screen_result_sha256':screen['screen_result_sha256'],
              'screen_archive_sha256':screen['screen_archive_sha256'],
              'screen_custody_sha256':screen['screen_custody_sha256'],
              'screen_observer_sha256':screen['screen_observer_sha256'],
              'screen_supervisor_checked':screen['supervisor_checked'], 'intervention':result}
    raw = (json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with args.report.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
    print(json.dumps({'report_sha256':s.sha(raw), 'forwards':result['forwards'],
                      'aggregate':result['aggregate']}))


if __name__ == '__main__':
    main()
