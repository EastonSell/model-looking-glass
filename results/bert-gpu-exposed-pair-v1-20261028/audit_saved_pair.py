"""Recompute the saved four-forward exposed-pair controls, without model or CUDA.

Invocation: python -B audit_saved_pair.py RESULT_JSON EXPECTED_SHA256 [EXPECTED_COMMIT]
The caller supplies an independently retained digest and separately verifies
source, execution, supervision, cleanup, and publication identity.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__:
    raise RuntimeError('this auditor requires assertions; do not run Python with -O or -OO')

def decode32(word):
    assert type(word) is int and 0 <= word <= 0xffffffff
    exponent, fraction = (word >> 23) & 0xff, word & 0x7fffff
    assert exponent != 0xff
    magnitude = math.ldexp(float(fraction if exponent == 0 else (1 << 23) + fraction),
                           -149 if exponent == 0 else exponent - 150)
    return -magnitude if word & 0x80000000 else magnitude

def decode_vector(value):
    assert type(value) is str
    raw = base64.b64decode(value, validate=True)
    assert len(raw) == 3072 and base64.b64encode(raw).decode('ascii') == value
    for word in struct.unpack('<768I', raw):
        decode32(word)
    return raw

def audit(path, expected_sha, expected_commit=None):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    def unique(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError('nonfinite JSON value')
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    if expected_commit is not None:
        assert len(expected_commit) == 40 and all(c in '0123456789abcdef' for c in expected_commit)
        assert result['source_commit'] == expected_commit
    measurement = result['measurement']
    assert measurement['target_ids'] == [101,1996,5276,3875,1996,3063,103,3201,1012,102]
    assert measurement['donor_ids'] == [101,1996,12499,3875,1996,3063,103,3201,1012,102]
    assert measurement['output_token_ids'] == {'is':2003,'are':2024}
    seq = measurement['sequence']
    assert seq['forward_count'] == 4 and all(seq[k] is True for k in
        ('self_mask_logits_bitwise_equal','cross_source_landed','source_differs_from_target'))
    rows = seq['observations']
    assert [row['stage'] for row in rows] == ['target','donor','self','cross']
    before, after, margins = [], [], {}
    for row in rows:
        before.append(decode_vector(row['before_float32_le_base64']))
        after.append(None if row['after_float32_le_base64'] is None else decode_vector(row['after_float32_le_base64']))
        names = ['before'] if row['stage'] in ('target','donor') else ['before','patch','after']
        operations = ['capture'] if len(names) == 1 else ['capture','patch','capture']
        path = 'distilbert.transformer.layer.2'
        expected_events = [{'name':name,'operation':operation,'order':index,
            'module_path':path,'module_aliases':[path],'call_site_known':False,
            'forward_index':0,'invocation':0,'batch_index':0,'token_index':2,
            'output_selector':[0],'output_shape':[1,10,768],'selected_shape':[768],
            'dtype':'torch.float32','device':'cuda:0',
            'source_capture':'before' if row['stage'] == 'self' and index == 1 else None,
            'preceding_same_site_actions':names[:index]}
            for index,(name,operation) in enumerate(zip(names,operations,strict=True))]
        canonical = lambda obj: json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
        assert canonical(row['events']) == canonical(expected_events)
        for name in ('is','are'):
            decoded = decode32(row[name+'_bits'])
            assert type(row[name+'_logit']) is float
            assert struct.pack('>d', row[name+'_logit']) == struct.pack('>d', decoded)
        mask = row['mask_logits_sha256']
        assert type(mask) is str and len(mask) == 64 and all(c in '0123456789abcdef' for c in mask)
        margins[row['stage']] = math.fsum((decode32(row['is_bits']), -decode32(row['are_bits'])))
    assert after[0] is None and after[1] is None
    assert before[0] != before[1] and before[2] == before[0] and after[2] == before[0]
    assert before[3] == before[0] and after[3] == before[1]
    assert rows[0]['mask_logits_sha256'] == rows[2]['mask_logits_sha256']
    assert (rows[0]['is_bits'],rows[0]['are_bits']) == (rows[2]['is_bits'],rows[2]['are_bits'])
    movement = math.fsum((margins['target'], -margins['cross']))
    sign_change = (margins['target'] > 0 and margins['cross'] < 0) or (margins['target'] < 0 and margins['cross'] > 0)
    assert measurement['pair_analysis'] == {'margins':margins,'target_minus_cross':movement,'strict_sign_change':sign_change}
    source_status = result.get('status')
    if source_status not in ('success','incomplete','quarantined','fabricated'):
        source_status = 'unknown'
    memory_summary = None
    if source_status == 'success':
        stages = result['partial_stage_observations']
        expected = [('before_model_load',0),('before_model_copy',None),('after_model_copy',None),
            ('before_target',0),('after_target',1),('before_donor',1),('after_donor',2),
            ('before_self',2),('after_self',3),('before_cross',3),('after_cross',4),
            ('cleanup_before_workspace_clear',None),('after_cleanup',None)]
        total, cap, commitment = (measurement[k] for k in
                                  ('device_total_bytes','allocator_cap_bytes','whole_run_commitment_bytes'))
        gib, headroom = 1024**3, 256*1024**2
        assert all(type(n) is int for n in (total,cap,commitment))
        assert 7*gib <= total <= 9*gib and cap == total//4 and commitment == cap+headroom
        reserve = max(gib,(15*total+99)//100)
        min_free, previous_allocated_peak, previous_reserved_peak = total, 0, 0
        for name,count in expected:
            row = stages[name]
            assert type(row['observations_live']) is type(count) and row['observations_live'] == count
            assert row['peak_scope'] == 'cumulative_since_pre_model_load_reset'
            memory = row['memory']
            keys = ('device_free_bytes','device_total_bytes','allocated_bytes','reserved_bytes',
                    'peak_allocated_bytes','peak_reserved_bytes')
            assert all(type(memory[k]) is int and memory[k] >= 0 for k in keys)
            free, reported_total, allocated, reserved, peak_a, peak_r = (memory[k] for k in keys)
            assert reported_total == total and free <= total and free+reserved <= total
            assert allocated <= reserved <= total
            assert allocated <= peak_a <= peak_r <= total and reserved <= peak_r
            assert peak_a >= previous_allocated_peak and peak_r >= previous_reserved_peak
            assert free >= reserve+commitment and reserved+reserve+commitment <= total
            assert free >= reserve+headroom and peak_r <= cap-headroom
            min_free = min(min_free,free)
            previous_allocated_peak,previous_reserved_peak = peak_a,peak_r
        assert stages['before_model_load']['memory']['device_free_bytes'] >= 3*gib
        assert stages['after_cleanup']['memory'] == measurement['cleanup']
        assert stages['cleanup_weakrefs'] == measurement['cleanup_weakrefs'] == {
            'model':False,'target':False,'donor':False}
        assert measurement['cleanup']['allocated_bytes'] == measurement['cleanup']['reserved_bytes'] == 0
        memory_summary = {'stage_count':len(expected),'device_total_bytes':total,
            'allocator_cap_bytes':cap,'whole_run_commitment_bytes':commitment,'reserve_bytes':reserve,
            'minimum_sampled_free_bytes':min_free,'cumulative_peak_allocated_bytes':previous_allocated_peak,
            'cumulative_peak_reserved_bytes':previous_reserved_peak}
    return {'status':'saved_pair_controls_consistent','source_status':source_status,
            'result_sha256':expected_sha,'margins':margins,
            'target_minus_cross':movement,'strict_sign_change':sign_change,
            'memory':memory_summary,
            'limitations':['recorded vectors and selected bits only',
                'full logits unavailable from digest; source/execution/cleanup require separate assessment',
                'one exposed nonblind pair; no lexical-control or generalization claim']}

if __name__ == '__main__':
    assert len(sys.argv) in (3,4)
    print(json.dumps(audit(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv) == 4 else None),sort_keys=True))
