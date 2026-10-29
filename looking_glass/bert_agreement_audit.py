"""Reconstruct one BERT agreement family from its recorded CPU events.

This validates internal evidence consistency, not whether a process executed
the described model. Full mask logits are represented by hashes, not retained
values; their equality and finiteness cannot be independently reconstructed.
"""

import base64
import math
import re
import struct

from .bert_agreement_runner import SITES, _family_rows, decode_float32_bits, signed_donor_effect


def _vector(record, name, hidden):
    value = record.get(name)
    if type(value) is not str:
        raise ValueError('missing recorded float32 vector')
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError('invalid vector encoding') from error
    if (len(raw) != 4 * hidden or base64.b64encode(raw).decode('ascii') != value
            or any((number & 0x7f800000) == 0x7f800000
                   for (number,) in struct.iter_unpack('<I', raw))):
        raise ValueError('invalid or nonfinite recorded vector')
    return raw


def _same_fields(record, expected):
    return (type(record) is dict and all(type(record.get(key)) is type(value)
            and record.get(key) == value for key, value in expected.items()))


def audit_family_events(manifest, records, *, split, family, mask_id, hidden_size):
    """Validate the exact event schedule and reconstruct all case effects."""
    if (type(records) is not list or type(hidden_size) is not int
            or not 1 <= hidden_size <= 4096):
        raise ValueError('bounded records and hidden size are required')
    rows, targets, by_id = _family_rows(manifest, split, family, mask_id)
    cursor, count = 0, 0
    references, baselines, case_effects, controls = {}, {}, {}, []

    def next_record():
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError('event stream is incomplete')
        record = records[cursor]
        cursor += 1
        return record

    def forward(kind, row_id, site, layer, token, *, donor_id=None, patch=False, self_patch=False):
        nonlocal count
        count += 1
        identity = {'split': split, 'family': family, 'forward_index': count, 'kind': kind,
                    'row_id': row_id, 'donor_row_id': donor_id, 'site': site,
                    'layer': layer, 'token': token}
        started, completed = next_record(), next_record()
        if (not _same_fields(started, {'status': 'started', **identity})
                or not _same_fields(completed, {'status': 'completed', **identity})):
            raise ValueError('forward start/completion inventory differs')
        before = _vector(completed, 'before_float32_le_base64', hidden_size)
        if patch:
            after = _vector(completed, 'after_float32_le_base64', hidden_size)
        else:
            if completed.get('after_float32_le_base64') is not None:
                raise ValueError('capture unexpectedly contains an action vector')
            after = None
        capture_bytes, source_bytes = hidden_size * 4 * (2 if patch else 1), hidden_size * 4 * (patch and not self_patch)
        if (type(completed.get('capture_bytes')) is not int or completed['capture_bytes'] != capture_bytes
                or type(completed.get('source_bytes')) is not int or completed['source_bytes'] != source_bytes):
            raise ValueError('recorded tensor byte counts differ')
        mask_sha = completed.get('mask_logits_sha256')
        if type(mask_sha) is not str or re.fullmatch('[0-9a-f]{64}', mask_sha) is None:
            raise ValueError('invalid mask-logit digest')
        is_bits, are_bits = completed.get('is_bits'), completed.get('are_bits')
        decoded_is, decoded_are = decode_float32_bits(is_bits), decode_float32_bits(are_bits)
        if (type(completed.get('is_logit')) not in (int, float)
                or type(completed.get('are_logit')) not in (int, float)
                or completed['is_logit'] != decoded_is or completed['are_logit'] != decoded_are):
            raise ValueError('recorded logits do not match their float32 bits')
        events = completed.get('events')
        names, operations = (('before', 'patch', 'after'), ('capture', 'patch', 'capture')) if patch else (('before',), ('capture',))
        if type(events) is not list or len(events) != len(names):
            raise ValueError('intervention event count differs')
        path = f'distilbert.transformer.layer.{layer}'
        for index, event in enumerate(events):
            expected = {'name': names[index], 'operation': operations[index], 'order': index,
                        'module_path': path, 'module_aliases': [path], 'call_site_known': False,
                        'forward_index': 0, 'invocation': 0, 'batch_index': 0, 'token_index': token,
                        'output_selector': [0], 'output_shape': [1, 10, hidden_size],
                        'selected_shape': [hidden_size], 'dtype': 'torch.float32', 'device': 'cpu',
                        'source_capture': 'before' if self_patch and index == 1 else None,
                        'preceding_same_site_actions': list(names[:index])}
            if not _same_fields(event, expected):
                raise ValueError('recorded intervention metadata differs')
            if any(type(value) is not int for key in ('output_selector', 'output_shape', 'selected_shape')
                   for value in event[key]):
                raise ValueError('recorded tensor coordinates must be integers')
        return {'before': before, 'after': after, 'mask_sha': mask_sha,
                'bits': (is_bits, are_bits)}

    def control(name, case_id, site, actual, expected_value):
        record = next_record()
        expected = {'status': 'control', 'split': split, 'family': family,
                    'name': name, 'case_id': case_id, 'site': site, 'passed': True}
        if (not _same_fields(record, expected) or actual['mask_sha'] != expected_value['mask_sha']
                or actual['bits'] != expected_value['bits']):
            raise ValueError('a recorded control is missing or disagrees with mask digests')
        controls.append(record)

    for site, layer, token in SITES:
        donors = {}
        for row in rows:
            row_id = row['row_id']
            observed = forward('donor_capture', row_id, site, layer, token)
            donors[row_id] = observed['before']
            if row['subject_source'] == 'A':
                if site == 'candidate':
                    references[row_id] = observed
                    baselines[row_id] = observed['bits']
                else:
                    control('clean_matches_candidate', row_id, site, observed, references[row_id])
        for target in targets:
            target_id, case = target['target_row_id'], target['case_id']
            cells = {}
            for cell in ('A_same', 'A_opposite', 'B_same', 'B_opposite'):
                donor_id = target['donor_row_ids'][cell]
                observed = forward('patch_' + cell, target_id, site, layer, token,
                                   donor_id=donor_id, patch=True, self_patch=cell == 'A_same')
                if observed['before'] != donors[target_id]:
                    raise ValueError('recorded target pre-action vector differs from its clean capture')
                expected_source = observed['before'] if cell == 'A_same' else donors[donor_id]
                if observed['after'] != expected_source:
                    raise ValueError('recorded donor vector did not land')
                if cell == 'A_same':
                    control('self_patch', case, site, observed, references[target_id])
                cells[cell.lower()] = observed['bits']
            case_effects.setdefault(case, {})[site] = signed_donor_effect(by_id[target_id]['subject_number'], cells)

    final_sources = {'final_subject': {}, 'final_mask': {}}
    final_before = {'final_subject': {}, 'final_mask': {}}
    for target in targets:
        target_id, case = target['target_row_id'], target['case_id']
        donor_id = target['donor_row_ids']['A_opposite']
        for name, token in (('final_subject', 2), ('final_mask', 6)):
            donor = forward('delivery_donor', donor_id, name, 5, token)
            patched = forward('delivery_patch', target_id, name, 5, token, donor_id=donor_id, patch=True)
            final_sources[name][donor_id] = donor['before']
            final_before[name][target_id] = patched['before']
            if patched['after'] != donor['before']:
                raise ValueError('final-block donor vector did not land')
            expected_value = references[target_id] if name == 'final_subject' else donor
            control(name, case, name, patched, expected_value)
        fresh = forward('final_clean', target_id, 'candidate', 2, 2)
        control('final_clean', case, 'candidate', fresh, references[target_id])
    if any(final_sources[name] != final_before[name] for name in final_sources):
        raise ValueError('final-block target vectors do not match clean captures')
    if count != 140 or cursor != len(records):
        raise ValueError('family event stream has extra or missing observations')
    cases = [{'case_id': target['case_id'], 'target_row_id': target['target_row_id'],
              'baseline_is_bits': baselines[target['target_row_id']][0],
              'baseline_are_bits': baselines[target['target_row_id']][1],
              'effects': case_effects[target['case_id']]} for target in targets]
    return {'split': split, 'family': family, 'forward_count': count, 'cases': cases,
            'site_means': {site: math.fsum(case['effects'][site] for case in cases) / 4.0
                           for site, _, _ in SITES}, 'controls': controls}
