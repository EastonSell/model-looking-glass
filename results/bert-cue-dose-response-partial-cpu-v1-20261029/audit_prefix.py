"""Audit an incomplete two-dose journal prefix without inferring a study result.

This post-stop reader reuses the pre-frozen full-witness reader's numerical
primitives, but does not invent the absent result.json or claim independent
implementation. It validates every available record up to an orderly end,
then reports complete cells separately from any unfinished cell.
"""
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re

FULL_READER_SHA = 'bd7a2cba0cb38c5def0e6642d479aeab6c2d1aec373a545651ee7e552d027c2d'


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_reader(path):
    if hash_file(path) != FULL_READER_SHA:
        raise ValueError('full-witness reader bytes differ')
    spec = importlib.util.spec_from_file_location('frozen_dose_full', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrefixEnd(Exception):
    pass


def replay(full, manifest, stream, width, vocab):
    if (len(manifest['cases']) != 8 or
            [(c['triple_id'], c['cue_set'], c['split'], c['frame']) for c in manifest['cases']] !=
            [(t, g, 'calibration' if t == full.CALIBRATION else 'evaluation', 'main')
             for t in (full.CALIBRATION, *full.TRIPLES) for g in full.GROUPS] or
            len({c['id'] for c in manifest['cases']}) != 8):
        raise ValueError('fixed manifest schedule differs')
    events = 0
    forwards = 0
    expected_next = None
    completed = []
    active_id = None
    partial = {}
    case_start_events = 0
    case_start_forwards = 0

    def take(kind, case_id, **identity):
        nonlocal events, forwards, expected_next
        expected_next = {'kind': kind, 'case_id': case_id, **identity}
        line = stream.readline(2 * 1024 * 1024 + 1)
        if not line:
            raise PrefixEnd
        if len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n'):
            raise ValueError('oversized or unterminated event')
        row = full.strict_json(line)
        if (row.get('kind') != kind or row.get('case_id') != case_id or
                any(row.get(k) != v for k, v in identity.items())):
            raise ValueError('saved event identity or order differs')
        events += 1
        if kind in ('clean', 'self', 'transfer', 'repeat'):
            forwards += 1
        return row

    def witness(row, token, length):
        before = full.raw(row['before_f32le'], width)
        after = full.raw(row['after_f32le'], width)
        a = full.raw(row['full_before_f32le'], length * width)
        b = full.raw(row['full_after_f32le'], length * width)
        lo, hi = token * width * 4, (token + 1) * width * 4
        if a[lo:hi] != before or b[lo:hi] != after or a[:lo] != b[:lo] or a[hi:] != b[hi:]:
            raise ValueError('selected or unselected witness bits differ')
        return before, after

    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if any_id == some_id or min(any_id, some_id) < 0 or max(any_id, some_id) >= vocab:
        raise ValueError('invalid scored-token identities')
    try:
        for case in manifest['cases']:
            active_id = case['id']
            partial = {'case_id': active_id}
            case_start_events, case_start_forwards = events, forwards
            cues = full.GROUPS[case['cue_set']]
            neg, aff = cues[:2], cues[2:]
            indices = case['indices']
            ids = {cue: case['rows'][cue]['input_ids'] for cue in cues}
            length = len(ids[cues[0]])
            if (set(case['rows']) != set(cues) or set(indices) != {'cue', 'mask', 'object'} or
                    not 1 <= length <= 64 or any(len(row) != length for row in ids.values()) or
                    len(set(indices.values())) != 3 or
                    any(type(i) is not int or not 0 <= i < length for i in indices.values())):
                raise ValueError('invalid sequence or selected positions')
            pos = indices['cue']
            if (len({ids[c][pos] for c in cues}) != 4 or
                    any(ids[c][:pos] + ids[c][pos + 1:] != ids[cues[0]][:pos] + ids[cues[0]][pos + 1:]
                        or case['rows'][c]['attention_mask'] != [1] * length for c in cues) or
                    ('cue_ids' in manifest and any(ids[c][pos] != manifest['cue_ids'][c] for c in cues))):
                raise ValueError('unaligned case rows')
            positions = {'cue2': indices['cue'], 'object2': indices['object'],
                         'mask2': indices['mask'], 'cue5': indices['cue']}
            clean, captures, margins, baselines = {}, {}, {}, {}
            for cue in cues:
                row = take('clean', active_id, cue=cue, input_ids=ids[cue])
                clean[cue] = full.raw(row['mask_logits_f32le'], vocab)
                if set(row['captures_f32le']) != set(full.SITES):
                    raise ValueError('clean capture sites differ')
                captures[cue] = {s: full.raw(row['captures_f32le'][s], width) for s in full.SITES}
                margins[cue] = full.margin(clean[cue], any_id, some_id)
                baselines[cue] = full.baseline(clean[cue], any_id, some_id)
            gaps = [{'negative': n, 'affirmative': a, 'gap': math.fsum((margins[n], -margins[a]))}
                    for n in neg for a in aff]
            baseline_eligible = all(g['gap'] > 0.5 for g in gaps)
            row = take('baseline', active_id)
            if type(row['eligible']) is not bool or row['eligible'] != baseline_eligible or row['gaps'] != gaps:
                raise ValueError('baseline screen differs')
            partial.update({'baseline_eligible': baseline_eligible, 'baseline_gaps': gaps})
            distances = [{'left': a, 'right': b, 'site': s, 'distance': full.norm(captures[a][s], captures[b][s])}
                         for a, b in itertools.combinations(cues, 2) for s in full.SITES]
            minimum = min(d['distance'] for d in distances)
            eligible = {label: minimum >= radius for label, radius in full.DOSES.items()}
            row = take('radius', active_id)
            if (row['distances'] != distances or row['minimum_distance'] != minimum or
                    row['binding_edges'] != [d for d in distances if d['distance'] == minimum] or
                    row['doses'] != {label: {'radius': radius, 'eligible': eligible[label]}
                                     for label, radius in full.DOSES.items()} or
                    any(type(row['doses'][label]['eligible']) is not bool for label in full.DOSES)):
                raise ValueError('dose eligibility or distances differ')
            partial.update({'minimum_distance': minimum, 'dose_eligibility': eligible})
            for label, radius in full.DOSES.items():
                if not eligible[label]:
                    take('dose_ineligible', active_id, dose=label, radius=radius, minimum_distance=minimum)
            details = {label: {} for label in full.DOSES if eligible[label]}
            geometry = {label: {'fraction_min': None, 'fraction_max': None,
                                'actual_norm_min': None, 'actual_norm_max': None,
                                'relative_error_max': None} for label in full.DOSES if eligible[label]}
            if eligible['low']:
                for site in full.SITES:
                    for cue in cues:
                        row = take('self', active_id, site=site, target=cue, input_ids=ids[cue])
                        before, after = witness(row, positions[site], length)
                        if (before != captures[cue][site] or after != before or
                                full.raw(row['mask_logits_f32le'], vocab) != clean[cue]):
                            raise ValueError('self control differs')
                    for label, radius in full.DOSES.items():
                        if not eligible[label]:
                            continue
                        effects = {}
                        for target in cues:
                            for donor in cues:
                                if donor == target:
                                    continue
                                row = take('transfer', active_id, site=site, dose=label,
                                           donor=donor, target=target, input_ids=ids[target])
                                before, after = witness(row, positions[site], length)
                                target_clean, source = captures[target][site], captures[donor][site]
                                distance = full.norm(target_clean, source)
                                if before != target_clean or distance <= 0 or radius > distance:
                                    raise ValueError('pre-state or nonamplification differs')
                                fraction = radius / distance
                                submitted = full.raw(row['submitted_f32le'], width)
                                if submitted != full.patch(target_clean, source, fraction) or after != submitted:
                                    raise ValueError('submitted or landed vector differs')
                                planned, actual = full.norm(target_clean, submitted), full.norm(before, after)
                                error = abs(actual - radius) / radius
                                if (row['requested_radius'] != radius or row['requested_fraction'] != fraction or
                                        row['planned_norm'] != planned or row['actual_norm'] != actual or
                                        row['relative_error'] != error or error > .001):
                                    raise ValueError('actual or planned dose differs')
                                stats = geometry[label]
                                for key, value, choose in (('fraction_min', fraction, min), ('fraction_max', fraction, max),
                                                           ('actual_norm_min', actual, min), ('actual_norm_max', actual, max),
                                                           ('relative_error_max', error, max)):
                                    stats[key] = value if stats[key] is None else choose(stats[key], value)
                                logits = full.raw(row['mask_logits_f32le'], vocab)
                                if site == 'cue5' and logits != clean[target]:
                                    raise ValueError('final cue changed full logits')
                                effects[donor, target] = math.fsum((full.margin(logits, any_id, some_id),
                                                                    -margins[target]))
                        details[label][site] = full.specificity(effects, cues)
            for cue in cues:
                row = take('repeat', active_id, cue=cue, input_ids=ids[cue])
                if (full.raw(row['mask_logits_f32le'], vocab) != clean[cue] or
                        set(row['captures_f32le']) != set(full.SITES) or
                        any(full.raw(row['captures_f32le'][s], width) != captures[cue][s] for s in full.SITES)):
                    raise ValueError('clean repeat differs')
            doses = {}
            for label in full.DOSES:
                if eligible[label]:
                    targets = details[label]['cue2']['targets']
                    value = min(item['margin'] for item in targets.values())
                    doses[label] = {'eligible': True, 'cue_min_margin': value,
                                    'limiting_targets': [c for c in cues if targets[c]['margin'] == value],
                                    'specificity_passed': details[label]['cue2']['passed'],
                                    'site_specificity': details[label], 'geometry': geometry[label]}
                else:
                    doses[label] = {'eligible': False, 'cue_min_margin': None,
                                    'limiting_targets': [], 'specificity_passed': None,
                                    'site_specificity': {}, 'geometry': None}
            completed.append({'case_id': active_id, 'split': case['split'], 'baseline_eligible': baseline_eligible,
                              'baseline_gaps': gaps, 'clean_baselines': baselines,
                              'minimum_distance': minimum, 'doses': doses})
    except PrefixEnd:
        partial.update({'validated_events': events - case_start_events,
                        'saved_forwards': forwards - case_start_forwards,
                        'dose_outcomes': 'not_computed_for_incomplete_case'})
        return {'status': 'incomplete_prefix_validated', 'events': events, 'saved_forwards': forwards,
                'completed_cases': completed, 'partial_case_id': active_id, 'partial_case': partial,
                'next_expected': expected_next,
                'study_outcome': 'incomplete'}
    if stream.read(1):
        raise ValueError('extra event after completed schedule')
    return {'status': 'complete_schedule_validated_without_result', 'events': events,
            'saved_forwards': forwards, 'completed_cases': completed,
            'partial_case_id': None, 'next_expected': None, 'study_outcome': 'not_attested'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--started', type=Path, required=True)
    parser.add_argument('--failure', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    full = load_reader(args.reader)
    full.numeric_mode()
    expected = {'manifest': full.MANIFEST_HASH,
                'events': 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05',
                'started': '19baa22fc6b6fe3da139bf22b125a4f3122a39e6e82dc9ec38604389127bd9e7',
                'failure': '12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a',
                'receipt': 'f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34'}
    paths = {name: getattr(args, name) for name in expected}
    before = {name: full.sha(path) for name, path in paths.items()}
    if before != expected or (args.events.parent / 'result.json').exists():
        raise ValueError('closed attempt identity or missing-result status differs')
    manifest = full.strict_json(args.manifest.read_bytes())
    started = full.strict_json(args.started.read_bytes())
    failure = full.strict_json(args.failure.read_bytes())
    receipt = full.strict_json(args.receipt.read_bytes())
    if (started.get('source_commit') != full.SOURCE_COMMIT or started.get('source_sha256') != full.SOURCE_HASHES or
            started.get('assets') != full.ASSETS or started.get('versions') != full.VERSIONS or
            started.get('seed') != 2026102905 or started.get('maximum_forwards') != 960 or
            not full.valid_cpu_environment(started.get('cpu_environment')) or
            failure.get('source_commit') != full.SOURCE_COMMIT or failure.get('status') != 'incomplete' or
            failure.get('phase') != 'forwards' or failure.get('error_type') != 'RuntimeError' or
            failure.get('message') != 'disk reserve exhausted before the next evidence event' or
            receipt.get('source_commit') != full.SOURCE_COMMIT or receipt.get('process_waited') is not True or
            receipt.get('exit_code') != 1):
        raise ValueError('recorded source, startup or failure metadata differs')
    with gzip.open(args.events, 'rb') as stream:
        report = replay(full, manifest, stream, 768, 30522)
    if (hash_file(args.reader) != FULL_READER_SHA or
            {name: full.sha(path) for name, path in paths.items()} != before):
        raise ValueError('an input changed during replay')
    full.numeric_mode()
    report.update({'input_sha256': before, 'scope': 'saved prefix consistency only; no complete-study result or execution attestation',
                   'failure': 'disk-headroom event guard', 'returned_unrecorded_forward': 'operator reports one; not present in this journal'})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
