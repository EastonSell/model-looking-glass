"""Post-hoc analysis of saved vision-model probability changes."""
import argparse
import hashlib
import json
import lzma
import math
from pathlib import Path
import re
import statistics
import struct
import sys
import tarfile
import tracemalloc

ARCHIVE_SHA256 = '519d30de52b85194f97559a7164031b83f6465e39fb659b399965efedb3373fd'
ARCHIVE_BYTES = 13180400
INDEX_SHA256 = '36af760029eefdf9dbaad83b8d15a2219507f3c448cea08a9bb4348d171225c8'
MANIFEST_SHA256 = '8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207'
SOURCE_REVISION = 'b56d07e16d275a0f3ee9326bcd878277b89ec99d'
DECODER_MEMORY_LIMIT = 32 * 1024 * 1024
DECOMPRESSED_LIMIT = 128 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
_PAIR_IDS = tuple(f'pair-{i:02d}' for i in range(4, 12))
_KINDS = ('initial', 'matched_inside', 'matched_outside', 'matched_random')
_LABEL = re.compile(r'(pair-(?:0[4-9]|1[01])):(cat|dog):(initial|matched_inside|matched_outside|matched_random)\Z')


def _groups(cat_ids, dog_ids, size):
    groups = (cat_ids, dog_ids)
    if any(not isinstance(group, (tuple, list)) or not group for group in groups):
        raise ValueError('both class groups must be nonempty')
    if any(type(i) is not int or i < 0 or i >= size for group in groups for i in group):
        raise ValueError('class group index outside logits')
    if any(len(set(group)) != len(group) for group in groups) or set(cat_ids) & set(dog_ids):
        raise ValueError('class groups must be disjoint with unique indices')
    return set(cat_ids) | set(dog_ids)


def _probabilities(logits):
    if not isinstance(logits, (list, tuple)) or not logits:
        raise ValueError('a nonempty logit vector is required')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
        raise ValueError('logits must be finite numbers')
    top = max(logits)
    values = [math.exp(x - top) for x in logits]
    total = math.fsum(values)
    return [x / total for x in values]


def _group_log_odds_parts(logits, dog_ids, cat_ids):
    dog_top = max(logits[i] for i in dog_ids)
    cat_top = max(logits[i] for i in cat_ids)
    dog_sum = math.fsum(math.exp(logits[i] - dog_top) for i in dog_ids)
    cat_sum = math.fsum(math.exp(logits[i] - cat_top) for i in cat_ids)
    return dog_top, cat_top, math.log(dog_sum), math.log(cat_sum)


def _group_log_odds(logits, dog_ids, cat_ids):
    dog_top, cat_top, dog_sum, cat_sum = _group_log_odds_parts(logits, dog_ids, cat_ids)
    try:
        result = math.fsum((dog_top, -cat_top, dog_sum, -cat_sum))
    except OverflowError as exc:
        raise ValueError('group log odds exceed the numeric range') from exc
    if not math.isfinite(result):
        raise ValueError('group log odds exceed the numeric range')
    return result


def _group_log_odds_change(clean, patched, dog_ids, cat_ids):
    clean_dog, clean_cat, clean_dsum, clean_csum = _group_log_odds_parts(clean, dog_ids, cat_ids)
    patch_dog, patch_cat, patch_dsum, patch_csum = _group_log_odds_parts(patched, dog_ids, cat_ids)
    try:
        result = math.fsum((patch_dog, -clean_dog, clean_cat, -patch_cat,
                            patch_dsum, -clean_dsum, clean_csum, -patch_csum))
    except OverflowError as exc:
        raise ValueError('group log odds change exceeds the numeric range') from exc
    if not math.isfinite(result):
        raise ValueError('group log odds change exceeds the numeric range')
    return result


def compare_distribution(clean_logits, patched_logits, *, cat_ids, dog_ids, side):
    """Compare full output distributions and the original group log-odds."""
    if side not in ('cat', 'dog') or len(clean_logits) != len(patched_logits):
        raise ValueError('matching logits and a cat or dog receiver are required')
    clean, patched = _probabilities(clean_logits), _probabilities(patched_logits)
    animal = _groups(cat_ids, dog_ids, len(clean))
    changes = [b - a for a, b in zip(clean, patched)]
    absolute = math.fsum(abs(x) for x in changes)
    off_abs = math.fsum(abs(x) for i, x in enumerate(changes) if i not in animal)
    animal_change = math.fsum(changes[i] for i in animal)
    off_change = math.fsum(x for i, x in enumerate(changes) if i not in animal)
    orientation = 1.0 if side == 'cat' else -1.0
    clean_top1 = max(range(len(clean_logits)), key=clean_logits.__getitem__)
    patched_top1 = max(range(len(patched_logits)), key=patched_logits.__getitem__)
    return {
        'total_variation': absolute / 2.0,
        'animal_mass_change': animal_change,
        'offgroup_mass_change': off_change,
        'offgroup_absolute_change_share': off_abs / absolute if absolute else None,
        'annotation_oriented_margin_change': orientation * _group_log_odds_change(
            clean_logits, patched_logits, dog_ids, cat_ids),
        'clean_top1_id': clean_top1,
        'patched_top1_id': patched_top1,
        'top1_changed': clean_top1 != patched_top1,
    }


def displacement_cosine(clean, first, second):
    """Cosine of two probability displacement vectors; zero displacement is undefined."""
    if not len(clean) or len(clean) != len(first) or len(clean) != len(second):
        raise ValueError('equal nonempty probability vectors are required')
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for vector in (clean, first, second) for x in vector):
        raise ValueError('probability vectors must be finite')
    a = [x - p for p, x in zip(clean, first)]
    b = [x - p for p, x in zip(clean, second)]
    if not all(math.isfinite(x) for x in a + b):
        raise ValueError('probability displacement exceeds the numeric range')
    scale_a, scale_b = max(abs(x) for x in a), max(abs(x) for x in b)
    if scale_a == 0.0 or scale_b == 0.0:
        return None
    a, b = [x / scale_a for x in a], [x / scale_b for x in b]
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
    value = math.fsum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    return max(-1.0, min(1.0, value))


def summarize_rows(rows):
    """Report descriptive receiver-side summaries, separated by the baseline gate."""
    if type(rows) is not list or len(rows) > 16:
        raise ValueError('at most 16 receiver sides are supported')
    if len({(row['pair_id'], row['side']) for row in rows}) != len(rows):
        raise ValueError('duplicate receiver side')
    if any(type(row['eligible']) is not bool for row in rows):
        raise ValueError('each baseline eligibility must be boolean')
    result = {}
    for name, eligible in (('evaluable', True), ('baseline_ineligible', False)):
        selected = [row for row in rows if row['eligible'] is eligible]
        block = {'sides': len(selected), 'pair_count': len({row['pair_id'] for row in selected})}
        for condition in ('matched_inside', 'matched_outside', 'matched_random'):
            entries = [row['conditions'][condition] for row in selected]
            shares = [entry['offgroup_absolute_change_share'] for entry in entries
                      if entry['offgroup_absolute_change_share'] is not None]
            block[condition] = {
                'median_total_variation': statistics.median([entry['total_variation'] for entry in entries])
                                          if entries else None,
                'median_annotation_oriented_margin_change':
                    statistics.median([entry['annotation_oriented_margin_change'] for entry in entries])
                    if entries else None,
                'median_offgroup_absolute_change_share': statistics.median(shares) if shares else None,
                'undefined_offgroup_share_count': len(entries) - len(shares),
            }
        block['inside_tv_exceeds_outside'] = sum(
            row['conditions']['matched_inside']['total_variation'] >
            row['conditions']['matched_outside']['total_variation'] for row in selected)
        block['inside_tv_exceeds_random'] = sum(
            row['conditions']['matched_inside']['total_variation'] >
            row['conditions']['matched_random']['total_variation'] for row in selected)
        block['random_margin_at_least_inside'] = sum(
            row['conditions']['matched_random']['annotation_oriented_margin_change'] >=
            row['conditions']['matched_inside']['annotation_oriented_margin_change'] for row in selected)
        for control in ('outside', 'random'):
            values = [row[f'inside_vs_{control}_probability_displacement_cosine'] for row in selected]
            defined = [value for value in values if value is not None]
            block[f'inside_vs_{control}_cosine'] = {
                'median': statistics.median(defined) if defined else None,
                'undefined_count': len(values) - len(defined),
            }
        result[name] = block
    return result


class BoundedXZReader:
    """Incrementally decompress one XZ stream with explicit memory and output caps."""

    def __init__(self, source, *, decompressed_limit, memory_limit):
        if type(decompressed_limit) is not int or decompressed_limit <= 0:
            raise ValueError('positive decompressed limit required')
        if type(memory_limit) is not int or memory_limit <= 0:
            raise ValueError('positive decoder memory limit required')
        self._source = source
        self._decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=memory_limit)
        self._limit = decompressed_limit
        self.decompressed_bytes = 0
        self._finished = False

    def read(self, size):
        if type(size) is not int or size < 0 or size > 1024 * 1024:
            raise ValueError('bounded read size required')
        if size == 0 or self._finished:
            return b''
        output = bytearray()
        while len(output) < size and not self._finished:
            if self._decoder.needs_input:
                chunk = self._source.read(65536)
                if not chunk:
                    raise ValueError('truncated XZ stream')
            else:
                chunk = b''
            try:
                data = self._decoder.decompress(
                    chunk, max_length=min(size - len(output), self._limit - self.decompressed_bytes + 1))
            except lzma.LZMAError as exc:
                raise ValueError('invalid XZ stream or decoder memory limit exceeded') from exc
            self.decompressed_bytes += len(data)
            if self.decompressed_bytes > self._limit:
                raise ValueError('decompressed data exceeds limit')
            output.extend(data)
            if self._decoder.eof:
                if self._decoder.unused_data or self._source.read(1):
                    raise ValueError('trailing data after XZ stream')
                self._finished = True
        return bytes(output)


def scan_verified_tar(reader, expected_members):
    """Yield bounded regular members only after their indexed hashes match."""
    if type(expected_members) is not dict or len(expected_members) > 709:
        raise ValueError('bounded member inventory required')
    seen = set()
    try:
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name in seen or member.name not in expected_members:
                    raise ValueError('unexpected, duplicate or non-file archive member')
                expected_size, expected_hash = expected_members[member.name]
                if (type(expected_size) is not int or not 0 <= expected_size <= 512 * 1024
                        or member.size != expected_size):
                    raise ValueError('archive member size differs or exceeds limit')
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    raise ValueError('archive member cannot be read')
                data = file_obj.read(expected_size + 1)
                if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_hash:
                    raise ValueError('archive member hash or length differs')
                seen.add(member.name)
                yield member.name, data
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError('invalid or truncated tar stream') from exc
    if seen != set(expected_members):
        raise ValueError('archive member inventory is incomplete')
    while reader.read(65536):
        pass


def _read_pinned_json(path, digest):
    with open(path, 'rb') as source:
        data = source.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('input JSON length or pinned SHA-256 differs')
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid input JSON') from exc


def _hash_archive(source):
    source.seek(0)
    digest, total = hashlib.sha256(), 0
    while True:
        block = source.read(65536)
        if not block:
            break
        total += len(block)
        if total > ARCHIVE_BYTES:
            raise ValueError('compressed archive exceeds pinned size')
        digest.update(block)
    if total != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError('archive length or pinned SHA-256 differs')
    source.seek(0)


def _saved_score(logits, cat_ids, dog_ids):
    probabilities = _probabilities(logits)
    animal = _groups(cat_ids, dog_ids, len(logits))
    return {
        'cat_mass': math.fsum(probabilities[i] for i in cat_ids),
        'dog_mass': math.fsum(probabilities[i] for i in dog_ids),
        'off_mass': math.fsum(p for i, p in enumerate(probabilities) if i not in animal),
        'margin': _group_log_odds(logits, dog_ids, cat_ids),
        'top1_id': max(range(len(logits)), key=logits.__getitem__),
    }


def _decode_f32_logits(data):
    """Decode finite IEEE-754 binary32 values via integer bits and binary64 math."""
    if len(data) != 4000:
        raise ValueError('expected 1000 binary32 logits')
    result = []
    for bits in struct.unpack('<1000I', data):
        exponent, fraction = (bits >> 23) & 0xff, bits & 0x7fffff
        if exponent == 0xff:
            raise ValueError('saved logits must be finite')
        magnitude = (math.ldexp(fraction, -149) if exponent == 0 else
                     math.ldexp((1 << 23) | fraction, exponent - 150))
        result.append(-magnitude if bits >> 31 else magnitude)
    return tuple(result)


def _check_score(actual, saved):
    for key in ('cat_mass', 'dog_mass', 'off_mass', 'margin'):
        if type(saved.get(key)) not in (int, float) or not math.isfinite(saved[key]) or not math.isclose(
                actual[key], saved[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('saved class score does not match retained logits')
    if actual['top1_id'] != saved.get('top1_id'):
        raise ValueError('saved top class does not match retained logits')


def _validated_dose(dose, condition):
    """Check the recorded matched dose; this does not replay interventions."""
    if dose.get('status') != 'matchable':
        raise ValueError('intervention was not marked matchable')
    values = {
        'planned_q': dose.get('q'),
        'dose_tolerance': dose.get('tolerance'),
        'actual_dose': condition.get('actual_dose'),
        'receiver_norm': condition.get('receiver_norm'),
        'q_over_receiver_norm': condition.get('q_over_receiver_norm'),
    }
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in values.values()):
        raise ValueError('invalid saved intervention dose')
    if abs(values['actual_dose'] - values['planned_q']) > values['dose_tolerance']:
        raise ValueError('saved intervention dose does not match planned dose')
    if not math.isclose(values['q_over_receiver_norm'],
                        values['planned_q'] / values['receiver_norm'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('saved intervention-to-receiver ratio differs')
    return values


def _index_members(index):
    if index.get('source_revision') != SOURCE_REVISION or type(index.get('members')) is not list:
        raise ValueError('evidence index source or schema differs')
    if len(index['members']) != 708:
        raise ValueError('unexpected evidence member count')
    members = {}
    for item in index['members']:
        path, size, digest = item.get('path'), item.get('bytes'), item.get('sha256')
        if (type(path) is not str or path in members or type(size) is not int or
                not 0 <= size <= 512 * 1024 or type(digest) is not str or
                re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('invalid indexed member')
        members[path] = (size, digest)
    members['evidence-index.json'] = (93485, INDEX_SHA256)
    return members


def _collect_heldout(reader, members):
    pending, selected, pairs = {}, {}, {}
    forward_count = 0
    for name, data in scan_verified_tar(reader, members):
        match = re.fullmatch(r'heldout/data/forward-(\d{3})\.(bin|json)', name)
        if match:
            number, extension = int(match.group(1)), match.group(2)
            if not 1 <= number <= 228:
                raise ValueError('held-out forward identity outside fixed inventory')
            if extension == 'bin':
                if number in pending or len(data) < 4000 or len(pending) >= 228:
                    raise ValueError('invalid or excessive pending forward data')
                pending[number] = (data[:4000], hashlib.sha256(data).hexdigest())
                continue
            if len(data) > MAX_JSON_BYTES or number not in pending:
                raise ValueError('forward metadata has no matching bounded binary')
            info = json.loads(data)
            head, digest = pending.pop(number)
            if (info.get('submission') != number or info.get('file') != f'forward-{number:03d}.bin'
                    or info.get('sha256') != digest or info.get('dtype') != 'little-endian-float32'
                    or info.get('layout', {}).get('logits') != {'bytes': 4000, 'offset': 0, 'shape': [1, 1000]}):
                raise ValueError('forward metadata and binary disagree')
            forward_count += 1
            label = info.get('label')
            matched = _LABEL.fullmatch(label) if type(label) is str else None
            if matched:
                key = matched.groups()
                if key in selected or len(selected) >= 64:
                    raise ValueError('duplicate or excessive selected forward')
                selected[key] = (head, info['score'], number, digest)
            continue
        match = re.fullmatch(r'heldout/data/(pair-(?:0[4-9]|1[01]))\.json', name)
        if match:
            if len(data) > MAX_JSON_BYTES or match.group(1) in pairs:
                raise ValueError('duplicate or oversized pair record')
            pair = json.loads(data)
            if pair.get('pair_id') != match.group(1) or pair.get('split') != 'heldout':
                raise ValueError('pair identity or split differs')
            pairs[match.group(1)] = pair
    expected = {(pair, side, kind) for pair in _PAIR_IDS for side in ('cat', 'dog') for kind in _KINDS}
    if pending or forward_count != 228 or set(selected) != expected or set(pairs) != set(_PAIR_IDS):
        raise ValueError('held-out forward or pair inventory is incomplete')
    return selected, pairs


def _rows(selected, pairs, cat_ids, dog_ids):
    rows = []
    for pair_id in _PAIR_IDS:
        pair = pairs[pair_id]
        for side in ('cat', 'dog'):
            clean_head, clean_saved, clean_submission, clean_sha = selected[(pair_id, side, 'initial')]
            clean = _decode_f32_logits(clean_head)
            baseline = _saved_score(clean, cat_ids, dog_ids)
            _check_score(baseline, clean_saved)
            _check_score(baseline, pair['sides'][side]['baseline'])
            conditions, probabilities = {}, {}
            for kind in _KINDS[1:]:
                head, saved, submission, sha = selected[(pair_id, side, kind)]
                logits = _decode_f32_logits(head)
                score = _saved_score(logits, cat_ids, dog_ids)
                _check_score(score, saved)
                expected = pair['sides'][side]['conditions'][kind]
                _check_score(score, expected)
                metric = compare_distribution(clean, logits, cat_ids=cat_ids, dog_ids=dog_ids, side=side)
                if not math.isclose(metric['annotation_oriented_margin_change'],
                                    expected['annotation_oriented_change'], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('saved margin change does not match logits')
                metric.update(_validated_dose(pair['sides'][side]['dose'], expected))
                metric['submission'] = submission
                metric['binary_sha256'] = sha
                conditions[kind] = metric
                probabilities[kind] = _probabilities(logits)
            p = _probabilities(clean)
            rows.append({
                'pair_id': pair_id, 'side': side, 'eligible': pair['baseline_eligibility']['eligible'],
                'pair_status': pair['status'], 'baseline': baseline, 'clean_submission': clean_submission,
                'clean_binary_sha256': clean_sha, 'conditions': conditions,
                'probabilities': {'clean': p, **probabilities},
                'inside_vs_outside_probability_displacement_cosine': displacement_cosine(
                    p, probabilities['matched_inside'], probabilities['matched_outside']),
                'inside_vs_random_probability_displacement_cosine': displacement_cosine(
                    p, probabilities['matched_inside'], probabilities['matched_random']),
            })
        cat, dog = rows[-2]['baseline'], rows[-1]['baseline']
        eligible = (cat['margin'] < 0 and dog['margin'] > 0 and dog['margin'] - cat['margin'] >= 1
                    and cat['cat_mass'] >= .2 and dog['dog_mass'] >= .2)
        if type(pair['baseline_eligibility']['eligible']) is not bool or eligible != pair['baseline_eligibility']['eligible']:
            raise ValueError('saved baseline eligibility differs from recomputed gate')
    return rows


def analyze_saved_evidence(archive_path, index_path, manifest_path):
    """Recompute fixed post-hoc contrasts from the pinned, bounded public archive."""
    index = _read_pinned_json(index_path, INDEX_SHA256)
    manifest = _read_pinned_json(manifest_path, MANIFEST_SHA256)
    cat_ids = tuple(item['id'] for item in manifest['class_groups']['cat'])
    dog_ids = tuple(item['id'] for item in manifest['class_groups']['dog'])
    if len(cat_ids) != 5 or len(dog_ids) != 118:
        raise ValueError('class group sizes differ from the public protocol')
    _groups(cat_ids, dog_ids, 1000)
    members = _index_members(index)
    with open(archive_path, 'rb') as source:
        _hash_archive(source)
        reader = BoundedXZReader(source, decompressed_limit=DECOMPRESSED_LIMIT,
                                 memory_limit=DECODER_MEMORY_LIMIT)
        selected, pairs = _collect_heldout(reader, members)
        decompressed_bytes = reader.decompressed_bytes
        _hash_archive(source)
    rows = _rows(selected, pairs, cat_ids, dog_ids)
    return {
        'schema': 'vision-coco-distribution-posthoc-v1',
        'scope': 'Post-hoc descriptive analysis of public saved logits; no model execution or population inference.',
        'source_revision': SOURCE_REVISION,
        'inputs': {'archive_sha256': ARCHIVE_SHA256, 'archive_bytes': ARCHIVE_BYTES,
                   'index_sha256': INDEX_SHA256, 'manifest_sha256': MANIFEST_SHA256},
        'limits': {'decoder_memory_bytes': DECODER_MEMORY_LIMIT, 'decompressed_bytes': DECOMPRESSED_LIMIT,
                   'max_member_bytes': 512 * 1024, 'max_selected_forwards': 64,
                   'max_pending_logit_bytes': 228 * 4000,
                   'max_report_probability_values': 64 * 1000,
                   'note': 'Logical/decoder caps; not a physical RSS bound.'},
        'verification': {'indexed_members': len(members), 'heldout_forward_records': 228,
                         'selected_forward_records': len(selected), 'decompressed_bytes': decompressed_bytes,
                         'archive_hash_checked_before_and_after': True},
        'rows': rows, 'summary': summarize_rows(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    tracemalloc.start()
    report = analyze_saved_evidence(args.archive, args.index, args.manifest)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report['execution'] = {
        'python_version': sys.version.split()[0],
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_traced_heap_peak_bytes': peak,
        'memory_note': 'Tracemalloc excludes report serialization, native allocations, decoder state and interpreter baseline.',
    }
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with open(args.output, 'xb') as target:
        target.write(data)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
