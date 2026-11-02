"""Pure public-pool helpers for a proposed natural-agreement successor.

These functions generate no nonce, read no private split, load no model and
cannot establish source publication, exposure currentness, reservation
ownership or permission to execute. A separately reviewed controller must
establish those conditions before using them in a one-shot protocol.
"""
import hashlib
import hmac
import json
import re


TARGETS = {' is': 318, ' are': 389, ' was': 373, ' were': 547}
_ID = re.compile(r'[A-Za-z0-9._-]{1,128}\Z')
_WORD = re.compile(r'[a-z0-9]{1,64}\Z')
_CONTEXT = re.compile(r'[a-z0-9]+(?: [a-z0-9]+){0,15}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_DOMAIN = b'MLG-natural-successor-v1/'
_COMMIT_DOMAIN = b'MLG-natural-successor-v1/heldout-commit\0'
PREDICTION_FREEZE_SHA256 = '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb'
CHECKPOINT_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'


def _valid_tokens(value):
    return (type(value) is list and 1 <= len(value) <= 32
            and all(type(token) is int and 0 <= token <= 50256 for token in value))


def _pool(value):
    if (type(value) is not dict or set(value) != {'schema', 'nouns', 'contexts'}
            or value['schema'] != 'mlg-natural-successor-pool-v1'
            or type(value['nouns']) is not list or not 16 <= len(value['nouns']) <= 64
            or type(value['contexts']) is not list or not 8 <= len(value['contexts']) <= 32):
        raise ValueError('invalid candidate pool')
    ids, subjects, contexts = set(), set(), set()
    numbers = {'singular': 0, 'plural': 0}
    for noun in value['nouns']:
        if (type(noun) is not dict or set(noun) != {'id', 'singular', 'plural'}
                or type(noun['id']) is not str or _ID.fullmatch(noun['id']) is None
                or noun['id'] in ids):
            raise ValueError('invalid noun identity')
        ids.add(noun['id'])
        for key in ('singular', 'plural'):
            word = noun[key]
            if type(word) is not str or _WORD.fullmatch(word) is None or word in subjects:
                raise ValueError('duplicate or invalid subject')
            subjects.add(word)
    for context in value['contexts']:
        if (type(context) is not dict or set(context) != {'id', 'number', 'text'}
                or type(context['id']) is not str or _ID.fullmatch(context['id']) is None
                or context['id'] in ids or type(context['number']) is not str
                or context['number'] not in numbers
                or type(context['text']) is not str or len(context['text']) > 256
                or _CONTEXT.fullmatch(context['text']) is None or context['text'] in contexts):
            raise ValueError('invalid context identity')
        ids.add(context['id'])
        contexts.add(context['text'])
        numbers[context['number']] += 1
    if min(numbers.values()) < 4:
        raise ValueError('candidate contexts are not balanced')
    return value


def _ordered(items, nonce, name):
    domain = _DOMAIN + name.encode('ascii') + b'\0'
    def key(item):
        identity = item['id'].encode('ascii')
        return hashlib.sha256(domain + nonce + len(identity).to_bytes(2, 'big') + identity).digest(), identity
    return sorted(items, key=key)


def allocate(pool, nonce):
    """Derive four balanced families from an already fixed public pool."""
    pool = _pool(pool)
    if type(nonce) is not bytes or len(nonce) != 32:
        raise ValueError('nonce must be exactly 32 bytes')
    nouns = _ordered(pool['nouns'], nonce, 'noun')[:8]
    selected = []
    for number in ('singular', 'plural'):
        selected += _ordered([x for x in pool['contexts'] if x['number'] == number],
                             nonce, number + '-context')[:2]
    contexts = _ordered(selected, nonce, 'family')
    output = []
    for index, context in enumerate(contexts):
        first, second = nouns[2 * index:2 * index + 2]
        subjects = [first['singular'], first['plural'], second['singular'], second['plural']]
        output.append({'family_index': index, 'context_id': context['id'],
                       'context': context['text'], 'noun_ids': [first['id'], second['id']],
                       'subjects': subjects,
                       'texts': ['The ' + subject + ' ' + context['text'] for subject in subjects]})
    return output


def tokenize_candidate_pool(pool, tokenizer):
    """Check every selectable public prompt, including appended continuations."""
    pool = _pool(pool)
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    prefix = encode('The')
    if not _valid_tokens(prefix):
        raise ValueError('invalid prefix tokens')
    tokens, subject_ids = {}, set()
    for noun in pool['nouns']:
        for number in ('singular', 'plural'):
            subject = noun[number]
            ids = encode('The ' + subject)
            if (not _valid_tokens(ids) or len(ids) != len(prefix) + 1 or ids[:-1] != prefix
                    or ids[-1] in subject_ids):
                raise ValueError('invalid or duplicate subject token')
            subject_ids.add(ids[-1])
            tokens[subject] = ids
    rows = []
    for context in pool['contexts']:
        suffix = encode(' ' + context['text'])
        if not _valid_tokens(suffix):
            raise ValueError('invalid context tokens')
        for noun in pool['nouns']:
            for number in ('singular', 'plural'):
                subject = noun[number]
                text = 'The ' + subject + ' ' + context['text']
                ids = encode(text)
                if (not _valid_tokens(ids) or ids != tokens[subject] + suffix
                        or not 0 < len(prefix) < len(ids) - 1):
                    raise ValueError('invalid prompt alignment')
                for continuation, target in TARGETS.items():
                    if encode(text + continuation) != ids + [target]:
                        raise ValueError('invalid appended continuation')
                rows.append(tuple(ids))
    if len(rows) != len(set(rows)):
        raise ValueError('duplicate candidate prompt')
    return tuple(rows)


def reject_known_rows(candidate_rows, records, *, owned_records=None):
    """Reject full-row prefix relations after record-level own filtering.

    The controller must independently establish that every supplied owned
    record really belongs to this study and is still unforwarded. Exact record
    hashes and the planned category are checked here; they do not authenticate
    that ownership claim.
    """
    if (type(candidate_rows) not in (list, tuple) or not 1 <= len(candidate_rows) <= 4096
            or type(records) is not list or len(records) > 4096):
        raise ValueError('invalid row inventory')
    if owned_records is None:
        owned_records = {}
    if (type(owned_records) is not dict or len(owned_records) > 4096
            or any(type(key) is not str or type(value) is not str or _SHA.fullmatch(value) is None
                   for key, value in owned_records.items())):
        raise ValueError('invalid owned-record binding')
    candidates = []
    for row in candidate_rows:
        if type(row) not in (list, tuple) or not _valid_tokens(list(row)):
            raise ValueError('invalid candidate tokens')
        candidates.append(tuple(row))
    seen, filtered, owned_seen = set(), [], set()
    for record in records:
        if (type(record) is not dict or type(record.get('record_id')) is not str
                or not 1 <= len(record['record_id']) <= 256 or record['record_id'] in seen
                or type(record.get('category')) is not str or not _valid_tokens(record.get('token_ids'))
                or ('target_token_id' in record and (type(record['target_token_id']) is not int
                                                      or not 0 <= record['target_token_id'] <= 50256))):
            raise ValueError('invalid known record')
        identity = record['record_id']
        seen.add(identity)
        if identity in owned_records:
            try:
                raw = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                 allow_nan=False).encode('ascii')
            except (TypeError, ValueError, UnicodeError) as error:
                raise ValueError('invalid owned record') from error
            if (record['category'] != 'planned'
                    or hashlib.sha256(raw).hexdigest() != owned_records[identity]):
                raise ValueError('owned record differs or is no longer planned')
            owned_seen.add(identity)
        else:
            row = tuple(record['token_ids'])
            filtered.append(row)
            if 'target_token_id' in record:
                filtered.append(row + (record['target_token_id'],))
    if owned_seen != set(owned_records):
        raise ValueError('owned record is missing')
    full = set(filtered)
    prefixes = {row[:i] for row in filtered for i in range(1, len(row) + 1)}
    if any(row in prefixes or any(row[:i] in full for i in range(1, len(row) + 1))
           for row in candidates):
        raise ValueError('candidate has a known full-row prefix relationship')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate pool key')
        result[key] = value
    return result


def seal_record(pool_bytes, nonce, tokenizer, *, pool_sha256, protocol_sha256,
                exposure_sha256, old_commitment, pool_clear_sha256):
    """Derive and salt a record using an externally authorized fixed nonce.

    This pure helper does not generate a nonce, establish the one-use claim,
    verify currentness or authenticate the old-pool clearance receipt. Those
    are mandatory external controller responsibilities before calling it.
    """
    pins = (pool_sha256, protocol_sha256, exposure_sha256, old_commitment, pool_clear_sha256)
    if any(type(item) is not str or _SHA.fullmatch(item) is None for item in pins):
        raise ValueError('invalid successor identity')
    if (type(pool_bytes) is not bytes or not 1 <= len(pool_bytes) <= 65536
            or not hmac.compare_digest(hashlib.sha256(pool_bytes).hexdigest(), pool_sha256)):
        raise ValueError('public pool bytes differ')
    try:
        pool = _pool(json.loads(pool_bytes, object_pairs_hook=_unique_pairs))
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('invalid public pool bytes') from error
    allocation = allocate(pool, nonce)
    rows = iter(tokenize_candidate_pool(pool, tokenizer))
    lookup = {(context['id'], noun['id'], number): next(rows)
              for context in pool['contexts'] for noun in pool['nouns']
              for number in ('singular', 'plural')}
    prefix = tokenizer.encode('The', add_special_tokens=False)
    families, selected = [], set()
    for family in allocation:
        cases = []
        for index, text in enumerate(family['texts']):
            key = (family['context_id'], family['noun_ids'][index // 2],
                   'singular' if index % 2 == 0 else 'plural')
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (not _valid_tokens(ids) or tuple(ids) != lookup[key] or tuple(ids) in selected
                    or not 0 < len(prefix) < len(ids) - 1
                    or any(tokenizer.encode(text + suffix, add_special_tokens=False) != ids + [target]
                           for suffix, target in TARGETS.items())):
                raise ValueError('selected prompt changed after pool validation')
            selected.add(tuple(ids))
            cases.append({'text': text, 'input_ids': ids, 'attention_mask': [1] * len(ids),
                          'subject_index': len(prefix)})
        families.append({'family_index': family['family_index'], 'context_id': family['context_id'],
                         'noun_ids': family['noun_ids'], 'cases': cases})
    record = {'schema': 'mlg-natural-successor-heldout-v1', 'checkpoint_revision': CHECKPOINT_REVISION,
              'prediction_freeze_sha256': PREDICTION_FREEZE_SHA256, 'protocol_sha256': protocol_sha256,
              'pool_sha256': pool_sha256, 'exposure_inventory_sha256': exposure_sha256,
              'old_natural_commitment': old_commitment, 'pool_clearance_sha256': pool_clear_sha256,
              'families': families}
    raw = json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(',', ':')).encode('ascii')
    if len(raw) > 1024 * 1024:
        raise ValueError('successor record exceeds its bound')
    return hashlib.sha256(_COMMIT_DOMAIN + nonce + b'\0' + raw).hexdigest(), raw


def verify_record(pool_bytes, nonce, tokenizer, record_bytes, commitment, **pins):
    """Recompute the committed record from fixed identities and nonce."""
    if (type(record_bytes) is not bytes or not 1 <= len(record_bytes) <= 1024 * 1024
            or type(commitment) is not str or _SHA.fullmatch(commitment) is None):
        raise ValueError('invalid successor commitment')
    expected_commitment, expected_record = seal_record(pool_bytes, nonce, tokenizer, **pins)
    if (not hmac.compare_digest(record_bytes, expected_record)
            or not hmac.compare_digest(commitment, expected_commitment)):
        raise ValueError('successor commitment differs')
    return json.loads(expected_record)
