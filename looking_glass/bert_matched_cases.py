"""Public, tokenizer-checkable cases selected after the matched-study freeze.

Case generation does not import or run a model. The selection is a convenience
sample, not a blinded or population sample, and the cited history is incomplete.
"""
import hashlib
import json
from pathlib import Path

from . import bert_agreement_cases as prior
from . import bert_agreement_exposures_v5 as exposure

SCHEMA = 'distilbert-matched-magnitude-public-cases-v1'
MANIFEST_SHA256 = '5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb'
PROTOCOL_COMMIT = '168939c651221e52bcc3a07bbe37c267ced49022'
PROTOCOL_SHA256 = '91683a8f58ec182428f54c29e00d2bace692de3288014f3bf5f16bcbf665d35c'
PRIOR_EXPOSURE_SHA256 = '34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c'

# A fixed convenience shortlist, selected without reading new model outcomes.
# Keep all candidates, including those excluded by prior use or vocabulary.
CANDIDATES = tuple(sorted(('architect', 'baker', 'barber', 'butcher', 'captain', 'cashier',
    'chemist', 'designer', 'engineer', 'florist', 'gardener', 'janitor', 'journalist', 'judge',
    'librarian', 'mechanic', 'musician', 'officer', 'pharmacist', 'photographer', 'plumber',
    'professor', 'scientist', 'secretary', 'surgeon', 'tailor', 'waiter', 'carpenter',
    'electrician', 'detective', 'dentist', 'physician', 'instructor', 'minister', 'politician',
    'police', 'farmer', 'nurse')))

# Each tuple is singular ID, plural ID in the pinned BERT WordPiece vocabulary.
LEXEMES = {
    'architect': (4944, 8160), 'captain': (2952, 15755), 'designer': (5859, 11216),
    'detective': (6317, 18145), 'engineer': (3992, 6145), 'instructor': (9450, 19922),
    'journalist': (4988, 8845), 'judge': (3648, 6794), 'mechanic': (15893, 9760),
    'minister': (2704, 7767), 'musician': (5455, 5389), 'officer': (2961, 3738),
    'photographer': (8088, 17008), 'physician': (7522, 11572),
    'politician': (3761, 8801), 'professor': (2934, 12655),
}
SELECTED = tuple(sorted(LEXEMES))
FAMILIES = tuple(zip(SELECTED[:8], SELECTED[8:]))
NUMBERS = ('singular', 'plural')


def _row_id(family, source, subject_number, distractor_number):
    return f'evaluation-matched-f{family:02}-{source}-{subject_number[0]}-{distractor_number[0]}'


def _row(family, source, subject, number, distractor, distractor_number):
    subject_id = LEXEMES[subject][NUMBERS.index(number)]
    distractor_id = LEXEMES[distractor][NUMBERS.index(distractor_number)]
    subject_word = subject + ('s' if number == 'plural' else '')
    distractor_word = distractor + ('s' if distractor_number == 'plural' else '')
    return {'row_id': _row_id(family, source, number, distractor_number), 'split': 'evaluation',
            'family': family, 'subject_source': source, 'subject_lemma': subject,
            'subject_number': number, 'distractor_lemma': distractor,
            'distractor_number': distractor_number,
            'text': f'The {subject_word} beside the {distractor_word} [MASK] ready.',
            'input_ids': [101, 1996, subject_id, 3875, 1996, distractor_id, 103, 3201, 1012, 102],
            'attention_mask': [1] * 10,
            'exposure_status': 'public-evaluation-not-observed-in-cited-evidence'}


def build_manifest():
    if exposure.snapshot_digest() != PRIOR_EXPOSURE_SHA256:
        raise ValueError('the cited prior exposure snapshot changed')
    prior_lemmas = set(prior.LEXEMES)
    if (len(CANDIDATES) != len(set(CANDIDATES)) or len(SELECTED) != 16
            or set(SELECTED) & prior_lemmas):
        raise ValueError('new selected lemmas overlap or are duplicated')
    rows, targets = [], []
    for index, (subject, distractor) in enumerate(FAMILIES):
        family, next_subject = index + 1, FAMILIES[(index + 1) % 8][0]
        for source, source_subject in (('A', subject), ('B', next_subject)):
            for number in NUMBERS:
                for distractor_number in NUMBERS:
                    rows.append(_row(family, source, source_subject, number, distractor, distractor_number))
        for number in NUMBERS:
            opposite = NUMBERS[1 - NUMBERS.index(number)]
            for distractor_number in NUMBERS:
                targets.append({'case_id': f'evaluation-matched-f{family:02}-{number[0]}-{distractor_number[0]}',
                    'split': 'evaluation', 'family': family,
                    'target_row_id': _row_id(family, 'A', number, distractor_number),
                    'donor_row_ids': {'A_same': _row_id(family, 'A', number, distractor_number),
                        'A_opposite': _row_id(family, 'A', opposite, distractor_number),
                        'B_same': _row_id(family, 'B', number, distractor_number),
                        'B_opposite': _row_id(family, 'B', opposite, distractor_number)}})
    return {'schema': SCHEMA, 'scope': 'public nonblind cases; uncited history and pretraining unknown',
        'protocol': {'commit': PROTOCOL_COMMIT, 'sha256': PROTOCOL_SHA256},
        'prior_exposure_sha256': PRIOR_EXPOSURE_SHA256,
        'identity': {'model': prior.MODEL_ID, 'revision': prior.MODEL_REVISION,
                     'asset_sha256': dict(prior.ASSET_SHA256)},
        'coordinates': {'subject': 2, 'distractor': 5, 'mask': 6, 'sequence_length': 10,
                        'output_token_ids': dict(prior.OUTPUT_TOKENS)},
        'selection': {'shortlist': list(CANDIDATES), 'selected': list(SELECTED),
            'rule': 'exclude cited prior lemmas; retain vocabulary entries with ordinary s-plurals; '
                    'sort lexically and take the first 16; pair first eight with last eight in order',
            'new_case_model_outcomes_used': False}, 'rows': rows, 'targets': targets}


def canonical_manifest_bytes():
    return (json.dumps(build_manifest(), sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + '\n').encode('utf-8')


def manifest_digest():
    digest = hashlib.sha256(canonical_manifest_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError('matched case manifest differs from its pinned digest')
    return digest


def verify_manifest_bytes(raw):
    if type(raw) is not bytes or len(raw) > 1_000_000 or raw != canonical_manifest_bytes():
        raise ValueError('matched case manifest differs from its canonical record')
    digest = manifest_digest()
    current = build_manifest()
    old = prior.build_manifest()
    rows = current['rows']
    old_tokens = {tuple(row['input_ids']) for row in old['rows']}
    new_tokens = {tuple(row['input_ids']) for row in rows}
    if (len(rows) != 64 or len(current['targets']) != 32 or len(new_tokens) != 64
            or len({row['row_id'] for row in rows}) != 64 or new_tokens & old_tokens):
        raise ValueError('new input rows overlap or are incomplete')
    return {'manifest_sha256': digest, 'evaluation_rows': 64, 'targets': 32,
            'exact_prior_row_overlap': 0, 'cited_history_complete': False,
            'execution_authorized': False}


def verify_vocab_bytes(raw):
    """Reproduce the selection and token IDs from the pinned vocabulary bytes."""
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != prior.ASSET_SHA256['vocab.txt']:
        raise ValueError('vocabulary bytes differ from the pinned checkpoint')
    words = raw.decode('utf-8').splitlines()
    if len(words) != 30522 or len(words) != len(set(words)):
        raise ValueError('invalid WordPiece vocabulary')
    ids = {word: index for index, word in enumerate(words)}
    eligible = tuple(word for word in CANDIDATES if word not in prior.LEXEMES
                     and word in ids and word + 's' in ids)
    if len(eligible) != 18 or eligible[:16] != SELECTED:
        raise ValueError('the recorded lexical selection does not reproduce')
    if any((ids[word], ids[word + 's']) != LEXEMES[word] for word in SELECTED):
        raise ValueError('the recorded WordPiece IDs do not reproduce')
    return {'eligible': len(eligible), 'selected': len(SELECTED)}


def verify_offline_tokenizer(snapshot):
    """Check pinned tokenizer bytes, isolated lexemes and all full prompt rows."""
    from tokenizers import Tokenizer

    snapshot = Path(snapshot)
    vocab_raw, tokenizer_raw = (snapshot / 'vocab.txt').read_bytes(), (snapshot / 'tokenizer.json').read_bytes()
    verify_vocab_bytes(vocab_raw)
    if hashlib.sha256(tokenizer_raw).hexdigest() != prior.ASSET_SHA256['tokenizer.json']:
        raise ValueError('tokenizer bytes differ from the pinned checkpoint')
    tokenizer = Tokenizer.from_str(tokenizer_raw.decode('utf-8'))
    for word in SELECTED:
        for form, expected in ((word, LEXEMES[word][0]), (word + 's', LEXEMES[word][1])):
            if tokenizer.encode(form, add_special_tokens=False).ids != [expected]:
                raise ValueError('an isolated lexeme is not the expected WordPiece')
    for row in build_manifest()['rows']:
        actual = tokenizer.encode(row['text'], add_special_tokens=True)
        if actual.ids != row['input_ids'] or actual.attention_mask != row['attention_mask']:
            raise ValueError('a full prompt does not reproduce its frozen IDs and mask')
    return {'isolated_lexemes': 32, 'prompt_rows': 64, 'model_forwarded': False}
