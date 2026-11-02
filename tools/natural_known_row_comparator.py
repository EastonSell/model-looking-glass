"""Compare complete token rows without returning either input collection."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat


def compare_rows(private, public):
    for rows in (private, public):
        if type(rows) is not list or not rows:
            raise ValueError('rows are incomplete')
        for row in rows:
            if (type(row) is not list or not row
                    or any(type(token) is not int or token < 0 for token in row)):
                raise ValueError('invalid token row')
    for left in private:
        for right in public:
            shared = min(len(left), len(right))
            if left[:shared] == right[:shared]:
                return 'collision'
    return 'clear'


def compare_evidence(*, nonce, record_raw, commitment_raw, proposal_raw,
                     supervisor_raw, observation_raw, pins):
    """Verify a sealed record and return a digest-only comparison result."""
    digest = lambda data: hashlib.sha256(data).hexdigest()
    try:
        if (type(nonce) is not bytes or len(nonce) != 32
                or any(type(raw) is not bytes or not raw or len(raw) > 1_000_000
                       for raw in (record_raw, commitment_raw, proposal_raw, supervisor_raw, observation_raw))):
            raise ValueError('evidence bounds differ')
        for raw, name in ((proposal_raw, 'proposal'), (commitment_raw, 'commitment'),
                          (supervisor_raw, 'supervisor'), (observation_raw, 'observation')):
            if not hmac.compare_digest(digest(raw), pins[name]):
                raise ValueError('evidence pin differs')
        record = json.loads(record_raw)
        if json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8') != record_raw:
            raise ValueError('record is not canonical')
        commitment, proposal = json.loads(commitment_raw), json.loads(proposal_raw)
        supervisor, observation = json.loads(supervisor_raw), json.loads(observation_raw)
        audit = supervisor['audit']['preparation']
        if (observation['status'] != 'completed' or type(observation['exit_code']) is not int
                or observation['exit_code'] != 0 or observation['supervisor_record_sha256'] != pins['supervisor']
                or audit['nonce.bin'] != digest(nonce) or audit['heldout-record.json'] != digest(record_raw)
                or audit['commitment.json'] != pins['commitment']):
            raise ValueError('supervised evidence differs')
        if (record['schema'] != 'mlg-number-heldout-v1'
                or commitment['schema'] != 'mlg-natural-heldout-public-commitment-v1'
                or record['prediction_freeze_sha256'] != pins['freeze']
                or commitment['prediction_freeze_sha256'] != pins['freeze']
                or record['exposure_inventory_sha256'] != pins['inventory']
                or commitment['exposure_inventory_sha256'] != pins['inventory']
                or record['source_commits'] != {'natural': pins['source_commit']}):
            raise ValueError('scientific identities differ')
        sealed = digest(b'MLG-number-v1-heldout-commit\0' + nonce + b'\0' + record_raw)
        if not hmac.compare_digest(sealed, commitment['heldout_commitment']):
            raise ValueError('record commitment differs')
        private = [case['input_ids'] for family in record['families'] for case in family['cases']]
        public = [case['input_ids'] for case in proposal['cases']]
        if (len(private) != pins['private_count'] or len(public) != pins['public_count']
                or len({tuple(row) for row in public}) != len(public)):
            raise ValueError('case counts differ')
        return {'status': compare_rows(private, public),
                'natural_heldout_commitment': commitment['heldout_commitment'],
                'natural_prediction_freeze_sha256': pins['freeze'],
                'public_input_file_sha256': pins['proposal'],
                'natural_case_count': len(private), 'public_case_count': len(public)}
    except (KeyError, TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise ValueError('comparison evidence is invalid') from None


def _read(path, limit):
    path = Path(path)
    info = path.lstat()
    if (not path.is_absolute() or not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or info.st_nlink != 1 or getattr(info, 'st_file_attributes', 0) & 1024
            or info.st_size > limit):
        raise ValueError('unsafe input')
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise ValueError('input changed')
    return data


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('incomplete receipt write')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    import argparse
    from datetime import datetime, timezone
    import sys
    parser = argparse.ArgumentParser()
    for name in ('phase', 'supervisor', 'proposal', 'pins', 'pins-sha256', 'receipt'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        raw_pins = _read(args.pins, 8192)
        if hashlib.sha256(raw_pins).hexdigest() != args.pins_sha256:
            raise ValueError('pins differ')
        pins = json.loads(raw_pins)
        expected = {'freeze', 'inventory', 'source_commit', 'proposal', 'commitment',
                    'supervisor', 'observation', 'private_count', 'public_count', 'comparator_source'}
        if type(pins) is not dict or set(pins) != expected:
            raise ValueError('pins are incomplete')
        if hashlib.sha256(_read(Path(__file__).resolve(), 1_000_000)).hexdigest() != pins['comparator_source']:
            raise ValueError('comparator source differs')
        phase, supervisor = Path(args.phase), Path(args.supervisor)
        if not phase.is_absolute() or not supervisor.is_absolute() or not Path(args.receipt).is_absolute():
            raise ValueError('absolute paths are required')
        result = compare_evidence(nonce=_read(phase / 'nonce.bin', 32),
            record_raw=_read(phase / 'heldout-record.json', 1_000_000),
            commitment_raw=_read(phase / 'commitment.json', 8192),
            proposal_raw=_read(args.proposal, 1_000_000),
            supervisor_raw=_read(supervisor / 'supervisor.json', 1_000_000),
            observation_raw=_read(supervisor / 'supervisor-observation.json', 8192), pins=pins)
        result.update({'schema': 'mlg-natural-external-input-comparison-v1',
            'comparison': 'bidirectional-complete-row-prefix',
            'comparator_source_sha256': pins['comparator_source'],
            'observed_utc': datetime.now(timezone.utc).isoformat()})
        output = json.dumps(result, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('ascii')
        _write_new(args.receipt, output)
        print(result['status'])
        return 0
    except BaseException:
        print('comparison incomplete', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
