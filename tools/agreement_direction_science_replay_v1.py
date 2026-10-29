"""Replay a saved agreement-direction journal without loading a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from looking_glass import agreement_science_saved_v1 as saved


def write_report(path, report):
    """Create a new report only after finite JSON serialization succeeds."""
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', required=True)
    parser.add_argument('--direction-directory', required=True)
    parser.add_argument('--journal-directory', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--prelaunch-sha256', required=True)
    parser.add_argument('--reservation-sha256', required=True)
    parser.add_argument('--latest-exposure-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    binding = {'prelaunch_sha256': args.prelaunch_sha256,
               'reservation_sha256': args.reservation_sha256,
               'latest_exposure_sha256': args.latest_exposure_sha256}
    cases, directions = saved.load_frozen_inputs(args.case_file, args.direction_directory)
    report = saved.audit_attempt(args.journal_directory, cases, directions,
                                 expected_binding=binding, expected_source_revision=args.source_revision)
    report['reader_scope'] = ('Independent arithmetic and saved-byte consistency; supplied binding and '
        'revision are compared to recorded fields, not authenticated. A candidate requires a separately '
        'observed child exit, source/custody/exposure review and scientific interpretation.')
    report['supplied_binding'] = binding
    report['supplied_source_revision'] = args.source_revision
    digest = write_report(args.output, report)
    print(json.dumps({'status': 'saved_candidate_consistent', 'report_sha256': digest}, sort_keys=True))


if __name__ == '__main__':
    main()
