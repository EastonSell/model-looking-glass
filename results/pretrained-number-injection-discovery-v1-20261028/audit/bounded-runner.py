"""Call the frozen public scorer and bound every file this wrapper creates."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
PLAN_SHA = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
SCORER_SHA = '19ae9cb66a42e010ca66eacde2315420e04fb5eeff5c0b609e9360d65fea9e17'
RESULT_LIMIT = 256 * 1024
FAILURE_LIMIT = 4096


def write_new(path, data, limit):
    if type(data) is not bytes or len(data) > limit:
        raise ValueError('bounded audit output is too large')
    with path.open('xb') as handle:
        count = handle.write(data)
        if count != len(data):
            raise OSError('bounded audit output was not fully written')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-dir', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--prediction-sha256', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('a fresh absolute audit directory is required')
    scorer_path = Path(__file__).resolve().with_name('review_injection_discovery.py')
    source = scorer_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCORER_SHA:
        raise ValueError('published scorer bytes differ')
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    allowed_messages = {call.args[1].value for call in ast.walk(ast.parse(source))
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == 'require' and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)}
    output.mkdir(mode=0o700)
    stage = 'load'
    try:
        namespace = {'__name__': 'frozen_independent_scorer', '__file__': str(scorer_path)}
        exec(compile(source, str(scorer_path), 'exec', dont_inherit=True), namespace)
        stage = 'audit'
        result = namespace['audit'](args.public_dir, args.plan, SOURCE,
                                    args.prediction_sha256, args.archive_sha256, PLAN_SHA)
        if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != SCORER_SHA:
            raise ValueError('scorer source changed during the audit')
        result['audit_source_sha256'] = SCORER_SHA
        result['bounded_runner_sha256'] = runner_sha
        payload = namespace['canonical'](result) + b'\n'
        if len(payload) > RESULT_LIMIT:
            raise ValueError('bounded audit result exceeds its allowance')
        stage = 'write'
        write_new(output / 'result.json', payload, RESULT_LIMIT)
        print(json.dumps({'status': 'passed', 'records': result['records'],
                          'result_bytes': len(payload), 'result_sha256': hashlib.sha256(payload).hexdigest()}))
        return 0
    except BaseException as error:
        # A partial result is retained. The bounded failure receipt carries no
        # exception text, file paths, private witness, or raw model tensor.
        reason = str(error) if isinstance(error, ValueError) and str(error) in allowed_messages else 'unclassified'
        failure = json.dumps({'status': 'incomplete', 'stage': stage, 'reason': reason,
                              'error_type': type(error).__name__}, sort_keys=True,
                             separators=(',', ':')).encode() + b'\n'
        try:
            write_new(output / 'failure.json', failure, FAILURE_LIMIT)
        except BaseException:
            pass
        print('Public audit incomplete; retain its bounded evidence.', file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            return 130
        return 1


if __name__ == '__main__':
    sys.exit(main())
