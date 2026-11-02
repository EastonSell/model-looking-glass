"""No-model startup component for the prospective successor.

Execution/claim and held-out adapters are deliberately not implemented here.
This component proves only that the bound frozen startup passes in the chosen
isolated process; the eventual owner must be re-bound and rehearsed end to end.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import sys


def check_frozen_startup(path, expected_sha256):
    source = Path(path)
    if (not source.is_absolute() or source.name != 'natural_agreement_cpu.py'
            or type(expected_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None):
        raise ValueError('invalid frozen startup identity')
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('frozen startup source differs')
    tree = ast.parse(raw, filename=str(source))
    wanted = ('_dependency_configuration', '_startup')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if (len(functions) != len(wanted) or {node.name for node in functions} != set(wanted)
            or any(node.decorator_list for node in functions)):
        raise ValueError('frozen startup functions differ')
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {'hashlib': hashlib, 'os': os, 'Path': Path, 'sys': sys, 'ROOT': source.parents[1]}
    exec(compile(module, str(source), 'exec', dont_inherit=True), namespace)
    namespace['_startup']()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', required=True, action='store_true')
    parser.add_argument('--frozen-launcher', required=True)
    parser.add_argument('--frozen-launcher-sha', required=True)
    args = parser.parse_args()
    check_frozen_startup(args.frozen_launcher, args.frozen_launcher_sha)
    print('startup-passed; no-model')


if __name__ == '__main__':
    main()
