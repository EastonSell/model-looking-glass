"""Model-free checks of the planned invariant-noun exposure successor."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('INVARIANT_V39_ROOT', HERE.parent)).resolve()
sys.path.insert(0, str(ROOT))
source = HERE / 'pretrained_exposures_v40.py'
if not source.is_file():
    source = ROOT / 'looking_glass/pretrained_exposures_v40.py'
spec = importlib.util.spec_from_file_location('looking_glass.pretrained_exposures_v40', source)
v40 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v40
spec.loader.exec_module(v40)


def test_exact_planned_records_and_predecessor():
    old = v40.prior.inventory_payload()
    payload = v40.inventory_payload()
    raw = v40.canonical_inventory_bytes()
    assert hashlib.sha256(raw).hexdigest() == v40.INVENTORY_SHA256
    assert payload['records'][:len(old['records'])] == old['records']
    appended = payload['records'][len(old['records']):]
    assert len(appended) == 68 and len(v40._prefixes(payload['records'])) == 6947
    for row, (case_id, role, text, tokens) in zip(appended, v40.INVARIANT_NOUN_CASES):
        assert row['record_id'] == 'planned-' + case_id
        assert row['category'] == 'planned' and row['case_role'] == role
        assert row['public_text'] == text and row['token_ids'] == list(tokens)
        assert row['candidate_token_ids'] == list(v40.INVARIANT_NOUN_ANSWER_IDS)
    assert sum(x['case_role'] == 'target' for x in appended) == 48
    assert sum(x['case_role'] == 'fragment' for x in appended) == 12
    assert sum(x['case_role'] == 'demonstrative' for x in appended) == 8


def test_foreign_prefix_and_extension_are_rejected():
    predecessor = v40.prior.inventory_payload()
    case_id, _, _, row = v40.INVARIANT_NOUN_CASES[0]
    for candidate in (row, row[:-1], row + (v40.INVARIANT_NOUN_ANSWER_IDS[0], 99)):
        modified = copy.deepcopy(predecessor)
        modified['records'].append({'record_id': 'fabricated-collision', 'token_ids': list(candidate)})
        with pytest.raises(ValueError, match='known prefix relation'):
            v40._build_payload(modified)


def test_known_checking_apis_unchanged():
    old = ast.parse((ROOT / 'looking_glass/pretrained_exposures_v39.py').read_bytes())
    new = ast.parse(source.read_bytes())
    functions = lambda tree: {x.name: ast.dump(x, include_attributes=False)
                              for x in tree.body if isinstance(x, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    for name in ('check_known_splits', 'check_exposed_discovery', '_validate_known_check',
                 'require_preseal_readiness', 'prepare_preseal_disclosure',
                 '_has_known_relation', '_forbidden_rows'):
        assert a[name] == b[name]
