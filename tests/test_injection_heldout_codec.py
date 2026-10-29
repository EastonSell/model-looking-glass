"""Fabricated inputs only; no committed site, split or pretrained model."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import pytest

from looking_glass import pretrained_number_archive as discovery_archive
from tests.test_pretrained_number_selection import fabricated_cases

try:
    from looking_glass.injection_heldout import bind_heldout_codec
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.injection_heldout':
        raise
    bind_heldout_codec = None

try:
    from looking_glass.injection_heldout import (HeldoutJournal, make_heldout_binding,
                                                 heldout_request_id, inspect_heldout_journal,
                                                 heldout_request_plan)
except ImportError:
    HeldoutJournal = None
    inspect_heldout_journal = None

try:
    from looking_glass.injection_heldout import committed_targets
except ImportError:
    committed_targets = None

try:
    from looking_glass.injection_heldout import verify_frozen_inputs
except ImportError:
    verify_frozen_inputs = None


def fake_binding():
    return make_heldout_binding(source_commit='a' * 40, runtime_sha256='b' * 64,
                                prediction_sha256='c' * 64, freeze_sha256='d' * 64,
                                heldout_commitment='e' * 64, site_commitment='f' * 64,
                                inventory_sha256='1' * 64, operation_sha256='2' * 64,
                                gate_sha256='4' * 64, disjoint_status='clear',
                                disjoint_receipt_sha256='3' * 64)


def test_heldout_codec_uses_committed_case_without_changing_discovery_codec():
    assert bind_heldout_codec is not None, 'held-out codec is missing'
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    observed = fabricated_cases(ids_rows=rows)[0].clean
    original_ids = discovery_archive._case_ids(0)
    codec = bind_heldout_codec(rows)
    public = codec.serialize_public(0, 'clean', observed)
    private = codec.serialize_private(0, 'clean', observed, ())
    assert codec.audit_pair(public, private, site=2)['verified'] is True
    assert discovery_archive._case_ids(0) == original_ids
    with pytest.raises(ValueError):
        discovery_archive.serialize_discovery_observation(0, 'clean', observed, execution_device='cpu')


def test_distinct_codecs_do_not_share_rows_or_accept_swapped_records():
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    shifted = rows[1:] + rows[:1]
    first, second = bind_heldout_codec(rows), bind_heldout_codec(shifted)
    cases = fabricated_cases(ids_rows=rows)
    raw = first.serialize_public(0, 'clean', cases[0].clean)
    with pytest.raises(ValueError):
        second.read_public(raw)
    with pytest.raises(ValueError):
        first.serialize_public(0, 'clean', cases[1].clean)
    assert first.read_public(raw)[0:2] == (0, 'clean')


def test_adapter_matches_frozen_discovery_serializers_on_the_same_rows():
    from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
    from looking_glass.pretrained_number_witness import encode_private_forward
    rows = tuple(tuple(PUBLIC_DISCOVERY[(i // 2) * 4 + i % 2][1]) for i in range(8))
    case = fabricated_cases()[0]
    codec = bind_heldout_codec(rows)
    assert codec.serialize_public(0, 'clean', case.clean) == discovery_archive.serialize_discovery_observation(
        0, 'clean', case.clean, execution_device='cpu')
    assert codec.serialize_private(0, 'clean', case.clean, ()) == encode_private_forward(0, 'clean', case.clean, ())


def test_actual_private_witness_rejects_a_wrong_donor():
    import torch
    from looking_glass.pretrained_number_adapter import to_selector_observation
    from looking_glass.pretrained_number_injection import NumberInjectionExecutor
    from tests.test_pretrained_number_injection import ToyModel
    rows = tuple((10, 200 + i, 300) for i in range(8))
    codec = bind_heldout_codec(rows)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            clean = executor.forward(rows[0], request_id='1' * 32, condition='clean')
            injected = executor.forward(rows[0], request_id='2' * 32, condition='injected')
            result = executor.forward(rows[0], request_id='3' * 32, condition='clean_to_injected',
                                      layer=1, donor=clean.mlp_donor(1))
            observation = to_selector_observation(result)
            evidence = tuple(item for item in executor.private_evidence() if item.request_id == result.request_id)
            public = codec.serialize_public(0, 'forward_1', observation)
            private = codec.serialize_private(0, 'forward_1', observation, evidence)
            assert codec.audit_pair(public, private, site=2, clean=to_selector_observation(clean),
                                    injected=to_selector_observation(injected))['verified'] is True
            wrong = tuple(replace(item, donor=bytes(len(item.donor))) if item.operation == 'patch' else item
                          for item in evidence)
            altered = codec.serialize_private(0, 'forward_1', observation, wrong)
            with pytest.raises(ValueError):
                codec.audit_pair(public, altered, site=2, clean=to_selector_observation(clean),
                                 injected=to_selector_observation(injected))
    finally:
        torch.set_num_threads(old_threads)


def test_heldout_journal_preserves_wrong_order_as_a_failed_attempt(tmp_path):
    assert HeldoutJournal is not None, 'held-out journal is missing'
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec = bind_heldout_codec(rows)
    cases = fabricated_cases(ids_rows=rows)
    binding = fake_binding()
    journal = HeldoutJournal(tmp_path / 'private-journal', codec=codec, binding=binding,
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    clean = replace(cases[0].clean, request_id=heldout_request_id(binding, 0, 'clean'))
    public = codec.serialize_public(0, 'clean', clean)
    private = codec.serialize_private(0, 'clean', clean, ())
    journal.append(public, private)
    other = replace(cases[1].clean, request_id=heldout_request_id(binding, 1, 'clean'))
    wrong = codec.serialize_public(1, 'clean', other)
    wrong_private = codec.serialize_private(1, 'clean', other, ())
    with pytest.raises(ValueError):
        journal.append(wrong, wrong_private)
    assert (tmp_path / 'private-journal/000.frame').is_file()
    assert (tmp_path / 'private-journal/000.ok').is_file()
    assert wrong in (tmp_path / 'private-journal/001.frame').read_bytes()
    assert wrong_private in (tmp_path / 'private-journal/001.frame').read_bytes()
    assert not (tmp_path / 'private-journal/001.ok').exists()
    assert (tmp_path / 'private-journal/terminal.json').is_file()
    with pytest.raises(RuntimeError):
        journal.append(public, private)


def test_bounded_malformed_pair_is_retained_without_acknowledgment(tmp_path):
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    journal = HeldoutJournal(tmp_path / 'journal', codec=bind_heldout_codec(rows), binding=fake_binding(),
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    with pytest.raises(ValueError):
        journal.append(b'malformed-public', b'bounded-private')
    frame = (tmp_path / 'journal/000.frame').read_bytes()
    assert b'malformed-public' in frame and b'bounded-private' in frame
    assert not (tmp_path / 'journal/000.ok').exists()
    assert (tmp_path / 'journal/terminal.json').is_file()


def test_journal_rejects_a_request_not_bound_to_the_split_plan(tmp_path):
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec = bind_heldout_codec(rows)
    clean = replace(fabricated_cases(ids_rows=rows)[0].clean, request_id='f' * 32)
    journal = HeldoutJournal(tmp_path / 'journal', codec=codec, binding=fake_binding(),
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    public = codec.serialize_public(0, 'clean', clean)
    private = codec.serialize_private(0, 'clean', clean, ())
    with pytest.raises(ValueError):
        journal.append(public, private)
    assert public in (tmp_path / 'journal/000.frame').read_bytes()
    assert not (tmp_path / 'journal/000.ok').exists()


def test_pending_peer_is_not_a_clear_receipt_and_plan_is_unique():
    binding = make_heldout_binding(source_commit='a' * 40, runtime_sha256='b' * 64,
                                   prediction_sha256='c' * 64, freeze_sha256='d' * 64,
                                   heldout_commitment='e' * 64, site_commitment='f' * 64,
                                   inventory_sha256='1' * 64, operation_sha256='2' * 64,
                                   gate_sha256='3' * 64, disjoint_status='pending',
                                   disjoint_receipt_sha256=None)
    record = json.loads(binding.record_bytes)
    assert record['disjoint_status'] == 'pending' and record['disjoint_receipt_sha256'] is None
    plan = heldout_request_plan(binding)
    assert len(plan) == 128 and len({item[2] for item in plan}) == 128
    assert plan[0][:2] == (0, 'clean') and plan[-1][:2] == (7, 'verify_clean')


def test_frozen_input_verifier_rejects_changed_prediction_before_model_work():
    assert verify_frozen_inputs is not None, 'frozen-input verifier is missing'
    with pytest.raises(ValueError):
        verify_frozen_inputs(prediction_bytes=b'changed', runtime_bytes=b'changed', freeze_bytes=b'changed',
                             plan_bytes=b'changed', inventory_bytes=b'changed', inventory_sha256='a' * 64,
                             split_nonce=b's' * 32, heldout_record_bytes=b'changed', heldout_commitment='b' * 64,
                             site_nonce=b't' * 32, site_record_bytes=b'changed', site_commitment='c' * 64,
                             tokenizer=None)


def fake_frozen_inputs(monkeypatch, *, known_prefix=False):
    """Create a sealed allocation from entirely invented data and identities."""
    from looking_glass import injection_heldout as heldout
    from looking_glass import pretrained_number_commitments as commitments
    from looking_glass import pretrained_number_requests as requests
    from looking_glass import pretrained_exposures_v3 as exposures
    from looking_glass.pretrained_number_selection import freeze_discovery
    from tests.test_pretrained_number_commitments import NOUNS, PLURAL, SINGULAR
    monkeypatch.setattr(commitments, 'NOUNS', NOUNS)
    monkeypatch.setattr(commitments, 'PLURAL_CONTEXTS', PLURAL)
    monkeypatch.setattr(commitments, 'SINGULAR_CONTEXTS', SINGULAR)
    source = 'd' * 40
    runtime = b'fabricated runtime'
    prediction = freeze_discovery(fabricated_cases(), private_controls_verified=True, execution_device='cpu')
    runtime_hash, prediction_hash = (hashlib.sha256(value).hexdigest() for value in (runtime, prediction))
    freeze = commitments.make_prediction_freeze({'injection': {'source_commit': source,
                                                                 'runtime_sha256': runtime_hash,
                                                                 'prediction_sha256': prediction_hash}})
    plan = requests.create_family_request_plan(b'p' * 32, source)
    site_nonce, split_nonce = b's' * 32, b't' * 32
    site = commitments.seal_site(site_nonce, source)
    inventory_payload = exposures.inventory_payload()
    inventory = exposures.canonical_inventory_bytes()
    snapshot = commitments.load_inventory_snapshot(inventory, hashlib.sha256(inventory).hexdigest())
    options = dict(site_nonce=site_nonce, site_record_bytes=site.record_bytes, site_commitment=site.commitment_sha256)
    allocation = commitments.sample_families(freeze, split_nonce, inventory_snapshot=snapshot,
                                              current_inventory_sha256=snapshot.sha256, **options)
    if known_prefix:
        extra = dict(inventory_payload['records'][0])
        extra.update(record_id='fabricated-known-short-row', category='planned', token_ids=[10, 20000])
        inventory_payload['records'].append(extra)
        inventory = json.dumps(inventory_payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        snapshot = commitments.load_inventory_snapshot(inventory, hashlib.sha256(inventory).hexdigest())
        allocation = commitments.sample_families(freeze, split_nonce, inventory_snapshot=snapshot,
                                                  current_inventory_sha256=snapshot.sha256, **options)
    encoded = {'The': [10]}
    tokenized = []
    for family in allocation.families:
        encoded[' ' + family.context] = [3000 + family.family_index]
        members = []
        for index, (subject, text) in enumerate(zip(family.subjects, family.texts)):
            prefix = [10, 20000 + 4 * family.family_index + index]
            ids = prefix + [3000 + family.family_index]
            encoded['The ' + subject] = prefix
            encoded[text] = ids
            for suffix, token in ((' is', 318), (' are', 389), (' was', 373), (' were', 547)):
                encoded[text + suffix] = ids + [token]
            members.append({'text': text, 'input_ids': ids, 'attention_mask': [1] * len(ids), 'subject_index': 1})
        tokenized.append(members)
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(encoded[text])
    sealed = commitments.seal_heldout(freeze, split_nonce, tokenized, allocation=allocation,
                                       inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256, **options)
    for name, value in (('INJECTION_SOURCE_COMMIT', source), ('INJECTION_RUNTIME_SHA256', runtime_hash),
                        ('INJECTION_PREDICTION_SHA256', prediction_hash), ('INJECTION_FREEZE_SHA256', freeze.sha256),
                        ('INJECTION_PLAN_SHA256', hashlib.sha256(plan).hexdigest()),
                        ('INJECTION_SITE_COMMITMENT', site.commitment_sha256)):
        monkeypatch.setattr(heldout, name, value)
    return dict(prediction_bytes=prediction, runtime_bytes=runtime, freeze_bytes=freeze.record_bytes,
                plan_bytes=plan, inventory_bytes=inventory, inventory_sha256=snapshot.sha256,
                split_nonce=split_nonce, heldout_record_bytes=sealed.record_bytes,
                heldout_commitment=sealed.commitment_sha256, tokenizer=FakeTokenizer(), **options)


def test_frozen_input_verifier_rechecks_the_sealed_record_and_all_tokenizations(monkeypatch):
    inputs = fake_frozen_inputs(monkeypatch)
    checked = verify_frozen_inputs(**inputs)
    assert len(checked.targets) == 8
    assert all(len(item.input_ids) == 3 for item in checked.targets)
    with pytest.raises(FrozenInstanceError):
        checked.targets[0].input_ids = (1, 2, 3)
    assert '20000' not in repr(checked)
    for key in ('prediction_bytes', 'runtime_bytes', 'freeze_bytes', 'plan_bytes', 'heldout_record_bytes'):
        altered = dict(inputs, **{key: inputs[key] + b' '})
        with pytest.raises(ValueError):
            verify_frozen_inputs(**altered)
    with pytest.raises(ValueError):
        verify_frozen_inputs(**dict(inputs, split_nonce=inputs['site_nonce']))
    class WrongTokenizer:
        def encode(self, text, *, add_special_tokens):
            ids = inputs['tokenizer'].encode(text, add_special_tokens=add_special_tokens)
            return ids[:-1] + [1] if text.endswith(' were') else ids
    with pytest.raises(ValueError):
        verify_frozen_inputs(**dict(inputs, tokenizer=WrongTokenizer()))


def test_frozen_input_verifier_rejects_a_known_row_that_is_a_candidate_prefix(monkeypatch):
    inputs = fake_frozen_inputs(monkeypatch, known_prefix=True)
    with pytest.raises(ValueError):
        verify_frozen_inputs(**inputs)


def test_prepared_binding_rejects_swapped_split_or_frozen_identity(monkeypatch):
    from looking_glass import injection_heldout as module
    inputs = fake_frozen_inputs(monkeypatch)
    prepared = verify_frozen_inputs(**inputs)
    assert hasattr(module, 'make_prepared_binding'), 'prepared binding is missing'
    assert hasattr(module, 'check_prepared_binding'), 'prepared binding verifier is missing'
    binding = module.make_prepared_binding(prepared, operation_sha256='2' * 64, gate_sha256='3' * 64,
                                           disjoint_status='pending', disjoint_receipt_sha256=None)
    assert module.check_prepared_binding(binding, prepared) is binding
    with pytest.raises(ValueError):
        module.check_prepared_binding(binding, replace(prepared, heldout_commitment='f' * 64))
    with pytest.raises(ValueError):
        module.check_prepared_binding(binding, replace(prepared, inventory_sha256='e' * 64))
    with pytest.raises(ValueError):
        module.check_prepared_binding(binding, replace(prepared, prediction_bytes=b'other prediction'))
    changed = replace(prepared.targets[0], input_ids=(1, 2, 3))
    with pytest.raises(ValueError):
        module.check_prepared_binding(binding, replace(prepared, targets=(changed,) + prepared.targets[1:]))
    with pytest.raises(ValueError):
        module.check_prepared_binding(binding, replace(prepared, site=1 if prepared.site != 1 else 2))
    for field in ('protocol_sha256', 'checkpoint_revision'):
        altered = json.loads(prepared.heldout_record_bytes)
        altered[field] = '0' * len(altered[field])
        raw = json.dumps(altered, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(module._commitments._HELDOUT_COMMIT_DOMAIN + prepared.split_nonce + b'\0' + raw).hexdigest()
        with pytest.raises(ValueError):
            module.make_prepared_binding(replace(prepared, heldout_record_bytes=raw, heldout_commitment=digest),
                                         operation_sha256='2' * 64, gate_sha256='3' * 64,
                                         disjoint_status='pending', disjoint_receipt_sha256=None)


def test_saved_journal_cannot_score_an_incomplete_prefix(tmp_path):
    assert inspect_heldout_journal is not None, 'saved journal auditor is missing'
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec, binding = bind_heldout_codec(rows), fake_binding()
    journal = HeldoutJournal(tmp_path / 'journal', codec=codec, binding=binding,
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    clean = replace(fabricated_cases(ids_rows=rows)[0].clean, request_id=heldout_request_id(binding, 0, 'clean'))
    journal.append(codec.serialize_public(0, 'clean', clean), codec.serialize_private(0, 'clean', clean, ()))
    with pytest.raises(ValueError):
        inspect_heldout_journal(tmp_path / 'journal', codec=codec, binding=binding, site=2,
                                max_disk_bytes=8 * 1024 * 1024, acknowledgments=(), externally_quiescent=True)


def test_logical_capacity_is_reserved_before_a_frame_is_staged(tmp_path):
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    journal = HeldoutJournal(tmp_path / 'journal', codec=bind_heldout_codec(rows), binding=fake_binding(),
                             site=2, max_disk_bytes=8 * 1024 * 1024, max_tensor_bytes=15)
    with pytest.raises(ValueError):
        journal.append(b'malformed-public', b'bounded-private')
    assert not (tmp_path / 'journal/000.frame').exists()
    assert not (tmp_path / 'journal/000.ok').exists()
    assert (tmp_path / 'journal/terminal.json').is_file()


def test_malformed_structured_pair_uses_full_size_for_admission(tmp_path):
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec, binding = bind_heldout_codec(rows), fake_binding()
    clean = replace(fabricated_cases(ids_rows=rows)[0].clean,
                    request_id=heldout_request_id(binding, 0, 'clean'))
    public = json.loads(codec.serialize_public(0, 'clean', clean))
    public['trace'] = 'x' * 8192
    public['payload'] = {'attention': [''] * 6, 'mlp': [''] * 6, 'block': '', 'logits': ''}
    raw = json.dumps(public, sort_keys=True, separators=(',', ':')).encode()
    private = codec.serialize_private(0, 'clean', clean, ())
    journal = HeldoutJournal(tmp_path / 'journal', codec=codec, binding=binding,
                             site=2, max_disk_bytes=8 * 1024 * 1024, max_tensor_bytes=1024)
    with pytest.raises(ValueError):
        journal.append(raw, private)
    assert not (tmp_path / 'journal/000.frame').exists()


def test_decoder_out_of_memory_is_terminal_and_pair_is_retained(tmp_path, monkeypatch):
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec, binding = bind_heldout_codec(rows), fake_binding()
    clean = replace(fabricated_cases(ids_rows=rows)[0].clean,
                    request_id=heldout_request_id(binding, 0, 'clean'))
    public = codec.serialize_public(0, 'clean', clean)
    private = codec.serialize_private(0, 'clean', clean, ())
    journal = HeldoutJournal(tmp_path / 'journal', codec=codec, binding=binding,
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    original = type(codec).read_public
    calls = []
    def one_shot_oom(self, raw):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError('fabricated out of memory')
        return original(self, raw)
    monkeypatch.setattr(type(codec), 'read_public', one_shot_oom)
    with pytest.raises(MemoryError):
        journal.append(public, private)
    assert calls == [1]
    assert public in (tmp_path / 'journal/000.frame').read_bytes()
    assert not (tmp_path / 'journal/000.ok').exists()
    assert (tmp_path / 'journal/terminal.json').is_file()
    with pytest.raises(RuntimeError):
        journal.append(public, private)


def test_final_ack_write_failure_cannot_substitute_for_returned_receipt(tmp_path, monkeypatch):
    from looking_glass import injection_heldout as module
    rows = tuple((1000 + i, 2000 + i, 3000 + i) for i in range(8))
    codec, binding = bind_heldout_codec(rows), fake_binding()
    journal = HeldoutJournal(tmp_path / 'journal', codec=codec, binding=binding,
                             site=2, max_disk_bytes=8 * 1024 * 1024)
    journal._ordinal = 127
    journal._baseline[7] = (None, None)
    clean = replace(fabricated_cases(ids_rows=rows)[7].verify_clean,
                    request_id=heldout_request_id(binding, 7, 'verify_clean'))
    public, private = codec.serialize_public(7, 'verify_clean', clean), codec.serialize_private(7, 'verify_clean', clean, ())
    monkeypatch.setattr(type(codec), 'audit_pair', lambda *args, **kwargs: {'verified': True})
    real_write = module._journal._write_parts
    def write_then_fail(path, parts):
        if path.suffix == '.ok':
            real_write(path, parts)
            raise OSError('acknowledgment durability unknown')
        if path.name == 'terminal.json':
            raise OSError('terminal unavailable')
        return real_write(path, parts)
    monkeypatch.setattr(module._journal, '_write_parts', write_then_fail)
    with pytest.raises(OSError):
        journal.append(public, private)
    assert (tmp_path / 'journal/127.frame').is_file()
    assert (tmp_path / 'journal/127.ok').is_file()
    assert not (tmp_path / 'journal/terminal.json').exists()
    with pytest.raises(ValueError):
        inspect_heldout_journal(tmp_path / 'journal', codec=codec, binding=binding, site=2,
                                max_disk_bytes=8 * 1024 * 1024, acknowledgments=(), externally_quiescent=True)


def test_committed_target_order_uses_first_lemma_of_each_family():
    assert committed_targets is not None, 'committed case mapper is missing'
    families = []
    for family in range(4):
        cases = [{'text': f'fabricated {family} {item}', 'input_ids': [100 + family, 200 + item, 300],
                  'attention_mask': [1, 1, 1], 'subject_index': 1} for item in range(4)]
        families.append({'family_index': family, 'context': 'fabricated',
                         'subjects': ['a', 'as', 'b', 'bs'], 'cases': cases})
    record = {'schema': 'mlg-number-heldout-v1', 'protocol_sha256': 'a' * 64,
              'source_commits': {'injection': 'b' * 40}, 'checkpoint_revision': 'c' * 40,
              'prediction_freeze_sha256': 'd' * 64, 'exposure_inventory_sha256': 'e' * 64,
              'families': families}
    result = committed_targets(json.dumps(record, sort_keys=True, separators=(',', ':')).encode())
    assert [item.text for item in result] == [f'fabricated {family} {item}'
                                                  for family in range(4) for item in range(2)]
    assert [item.input_ids for item in result] == [(100 + family, 200 + item, 300)
                                                       for family in range(4) for item in range(2)]
