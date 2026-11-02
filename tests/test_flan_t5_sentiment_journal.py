"""Bounded, synthetic-only journal tests; no model or held-out cases."""
import json
from pathlib import Path
from unittest import mock

import pytest

from looking_glass.flan_t5_sentiment_journal import (
    EvidenceJournal, JournalError, Limits, WORST_CASE_BYTES, require_protocol_allocation,
)


def small_limits(**changes):
    values = dict(total_bytes=400_000, metadata_bytes=200_000, failure_bytes=10_000,
                  forward_bytes=1024, max_forwards=4, free_bytes=0)
    values.update(changes)
    return Limits(**values)


def test_journal_persists_start_before_tensor_and_completion(tmp_path):
    journal = EvidenceJournal(tmp_path / 'attempt', {'schema': 'fabricated'}, limits=small_limits())
    journal.begin_component({'kind': 'decoder', 'case': 'synthetic'}, reserve_bytes=16)
    events = [json.loads(line) for line in (journal.directory / 'events.jsonl').read_text().splitlines()]
    assert [item['event'] for item in events] == ['component_start']
    ref = journal.tensor((1, 4), bytes(16))
    journal.finish_component({'logits': ref})
    events = [json.loads(line) for line in (journal.directory / 'events.jsonl').read_text().splitlines()]
    assert [item['event'] for item in events] == ['component_start', 'component_return']
    assert (journal.directory / ref['path']).read_bytes() == bytes(16)
    journal.finish('complete_candidate', {'testing': True})
    assert json.loads((journal.directory / 'terminal.json').read_text())['status'] == 'complete_candidate'


def test_insufficient_capacity_or_space_is_rejected_before_component(tmp_path):
    with mock.patch('looking_glass.flan_t5_sentiment_journal.shutil.disk_usage',
                    return_value=(100, 99, 1)):
        with pytest.raises(JournalError, match='free'):
            EvidenceJournal(tmp_path / 'no-space', {}, limits=small_limits(free_bytes=2))
    assert not (tmp_path / 'no-space').exists()
    journal = EvidenceJournal(tmp_path / 'attempt', {}, limits=small_limits(forward_bytes=8))
    with pytest.raises(JournalError, match='reservation'):
        journal.begin_component({'kind': 'decoder'}, reserve_bytes=16)
    assert journal.component_count == 0


def test_failed_or_unfinished_component_cannot_publish_complete(tmp_path):
    journal = EvidenceJournal(tmp_path / 'attempt', {}, limits=small_limits())
    journal.begin_component({'kind': 'encoder'}, reserve_bytes=16)
    with pytest.raises(JournalError, match='incomplete'):
        journal.finish('complete_candidate', {})
    journal.finish('incomplete', {'reason': 'synthetic failure'})
    terminal = json.loads((journal.directory / 'terminal.json').read_text())
    assert terminal['status'] == 'incomplete'
    assert terminal['active_component']['kind'] == 'encoder'
    assert journal.component_count == 1


def test_write_failure_keeps_start_and_forbids_further_forward(tmp_path):
    journal = EvidenceJournal(tmp_path / 'attempt', {}, limits=small_limits())
    journal.begin_component({'kind': 'decoder'}, reserve_bytes=16)
    with mock.patch('looking_glass.flan_t5_sentiment_journal._write_new', side_effect=OSError('synthetic')):
        with pytest.raises(OSError):
            journal.tensor((1, 4), bytes(16))
    with pytest.raises(JournalError):
        journal.begin_component({'kind': 'decoder'}, reserve_bytes=16)
    journal.finish('incomplete', {'reason': 'write-failed'})
    terminal = json.loads((journal.directory / 'terminal.json').read_text())
    assert terminal['storage_accounting_complete'] is False


def test_protocol_preflight_rejects_insufficient_retained_capacity():
    assert require_protocol_allocation(Limits()) == WORST_CASE_BYTES
    with pytest.raises(JournalError, match='allocation'):
        require_protocol_allocation(Limits(total_bytes=WORST_CASE_BYTES - 1))
    with pytest.raises(JournalError, match='allocation'):
        require_protocol_allocation(Limits(free_bytes=0))
