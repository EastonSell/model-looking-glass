"""Regression for the fixed successor schedule; no model or protected inputs."""
import hashlib
import json
import pytest
from looking_glass import agreement_invariant_successor_science_v2 as reader

def test_successor_schedule_and_identity_are_bound():
    pairs = reader.fixed_schedule()
    raw = json.dumps(pairs, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert len(pairs) == 612
    assert hashlib.sha256(raw).hexdigest() == "f0625d27ee196eeeb1a85a3577238f0cbcad19017fc6596dfe429dcbc60c610d"
    assert all(name.startswith("v2_discovery_") for name, _ in pairs[:306])
    assert all(name.startswith("evaluation_") for name, _ in pairs[306:])
    assert reader._identity("v2_discovery_target_f0_c0_n0_this")["split"] == "discovery"
    with pytest.raises(ValueError):
        reader._identity("discovery_target_f0_c0_n0_this")


def test_production_start_scope_is_the_exact_successor_scope(monkeypatch):
    class ReachedEvidence(Exception):
        pass
    def reached(_):
        raise ReachedEvidence()
    monkeypatch.setattr(reader, 'ordinary_directory', reached)
    binding = {'prelaunch_sha256': '1'*64, 'reservation_sha256': '2'*64, 'latest_exposure_sha256': '3'*64}
    kwargs = dict(expected_binding=binding, expected_source_revision='a'*40, expected_events_sha256='4'*64)
    with pytest.raises(ReachedEvidence):
        reader.audit_attempt('unused', [], {}, None, expected_scope='public prospective agreement invariant-noun successor v2', **kwargs)
    with pytest.raises(ValueError, match='identities'):
        reader.audit_attempt('unused', [], {}, None, expected_scope='public prospective agreement invariant-noun', **kwargs)
