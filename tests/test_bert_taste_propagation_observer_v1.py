"""Tiny untrained-model tests for same-forward downstream capture."""
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from scripts import bert_taste_propagation_observer_v1 as observer
from scripts import bert_taste_patch_v1 as previous


def tiny_model():
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(9261)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32, n_layers=6, dim=16,
                              hidden_dim=32, n_heads=2, dropout=0., attention_dropout=0.)
    config._attn_implementation = 'eager'
    return DistilBertForMaskedLM(config).eval()


def same(left, right):
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def test_early_patch_and_later_captures_are_same_forward():
    model = tiny_model()
    ids = (0, 1, 2, 9, 10, 5, 6, 3, 11, 12)
    clean = observer.observe(model, ids, layer=0, position=7, mask_index=4, capture_late=True)
    identity = observer.observe(model, ids, layer=0, position=7, mask_index=4,
                                self_patch=True, capture_late=True)
    assert same(clean.mask_logits, identity.mask_logits)
    assert same(clean.before, identity.before) and same(identity.before, identity.after)
    assert all(same(clean.later[name], identity.later[name]) for name in observer.LATE_POSITIONS)
    source = clean.before.clone()
    source[0] += .5
    changed = observer.observe(model, ids, layer=0, position=7, mask_index=4,
                               source=source, capture_late=True)
    assert changed.model_entry == {'input_ids': list(ids), 'attention_mask': [1] * 10}
    assert same(clean.before, changed.before) and same(source, changed.after)
    assert changed.untouched_positions_equal
    assert not same(clean.later['late_the'], changed.later['late_the'])
    names = [event['name'] for event in changed.events]
    assert names == ['full_before', 'before', 'patch', 'after', 'full_after', 'late_full',
                     'late_ingredient', 'late_the', 'late_tasted', 'late_mask']
    assert all(event['module_path'].endswith('.0') for event in changed.events[:5])
    assert all(event['module_path'].endswith('.3') for event in changed.events[5:])
    assert 'forward' not in model.__dict__ and not previous.has_hooks(model)


def test_late_patch_and_final_block_control():
    model = tiny_model()
    left = (0, 1, 2, 9, 10, 5, 6, 3, 11, 12)
    right = (0, 1, 2, 9, 10, 5, 6, 4, 11, 12)
    baseline = observer.observe(model, left, layer=3, position=1, mask_index=4)
    donor = observer.observe(model, right, layer=3, position=1, mask_index=4)
    patched = observer.observe(model, left, layer=3, position=1, mask_index=4, source=donor.before)
    assert same(patched.before, baseline.before) and same(patched.after, donor.before)
    assert patched.untouched_positions_equal and patched.later == {}
    final_left = observer.observe(model, left, layer=5, position=7, mask_index=4)
    final_right = observer.observe(model, right, layer=5, position=7, mask_index=4)
    final_patch = observer.observe(model, left, layer=5, position=7, mask_index=4, source=final_right.before)
    assert same(final_left.mask_logits, final_patch.mask_logits)
    assert not previous.has_hooks(model)
