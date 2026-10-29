"""Execute one fixed matched-dose BERT family on CPU.

This module neither authenticates a checkpoint nor grants process custody. Its
caller must verify the committed manifest, runtime, model and exposure ledger,
exclusively own the model, and use a durable bounded event sink. The in-process
audit is repeated independently on the closed journal before publication.
"""
import base64
import copy
import hashlib
import struct

import torch

from .bert_agreement_runner import Observation, _family_rows, float32_bytes, run_observation
from .bert_matched_evidence import audit_family_events
from .bert_matched_numeric import DONORS, SITES, binary64_rounding_probe, plan_case


def _encoded(value):
    return base64.b64encode(value).decode('ascii')


def _stored_plan(plan):
    return {site: {name: {key if key != 'value' else 'value_float32_le_base64':
                           _encoded(value) if key == 'value' else value
                          for key, value in action.items()}
                   for name, action in plan['sites'][site].items()} for site in SITES}


def run_family(model, manifest, *, family, emit, split='evaluation', is_id=2003,
               are_id=2024, mask_id=103):
    """Run exactly 60 observations for a family on the CPU-only primitive.

    All source/asset checks, model-state witness, calling-thread numerical
    probes before and after the whole attempt, and process cleanup belong to
    the supervised owner. An exception always invalidates the attempt.
    """
    return _run_family(model, manifest, split=split, family=family, is_id=is_id,
                       are_id=are_id, mask_id=mask_id, emit=emit, observer=run_observation)


def _run_family(model, manifest, *, split, family, is_id, are_id, mask_id, emit, observer):
    """Private observer seam for fabricated tests; it grants no execution trust."""
    if not callable(emit) or not callable(observer):
        raise ValueError('event sink and trusted observer are required')
    rows, targets, by_id = _family_rows(manifest, split, family, mask_id)
    if len(model.distilbert.transformer.layer) != 6:
        raise ValueError('the frozen study requires six encoder blocks')
    binary64_rounding_probe()
    records, captures, references = [], {site: {} for site in SITES}, {}
    count = 0

    def write(record):
        emit(record)
        records.append(copy.deepcopy(record))

    def forward(kind, row_id, site, *, case_id=None, donor_id=None,
                source_bytes=None, self_patch=False, expected_before=None):
        nonlocal count
        count += 1
        if count > 60 or site not in SITES:
            raise RuntimeError('family forward budget or site exceeded')
        layer = 0 if site == 'block0' else 2
        identity = {'split': split, 'family': family, 'forward_index': count,
                    'kind': kind, 'row_id': row_id, 'donor_row_id': donor_id,
                    'case_id': case_id, 'site': site, 'layer': layer, 'token': 2}
        write({'status': 'started', **identity})
        try:
            inputs = torch.tensor([by_id[row_id]['input_ids']], dtype=torch.int64, device='cpu')
            source = (None if source_bytes is None else
                      torch.frombuffer(bytearray(source_bytes), dtype=torch.int32).view(torch.float32))
            observation = observer(model, inputs, layer=layer, token=2, mask_token=6,
                                   is_id=is_id, are_id=are_id, source=source, self_patch=self_patch)
            if type(observation) is not Observation:
                raise TypeError('observer returned an unsupported observation')
            if (observation.model_input_ids != tuple(by_id[row_id]['input_ids'])
                    or observation.model_attention_mask != tuple(by_id[row_id]['attention_mask'])):
                raise RuntimeError('model inputs differ from the declared row')
            before = float32_bytes(observation.before)
            after = None if observation.after is None else float32_bytes(observation.after)
            mask_raw = float32_bytes(observation.mask_logits)
            if len(before) != 3072 or len(mask_raw) <= 4 * max(is_id, are_id):
                raise RuntimeError('unexpected vector or mask shape')
            if expected_before is not None and before != expected_before:
                raise RuntimeError('pre-action vector differs from its clean capture')
            expected_after = before if self_patch else source_bytes
            if after != expected_after:
                raise RuntimeError('action bits do not match the submitted vector')
            if (type(observation.is_bits) is not int or type(observation.are_bits) is not int
                    or observation.is_bits != struct.unpack_from('<I', mask_raw, 4 * is_id)[0]
                    or observation.are_bits != struct.unpack_from('<I', mask_raw, 4 * are_id)[0]):
                raise RuntimeError('selected logits differ from the retained mask vector')
            completed = {'status': 'completed', **identity,
                         'is_bits': observation.is_bits, 'are_bits': observation.are_bits,
                         'is_logit': observation.is_logit, 'are_logit': observation.are_logit,
                         'mask_logits_sha256': hashlib.sha256(mask_raw).hexdigest(),
                         'before_float32_le_base64': _encoded(before),
                         'after_float32_le_base64': None if after is None else _encoded(after),
                         'submitted_float32_le_base64': None if source_bytes is None else _encoded(source_bytes),
                         'events': list(observation.events), 'capture_bytes': observation.capture_bytes,
                         'source_bytes': observation.source_bytes,
                         'model_input_ids': list(observation.model_input_ids),
                         'model_attention_mask': list(observation.model_attention_mask)}
            write(completed)
        except BaseException as error:
            try:
                write({'status': 'failed', **identity, 'error_type': type(error).__name__})
            except (KeyboardInterrupt, SystemExit) as cancellation:
                raise cancellation from error
            except Exception:
                pass
            raise
        return before, mask_raw

    def check(name, case_id, site, actual, expected):
        passed = actual == expected
        write({'status': 'control', 'split': split, 'family': family, 'name': name,
               'case_id': case_id, 'site': site, 'passed': passed})
        if not passed:
            raise RuntimeError('required matched-dose control failed: ' + name)

    for site in SITES:
        for row in rows:
            row_id = row['row_id']
            before, mask = forward('capture', row_id, site)
            captures[site][row_id] = before
            if site == 'block0':
                references[row_id] = mask
            else:
                check('clean_matches_block0', row_id, site, mask, references[row_id])

    for target in targets:
        target_id, case_id = target['target_row_id'], target['case_id']
        bases = {site: captures[site][target_id] for site in SITES}
        donors = {site: {cell: captures[site][target['donor_row_ids'][cell]]
                         for cell in DONORS} for site in SITES}
        try:
            plan = plan_case(target_id, bases, donors)
        except BaseException as error:
            try:
                write({'status': 'dose_plan_failed', 'split': split, 'family': family,
                       'case_id': case_id, 'row_id': target_id, 'error_type': type(error).__name__})
            except (KeyboardInterrupt, SystemExit) as cancellation:
                raise cancellation from error
            except Exception:
                pass
            raise
        write({'status': 'dose_plan', 'split': split, 'family': family, 'case_id': case_id,
               'row_id': target_id, 'radius': plan['radius'], 'sites': _stored_plan(plan)})
        for site in SITES:
            _, mask = forward('self', target_id, site, case_id=case_id, donor_id=target_id,
                              self_patch=True, expected_before=bases[site])
            check('self_patch', case_id, site, mask, references[target_id])
            for cell in DONORS:
                forward(cell, target_id, site, case_id=case_id,
                        donor_id=target['donor_row_ids'][cell],
                        source_bytes=plan['sites'][site][cell]['value'], expected_before=bases[site])
            forward('artificial', target_id, site, case_id=case_id,
                    source_bytes=plan['sites'][site]['artificial']['value'], expected_before=bases[site])

    for target in targets:
        target_id, case_id = target['target_row_id'], target['case_id']
        _, mask = forward('final_clean', target_id, 'block2', case_id=case_id,
                          expected_before=captures['block2'][target_id])
        check('final_clean', case_id, 'block2', mask, references[target_id])
    if count != 60:
        raise RuntimeError('family forward budget did not complete')
    binary64_rounding_probe()
    return audit_family_events(manifest, records, split=split, family=family, mask_id=mask_id)
