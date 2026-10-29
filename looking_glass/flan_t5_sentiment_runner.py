"""CPU sentiment execution core, to be admitted by a separately reviewed owner.

The caller must bind the frozen selector, custody, assets and source, establish
process teardown, and independently audit saved evidence. A complete candidate
is a provisional saved execution, not proof of admission or scientific outcome.
"""
import copy
import hashlib
import json
import math
import os
import re
import struct
import sys
import unicodedata

import torch

from .bert_matched_numeric import f32_bits_to_f64
from .flan_t5_sentiment_journal import EvidenceJournal, JournalError, require_protocol_allocation
from .flan_t5_sentiment_numeric import (
    ACTIONS, OPPOSITE, RECEIVERS, SAME, _cosine, _norm, fixed_schedule, group_decision, plan_group,
    primary_decision, runtime_numeric_probe,
)


def _label_targets(tokenizer, *, vocab_size):
    if type(vocab_size) is not int or not 2 < vocab_size <= 32128:
        raise ValueError('invalid vocabulary')
    special = set(tokenizer.all_special_ids)
    targets = {}
    for label in ('positive', 'negative'):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if (type(ids) is not list or not 1 <= len(ids) <= 4
                or any(type(x) is not int or not 0 <= x < vocab_size or x in special for x in ids)):
            raise ValueError('label lexical tokenization is invalid')
        targets[label] = ids + [1]
    if targets['positive'] == targets['negative']:
        raise ValueError('label tokenizations are identical')
    return targets


def _raw(tensor):
    if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32
            or tensor.device.type != 'cpu' or tensor.layout != torch.strided or sys.byteorder != 'little'):
        raise RuntimeError('expected a CPU little-endian float32 tensor')
    return tensor.detach().contiguous().view(torch.int32).numpy().tobytes()


def _finite(raw):
    return all((word & 0x7f800000) != 0x7f800000 for (word,) in struct.iter_unpack('<I', raw))


def _same_tensor(actual, snapshot):
    """Check type, representation and values; torch.equal alone ignores dtype."""
    return (type(actual) is torch.Tensor and type(snapshot) is torch.Tensor
            and actual.dtype == snapshot.dtype and actual.device == snapshot.device
            and actual.layout == snapshot.layout and actual.shape == snapshot.shape
            and torch.equal(actual, snapshot))


def _values(raw):
    return tuple(f32_bits_to_f64(word) for (word,) in struct.iter_unpack('<I', raw))


def _strict_answer(tokenizer, ids, expected):
    if (not ids or ids[-1] != 1 or ids.count(1) != 1
            or any(token in set(tokenizer.all_special_ids) for token in ids[:-1])):
        return False
    text = tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return unicodedata.normalize('NFC', text).strip().casefold() == expected


def _check_model_mode(model):
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError('model or a child module left evaluation mode')
    for value in tuple(model.parameters()) + tuple(model.buffers()):
        if value.device.type != 'cpu' or (value.is_floating_point() and value.dtype != torch.float32):
            raise RuntimeError('model state left CPU float32')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized during CPU execution')


def _runtime_state(model):
    """Observe calling-thread settings; the external owner must set the policy."""
    _check_model_mode(model)
    keys = ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    return {'device': 'cpu', 'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'python_version': sys.version.split()[0], 'torch_version': str(torch.__version__),
            'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'default_dtype': str(torch.get_default_dtype()),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cpu_autocast': torch.is_autocast_enabled('cpu'),
            'attention_implementation': getattr(model.config, '_attn_implementation', None),
            'config_use_cache': getattr(model.config, 'use_cache', None),
            'hidden_width': getattr(model.config, 'd_model', None),
            'vocabulary': getattr(model.config, 'vocab_size', None),
            'decoder_start_id': getattr(model.config, 'decoder_start_token_id', None),
            'eos_id': getattr(model.config, 'eos_token_id', None),
            'environment': {key: os.environ.get(key) for key in keys}}


def _require_live_runtime(runtime):
    """Require the intended CPU settings; this does not authenticate them."""
    expected = {'device': 'cpu', 'num_threads': 1, 'num_interop_threads': 1,
                'deterministic': True, 'deterministic_warn_only': False,
                'default_dtype': 'torch.float32', 'float32_matmul_precision': 'highest',
                'cpu_autocast': False, 'attention_implementation': 'eager',
                'config_use_cache': False, 'hidden_width': 512, 'vocabulary': 32128,
                'decoder_start_id': 0, 'eos_id': 1}
    environment = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1'}
    if (type(runtime) is not dict
            or any(type(runtime.get(key)) is not type(value) or runtime.get(key) != value
                   for key, value in expected.items())
            or type(runtime.get('environment')) is not dict
            or any(type(runtime['environment'].get(key)) is not str
                   or runtime['environment'].get(key) != value for key, value in environment.items())):
        raise RuntimeError('CPU runtime does not match the prospective contract')


class _EvaluationRunner:
    """Output evaluator; it does not independently authorize execution."""

    def __init__(self, model, tokenizer, journal, *, width, vocab_size, test_only=False, custody_check=None):
        if type(test_only) is not bool or (custody_check is not None and not callable(custody_check)):
            raise ValueError('invalid evaluator or custody callback')
        if not test_only:
            from transformers import T5ForConditionalGeneration, T5TokenizerFast
            if (type(model) is not T5ForConditionalGeneration or type(tokenizer) is not T5TokenizerFast
                    or tokenizer.pad_token_id != 0 or tokenizer.eos_token_id != 1
                    or getattr(model.config, 'pad_token_id', None) != 0
                    or not callable(custody_check)):
                raise ValueError('live execution requires the exact T5 classes, token IDs and custody callback')
        if (type(width) is not int or not 1 <= width <= 512 or type(vocab_size) is not int
                or not 2 < vocab_size <= 32128 or model.training
                or any(module.training for module in model.modules())
                or getattr(model.config, 'd_model', None) != width
                or getattr(model.config, 'vocab_size', None) != vocab_size
                or getattr(model.config, 'decoder_start_token_id', None) != 0
                or getattr(model.config, 'eos_token_id', None) != 1):
            raise ValueError('model dimensions or evaluation configuration do not match')
        for value in tuple(model.parameters()) + tuple(model.buffers()):
            if value.device.type != 'cpu' or (value.is_floating_point() and value.dtype != torch.float32):
                raise ValueError('model state must be CPU float32')
        self.model, self.tokenizer, self.journal = model, tokenizer, journal
        self.width, self.vocab_size = width, vocab_size
        self._custody_check = custody_check
        self._failed = self._evaluating = False
        self.runtime = _runtime_state(model)
        if not test_only:
            _require_live_runtime(self.runtime)

    def _check_runtime(self):
        if self._custody_check is not None:
            self._custody_check()
        if _runtime_state(self.model) != self.runtime:
            raise RuntimeError('CPU runtime settings changed during the attempt')

    def _decoder(self, case, state, state_raw, prefix, *, phase, final_only):
        if not self._evaluating:
            raise RuntimeError('an uncertain component forbids more forwards')
        numeric_before = runtime_numeric_probe()
        self._check_runtime()
        source_ids = torch.tensor([case['input_ids']], dtype=torch.int64)
        source_mask = torch.tensor([case['attention_mask']], dtype=torch.int64)
        decoder_ids = torch.tensor([prefix], dtype=torch.int64)
        decoder_mask = torch.ones_like(decoder_ids)
        state_shape = tuple(state.shape)
        originals = tuple(x.clone() for x in (source_ids, source_mask, decoder_ids, decoder_mask))
        max_return = (len(prefix) + 1) * self.vocab_size * 4 + len(state_raw)
        self.journal.begin_component({'kind': 'decoder', 'phase': phase,
                                      'side': case['side'], 'decoder_ids': list(prefix),
                                      'decoder_mask': [1] * len(prefix),
                                      'numeric_before': numeric_before}, reserve_bytes=max_return)
        output = logits = ref = None
        rejected = False
        try:
            with torch.inference_mode():
                self._check_runtime()
                output = self.model(encoder_outputs=(state,), attention_mask=source_mask,
                                    decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
                                    use_cache=False, return_dict=True)
            logits = getattr(output, 'logits', None)
            self._check_runtime()
            if (type(logits) is not torch.Tensor or logits.ndim != 3
                    or tuple(logits.shape) != (1, len(prefix), self.vocab_size)):
                raise RuntimeError('decoder returned malformed logits')
            raw = _raw(logits)
            saved_raw = raw[-self.vocab_size * 4:] if final_only else raw
            saved_shape = (1, self.vocab_size) if final_only else tuple(logits.shape)
            ref = self.journal.tensor(saved_shape, saved_raw)
            problems = []
            extra = {}
            if not _finite(raw):
                problems.append('nonfinite logits')
                if final_only and len(raw) != len(saved_raw):
                    extra['full_return'] = self.journal.tensor(tuple(logits.shape), raw)
            current = _raw(state)
            if tuple(state.shape) != state_shape or current != state_raw:
                problems.append('encoder state mutated')
                extra['observed_state'] = self.journal.tensor(tuple(state.shape), current)
            if any(not _same_tensor(a, b) for a, b in zip((source_ids, source_mask, decoder_ids, decoder_mask), originals)):
                problems.append('source or decoder input mutated')
            if getattr(output, 'past_key_values', None) is not None:
                problems.append('decoder returned a cache')
            self._check_runtime()
            numeric_after = runtime_numeric_probe()
            if problems:
                self.journal.record('component_rejected', {'reasons': problems, 'logits': ref, **extra})
                rejected = True
                raise RuntimeError('decoder validation failed: ' + ', '.join(problems))
            self.journal.finish_component({'logits': ref, 'prefix': list(prefix), 'phase': phase,
                                           'numeric_after': numeric_after})
            return logits, saved_raw, ref
        except BaseException as primary:
            self._failed = True
            self._evaluating = False
            if not rejected:
                try:
                    salvage = {'error_type': type(primary).__name__, 'return_observed': output is not None,
                               'logits': ref}
                    if type(logits) is torch.Tensor:
                        salvage['original_shape'] = list(logits.shape)
                        capacity = self.journal.remaining_component_bytes // 4
                        if (ref is None and logits.device.type == 'cpu' and logits.dtype == torch.float32
                                and logits.layout == torch.strided and logits.is_contiguous() and capacity > 0):
                            count = min(logits.numel(), capacity)
                            if count > 0:
                                part = logits.detach().view(-1)[:count]
                                salvage['retained_prefix'] = self.journal.tensor((count,), _raw(part))
                                salvage['truncated'] = count != logits.numel()
                    self.journal.record('component_rejected', salvage)
                except BaseException as salvage_error:
                    if isinstance(primary, Exception) and not isinstance(salvage_error, Exception):
                        raise salvage_error from primary
            raise

    def evaluate(self, case, before, after, action, targets):
        if self._failed or self._evaluating:
            raise RuntimeError('an incomplete evaluation forbids more forwards')
        # Install the fence before any evidence, model or callback work. The
        # caller must stop on a lost return even after a successful record.
        self._failed = True
        self._evaluating = True
        result = self._evaluate(case, before, after, action, targets)
        self._evaluating = False
        self._failed = False
        return result

    def _evaluate(self, case, before, after, action, targets):
        if (type(case) is not dict or case.get('side') not in ('P1', 'N1', 'P2', 'N2')
                or type(case.get('input_ids')) is not list or type(case.get('attention_mask')) is not list
                or not 1 <= len(case['input_ids']) <= 64
                or len(case['attention_mask']) != len(case['input_ids'])
                or any(type(x) is not int or x != 1 for x in case['attention_mask'])
                or any(type(x) is not int or not 0 <= x < self.vocab_size for x in case['input_ids'])
                or type(action) is not dict or targets != _label_targets(self.tokenizer, vocab_size=self.vocab_size)):
            raise ValueError('invalid evaluation inputs')
        shape = (1, len(case['input_ids']), self.width)
        if (type(before) is not torch.Tensor or type(after) is not torch.Tensor
                or tuple(before.shape) != shape or tuple(after.shape) != shape):
            raise ValueError('encoder witness shape does not match the case')
        before_raw, state_raw = _raw(before), _raw(after)
        if not _finite(before_raw) or not _finite(state_raw):
            raise ValueError('nonfinite encoder witness')
        before_ref = self.journal.tensor(shape, before_raw)
        after_ref = self.journal.tensor(shape, state_raw)
        self.journal.record('evaluation_start', {'action': action, 'side': case['side'],
                                                  'input_ids': case['input_ids'],
                                                  'attention_mask': case['attention_mask'],
                                                  'before': before_ref, 'after': after_ref})
        greedy, generated, all_refs = [], [], []
        for _ in range(4):
            prefix = [0] + generated
            _, raw, ref = self._decoder(case, after, state_raw, prefix, phase='greedy', final_only=True)
            values = _values(raw)
            emitted = max(range(len(values)), key=values.__getitem__)
            generated.append(emitted)
            greedy.append({'prefix': prefix, 'emitted': emitted, 'logits': ref})
            all_refs.append(ref)
            if emitted == 1:
                break
        teacher = {}
        for label in ('positive', 'negative'):
            ids = targets[label]
            decoder_ids = [0] + ids[:-1]
            logits, _, ref = self._decoder(case, after, state_raw, decoder_ids,
                                           phase='teacher_' + label, final_only=False)
            score_numeric_before = runtime_numeric_probe()
            selected = torch.log_softmax(logits[0], dim=-1)[torch.arange(len(ids)),
                                                             torch.tensor(ids, dtype=torch.int64)]
            words = selected.detach().contiguous().view(torch.int32).tolist()
            per_token = [f32_bits_to_f64(int(word) & 0xffffffff) for word in words]
            if not all(math.isfinite(x) for x in per_token):
                self._failed = True
                raise RuntimeError('teacher-forced score is nonfinite')
            ranks = [1 + int((logits[0, i] > logits[0, i, token]).sum().item())
                     for i, token in enumerate(ids)]
            score = math.fsum(per_token)
            score_numeric_after = runtime_numeric_probe()
            teacher[label] = {'target_ids': list(ids), 'decoder_ids': decoder_ids,
                              'decoder_mask': [1] * len(decoder_ids), 'logits': ref,
                              'selected_logprobs': per_token, 'score': score,
                              'ranks': ranks, 'numeric_before': score_numeric_before,
                              'numeric_after': score_numeric_after}
            all_refs.append(ref)
        margin_before = runtime_numeric_probe()
        margin = teacher['positive']['score'] - teacher['negative']['score']
        margin_after = runtime_numeric_probe()
        expected = 'positive' if case['side'].startswith('P') else 'negative'
        result = {'action': action, 'side': case['side'], 'before': before_ref, 'after': after_ref,
                  'greedy': greedy, 'generated_ids': generated, 'teacher': teacher,
                  'strict_correct': _strict_answer(self.tokenizer, generated, expected),
                  'margin': margin, 'margin_numeric_before': margin_before,
                  'margin_numeric_after': margin_after, 'all_logit_refs': all_refs}
        self.journal.record('evaluation_return', result)
        return result


def _validated_groups(groups, *, vocab_size, special_ids):
    if type(groups) not in (tuple, list) or len(groups) != 8:
        raise ValueError('expected four units with two instruction groups')
    groups = copy.deepcopy(groups)
    result = {}
    for group in groups:
        if (type(group) is not dict or set(group) != {'unit', 'instruction', 'cases'}
                or type(group['unit']) is not int or not 0 <= group['unit'] < 4
                or type(group['instruction']) is not int or group['instruction'] not in (0, 1)
                or type(group['cases']) is not dict or set(group['cases']) != set(RECEIVERS)):
            raise ValueError('invalid group identity')
        key = (group['unit'], group['instruction'])
        if key in result:
            raise ValueError('duplicate group')
        reference = None
        adjectives = []
        for side in RECEIVERS:
            case = group['cases'][side]
            if (type(case) is not dict or case.get('side') != side
                    or type(case.get('input_ids')) is not list or not 3 <= len(case['input_ids']) <= 64
                    or any(type(v) is not int or not 0 <= v < vocab_size for v in case['input_ids'])
                    or case['input_ids'][-1] != 1 or case['input_ids'].count(1) != 1
                    or type(case.get('attention_mask')) is not list
                    or len(case['attention_mask']) != len(case['input_ids'])
                    or any(type(v) is not int or v != 1 for v in case['attention_mask'])
                    or type(case.get('special_tokens_mask')) is not list
                    or len(case['special_tokens_mask']) != len(case['input_ids'])
                    or any(type(v) is not int or v not in (0, 1) for v in case['special_tokens_mask'])
                    or type(case.get('subject_index')) is not int or type(case.get('adjective_index')) is not int):
                raise ValueError('invalid selected case')
            subject, adjective = case['subject_index'], case['adjective_index']
            if (not 0 <= subject < len(case['input_ids']) or not 0 <= adjective < len(case['input_ids'])
                    or len({0, subject, adjective}) != 3
                    or any(case['special_tokens_mask'][index] != 0
                           or case['input_ids'][index] in special_ids for index in (0, subject, adjective))):
                raise ValueError('selected positions are invalid')
            static = (tuple(v for i, v in enumerate(case['input_ids']) if i != adjective),
                      tuple(case['attention_mask']), tuple(case['special_tokens_mask']), subject, adjective)
            if reference is not None and static != reference:
                raise ValueError('group inputs differ outside the adjective')
            reference = static
            adjectives.append(case['input_ids'][adjective])
        if len(set(adjectives)) != 4:
            raise ValueError('adjective token IDs must differ')
        result[key] = group
    return result


def selection_to_groups(selection, *, expected_runner_revision, vocab_size, special_ids):
    """Project a structurally checked selector record to fixed runner groups.

    This does not verify a commitment, publication, source origin, exposure
    currentness, tokenizer output or custodian independence. Those remain
    mandatory pre-forward responsibilities of the future controller.
    """
    if type(selection) is not dict:
        raise ValueError('selection must be a mapping')
    try:
        raw = json.dumps(selection, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf8')
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('selection is not finite JSON') from error
    if len(raw) > 8 * 1024**2:
        raise ValueError('selection exceeds the custody allowance')
    # The controller must exclusively own the input during this snapshot.
    selection = copy.deepcopy(selection)
    digest = lambda value, length: type(value) is str and re.fullmatch(r'[0-9a-f]{%d}' % length, value) is not None
    keys = {'schema', 'pool_commitment', 'protocol_sha256', 'runner_revision', 'asset_manifest_sha256',
            'exposure_sources', 'unit_screens', 'selected_units', 'donor_mapping'}
    if (type(selection) is not dict or set(selection) != keys
            or selection['schema'] != 'flan-sentiment-private-selection-v1'
            or not digest(selection['pool_commitment'], 64)
            or selection['protocol_sha256'] != '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813'
            or selection['asset_manifest_sha256'] != 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
            or not digest(expected_runner_revision, 40) or selection['runner_revision'] != expected_runner_revision
            or type(vocab_size) is not int or not 2 < vocab_size <= 32128
            or type(special_ids) not in (set, frozenset)
            or any(type(x) is not int or not 0 <= x < vocab_size for x in special_ids)):
        raise ValueError('selection identity or runner binding differs')
    sources = selection['exposure_sources']
    if type(sources) is not list or not 1 <= len(sources) <= 64:
        raise ValueError('invalid exposure source inventory')
    paths = set()
    for source in sources:
        if (type(source) is not dict or set(source) != {'path', 'revision', 'sha256'}
                or type(source['path']) is not str or not source['path']
                or len(source['path'].encode('utf8')) > 256 or source['path'].startswith(('/', '\\'))
                or '\\' in source['path'] or ':' in source['path'] or '\0' in source['path']
                or unicodedata.normalize('NFC', source['path']) != source['path']
                or any(part in ('', '.', '..') for part in source['path'].split('/'))
                or source['path'] in paths or not digest(source['revision'], 40)
                or not digest(source['sha256'], 64)):
            raise ValueError('invalid exposure source descriptor')
        paths.add(source['path'])
    screens = selection['unit_screens']
    if type(screens) is not list or len(screens) != 12:
        raise ValueError('the complete twelve-unit screen is required')
    eligible = []
    for unit, screen in enumerate(screens):
        if (type(screen) is not dict or set(screen) != {'unit_id', 'semantic_ok', 'lexical_exclusion_ok', 'instructions'}
                or screen['unit_id'] != f'u{unit:02d}' or type(screen['semantic_ok']) is not bool
                or type(screen['lexical_exclusion_ok']) is not bool or type(screen['instructions']) is not list
                or len(screen['instructions']) != 2):
            raise ValueError('invalid screened unit')
        for instruction, group in enumerate(screen['instructions']):
            if (type(group) is not dict or set(group) != {'instruction_index', 'cases', 'source_comparisons',
                                                        'eligible', 'failure_reasons'}
                    or type(group['instruction_index']) is not int or group['instruction_index'] != instruction
                    or type(group['eligible']) is not bool or type(group['failure_reasons']) is not list
                    or len(group['failure_reasons']) > 128
                    or any(type(reason) is not str or not reason or len(reason.encode('utf8')) > 256
                           for reason in group['failure_reasons'])
                    or group['eligible'] != (not group['failure_reasons'])
                    or (group['eligible'] and (not screen['semantic_ok'] or not screen['lexical_exclusion_ok']))
                    or type(group['cases']) is not list or len(group['cases']) != 4
                    or type(group['source_comparisons']) is not list
                    or len(group['source_comparisons']) != len(sources)):
                raise ValueError('invalid screened instruction')
            for index, comparison in enumerate(group['source_comparisons']):
                if (type(comparison) is not dict or set(comparison) != {'source_index', 'comparisons', 'conflict'}
                        or type(comparison['source_index']) is not int or comparison['source_index'] != index
                        or type(comparison['comparisons']) is not int or comparison['comparisons'] < 0
                        or type(comparison['conflict']) is not bool
                        or (group['eligible'] and comparison['conflict'])):
                    raise ValueError('invalid source comparison')
            for side, case in zip(RECEIVERS, group['cases']):
                required = {'side', 'text', 'input_ids', 'attention_mask', 'special_tokens_mask',
                            'offset_mapping', 'adjective_index', 'subject_index'}
                if (type(case) is not dict or set(case) != required or case['side'] != side
                        or type(case['text']) is not str or len(case['text'].encode('utf8')) > 2048
                        or unicodedata.normalize('NFC', case['text']) != case['text']
                        or type(case['input_ids']) is not list or not 1 <= len(case['input_ids']) <= 4096
                        or any(type(v) is not int or not 0 <= v < vocab_size for v in case['input_ids'])
                        or any(type(case[name]) is not list or len(case[name]) != len(case['input_ids'])
                               for name in ('attention_mask', 'special_tokens_mask', 'offset_mapping'))
                        or any(type(v) is not int or v not in (0, 1) for name in
                               ('attention_mask', 'special_tokens_mask') for v in case[name])
                        or any(type(pair) is not list or len(pair) != 2
                               or any(type(v) is not int for v in pair)
                               or not 0 <= pair[0] <= pair[1] <= len(case['text'])
                               for pair in case['offset_mapping'])
                        or any(value is not None and (type(value) is not int or not 0 <= value < len(case['input_ids']))
                               for value in (case['adjective_index'], case['subject_index']))):
                    raise ValueError('invalid screened case')
        if (screen['semantic_ok'] and screen['lexical_exclusion_ok']
                and all(group['eligible'] for group in screen['instructions'])):
            eligible.append(screen['unit_id'])
    selected = eligible[:4]
    mapping = [{'unit_id': unit, 'opposite': dict(OPPOSITE), 'same': dict(SAME)} for unit in selected]
    if selection['selected_units'] != selected or selection['donor_mapping'] != mapping:
        raise ValueError('selected order or reciprocal donor mapping differs')
    if len(selected) < 4:
        return ()
    by_id = {screen['unit_id']: screen for screen in screens}
    groups = []
    for unit, ident in enumerate(selected):
        for instruction, group in enumerate(by_id[ident]['instructions']):
            groups.append({'unit': unit, 'instruction': instruction,
                           'cases': {case['side']: copy.deepcopy(case) for case in group['cases']}})
    checked = _validated_groups(groups, vocab_size=vocab_size, special_ids=special_ids)
    return tuple(checked[(unit, instruction)] for unit in range(4) for instruction in range(2))


def _encode_case(model, case, journal, width, check_runtime):
    numeric_before = runtime_numeric_probe()
    check_runtime()
    ids = torch.tensor([case['input_ids']], dtype=torch.int64)
    mask = torch.tensor([case['attention_mask']], dtype=torch.int64)
    ids_before, mask_before = ids.clone(), mask.clone()
    expected = (1, len(case['input_ids']), width)
    journal.begin_component({'kind': 'encoder', 'side': case['side'], 'input_ids': case['input_ids'],
                             'attention_mask': case['attention_mask'], 'numeric_before': numeric_before},
                            reserve_bytes=4 * math.prod(expected))
    output = state = ref = None
    try:
        with torch.inference_mode():
            check_runtime()
            output = model.get_encoder()(input_ids=ids, attention_mask=mask, return_dict=True)
        state = getattr(output, 'last_hidden_state', None)
        check_runtime()
        if type(state) is not torch.Tensor or tuple(state.shape) != expected:
            raise RuntimeError('encoder returned malformed state')
        raw = _raw(state)
        ref = journal.tensor(expected, raw)
        if not _finite(raw) or not _same_tensor(ids, ids_before) or not _same_tensor(mask, mask_before):
            raise RuntimeError('encoder returned nonfinite state or mutated its inputs')
        check_runtime()
        numeric_after = runtime_numeric_probe()
        journal.finish_component({'state': ref, 'numeric_after': numeric_after})
        return state.detach().clone(), ref
    except BaseException as primary:
        try:
            detail = {'error_type': type(primary).__name__, 'return_observed': output is not None, 'state': ref}
            if type(state) is torch.Tensor:
                detail['original_shape'] = list(state.shape)
                capacity = journal.remaining_component_bytes // 4
                if (ref is None and state.device.type == 'cpu' and state.dtype == torch.float32
                        and state.layout == torch.strided and state.is_contiguous() and capacity > 0):
                    count = min(state.numel(), capacity)
                    if count:
                        detail['retained_prefix'] = journal.tensor((count,), _raw(state.detach().view(-1)[:count]))
                        detail['truncated'] = count != state.numel()
            journal.record('component_rejected', detail)
        except BaseException as salvage_error:
            if isinstance(primary, Exception) and not isinstance(salvage_error, Exception):
                raise salvage_error from primary
        raise


def _patched(original, position, replacement):
    before = _raw(original)
    width = original.shape[2]
    if type(position) is not int or not 0 <= position < original.shape[1] or len(replacement) != width * 4:
        raise ValueError('invalid state patch')
    result = original.detach().clone()
    value = torch.frombuffer(bytearray(replacement), dtype=torch.float32).clone()
    with torch.no_grad():
        result[0, position].copy_(value)
    after = _raw(result)
    offset = position * width * 4
    if (after[:offset] != before[:offset] or after[offset:offset + width * 4] != replacement
            or after[offset + width * 4:] != before[offset + width * 4:]):
        raise RuntimeError('patch landing or an untouched encoder element changed')
    return result


def _same_output(journal, first, second):
    if first['generated_ids'] != second['generated_ids'] or len(first['all_logit_refs']) != len(second['all_logit_refs']):
        return False
    if [step['prefix'] for step in first['greedy']] != [step['prefix'] for step in second['greedy']]:
        return False
    for a, b in zip(first['all_logit_refs'], second['all_logit_refs']):
        if a['shape'] != b['shape'] or journal.read_tensor(a) != journal.read_tensor(b):
            return False
    return True


def _first_step_tv(journal, first, second):
    numeric_before = runtime_numeric_probe()
    left = _values(journal.read_tensor(first['greedy'][0]['logits']))
    right = _values(journal.read_tensor(second['greedy'][0]['logits']))
    if len(left) != len(right):
        raise ValueError('first-step vocabularies differ')
    ma, mb = max(left), max(right)
    aa, bb = [math.exp(x - ma) for x in left], [math.exp(x - mb) for x in right]
    za, zb = math.fsum(aa), math.fsum(bb)
    result = {'tv': .5 * math.fsum(abs(a / za - b / zb) for a, b in zip(aa, bb)),
            'clean_top_id': max(range(len(left)), key=left.__getitem__),
            'action_top_id': max(range(len(right)), key=right.__getitem__),
            'numeric_before': numeric_before}
    result['numeric_after'] = runtime_numeric_probe()
    return result


def _natural_geometry(original, donor, position):
    own, target = _values(_raw(original[0, position])), _values(_raw(donor[0, position]))
    direction = tuple(b - a for a, b in zip(own, target))
    dose, receiver_norm = _norm(direction), _norm(own)
    cosine = _cosine(direction, direction, dose, dose)
    return {'target_dose': dose, 'achieved_norm': dose, 'cosine': cosine,
            'receiver_norm': receiver_norm,
            'achieved_over_receiver_norm': None if receiver_norm == 0 else dose / receiver_norm,
            'ratio_reason': 'zero receiver norm' if receiver_norm == 0 else None}


def _record_comparison(journal, action, clean, measured):
    journal.record('output_comparison', {
        'action': action, 'clean_margin': clean['margin'], 'action_margin': measured['margin'],
        'clean_generated_ids': clean['generated_ids'], 'action_generated_ids': measured['generated_ids'],
        'clean_strict_correct': clean['strict_correct'], 'action_strict_correct': measured['strict_correct'],
        'clean_teacher': {name: {'score': item['score'], 'ranks': item['ranks']}
                          for name, item in clean['teacher'].items()},
        'action_teacher': {name: {'score': item['score'], 'ranks': item['ranks']}
                           for name, item in measured['teacher'].items()},
        **_first_step_tv(journal, clean, measured),
    })


def _persist_geometry(journal, plan, *, unit, instruction, width):
    landings = {}
    for key, item in plan['landings'].items():
        entry = dict(item)
        value = entry.pop('value')
        entry['value_ref'] = journal.tensor((width,), value)
        landings[key] = entry
    record = {key: value for key, value in plan.items() if key != 'landings'}
    record.update(unit=unit, instruction=instruction, landings=landings)
    journal.record('group_geometry', record)
    return record


def _run_core(model, tokenizer, groups, journal, *, width, vocab_size, test_only, custody_check):
    step = None
    try:
        if not test_only:
            require_protocol_allocation(journal.limits)
            journal._check()
            if journal.component_count != 0 or journal.active_component is not None:
                raise JournalError('the production journal already contains a component')
            if not callable(custody_check):
                raise ValueError('production execution requires a custody callback')
        if custody_check is not None:
            custody_check()
        selected = _validated_groups(groups, vocab_size=vocab_size, special_ids=set(tokenizer.all_special_ids))
        targets = _label_targets(tokenizer, vocab_size=vocab_size)
        evaluator = _EvaluationRunner(model, tokenizer, journal, width=width, vocab_size=vocab_size,
                                      test_only=test_only, custody_check=custody_check)
        evaluator._check_runtime()
        journal.record('runtime_observation', {'testing_only': test_only, 'settings': evaluator.runtime,
                                               'numeric': runtime_numeric_probe()})
        journal.record('label_targets', {'targets': targets, 'decoder_start_id': 0, 'eos_id': 1})
        schedule = fixed_schedule()
        states, baseline, output = {}, {}, {}
        geometry, support = {}, {}
        for step in schedule:
            kind, unit, instruction, side = step['kind'], step['unit'], step['instruction'], step['receiver']
            key = (unit, instruction, side)
            group = selected[(unit, instruction)]
            case = group['cases'][side]
            position = {'instruction': 0, 'subject': case['subject_index'], 'adjective': case['adjective_index']}
            if kind == 'clean':
                state, encoder_ref = _encode_case(model, case, journal, width, evaluator._check_runtime)
                states[key] = state
                action = dict(step, encoder_state=encoder_ref)
                baseline[key] = evaluator.evaluate(case, state, state.clone(), action, targets)
                continue
            if kind in ('passthrough', 'self', 'full_swap'):
                original = states[key]
                action = dict(step, source_state=baseline[key]['action']['encoder_state'])
                if kind == 'self':
                    site = step['site']
                    raw = _raw(original[0, position[site]])
                    after = _patched(original, position[site], raw)
                    action['position'] = position[site]
                elif kind == 'full_swap':
                    after = states[(unit, instruction, step['donor'])].clone()
                    action['donor_state'] = baseline[(unit, instruction, step['donor'])]['action']['encoder_state']
                else:
                    after = original.clone()
                measured = evaluator.evaluate(case, original, after, action, targets)
                expected_key = (unit, instruction, step['donor']) if kind == 'full_swap' else key
                if not _same_output(journal, measured, baseline[expected_key]):
                    raise RuntimeError('identity or full-state donor control did not match')
                journal.record('control_verified', {'action': step, 'expected': list(expected_key)})
                _record_comparison(journal, action, baseline[key], measured)
                continue
            if kind == 'repeat':
                original, encoder_ref = _encode_case(model, case, journal, width, evaluator._check_runtime)
                action = dict(step, encoder_state=encoder_ref)
                measured = evaluator.evaluate(case, original, original.clone(), action, targets)
                if not _same_output(journal, measured, baseline[key]):
                    raise RuntimeError('terminal repeat did not match clean output')
                journal.record('control_verified', {'action': step, 'expected': list(key)})
                _record_comparison(journal, action, baseline[key], measured)
                continue
            group_key = (unit, instruction)
            if group_key not in geometry:
                items = {receiver: states[(unit, instruction, receiver)] for receiver in RECEIVERS}
                geometry_before = runtime_numeric_probe()
                plan = plan_group(items, position)
                plan['numeric_before'] = geometry_before
                plan['numeric_after'] = runtime_numeric_probe()
                geometry[group_key] = _persist_geometry(journal, plan, unit=unit, instruction=instruction,
                                                        width=width)
                support[group_key] = all(baseline[(unit, instruction, receiver)]['strict_correct']
                    and (baseline[(unit, instruction, receiver)]['margin'] > 0 if receiver.startswith('P')
                         else baseline[(unit, instruction, receiver)]['margin'] < 0)
                    for receiver in RECEIVERS)
                journal.record('baseline_support', {'unit': unit, 'instruction': instruction,
                                                     'supported': support[group_key]})
            plan = geometry[group_key]
            if not support[group_key] or plan['status'] != 'matchable':
                reason = 'baseline-unsupported' if not support[group_key] else plan['status']
                journal.record('action_skipped', {'action': step, 'reason': reason})
                continue
            original = states[key]
            if kind == 'matched':
                landing = plan['landings'][f'{side}:{step["action"]}:{step["dose"]}']
                raw = journal.read_tensor(landing['value_ref'])
                after = _patched(original, position[landing['site']], raw)
                action = dict(step, **{name: value for name, value in landing.items()
                                       if name not in ('action', 'level', 'receiver')})
                action['position'] = position[landing['site']]
                action['source_state'] = baseline[key]['action']['encoder_state']
                action['donor_state'] = baseline[(unit, instruction, landing['donor'])]['action']['encoder_state']
            else:
                donor = states[(unit, instruction, OPPOSITE[side])]
                raw = _raw(donor[0, position['adjective']])
                after = _patched(original, position['adjective'], raw)
                natural_before = runtime_numeric_probe()
                natural_geometry = _natural_geometry(original, donor, position['adjective'])
                action = dict(step, site='adjective', position=position['adjective'],
                              source_state=baseline[key]['action']['encoder_state'],
                              donor_state=baseline[(unit, instruction, OPPOSITE[side])]['action']['encoder_state'],
                              natural_distance=plan['distances'][f'{side}:opposite_adjective'],
                              numeric_before=natural_before, numeric_after=runtime_numeric_probe(),
                              **natural_geometry)
            measured = evaluator.evaluate(case, original, after, action, targets)
            output[(unit, instruction, side, step.get('action', 'natural'), step.get('dose'))] = measured
            _record_comparison(journal, action, baseline[key], measured)
        groups_out = []
        for unit in range(4):
            pair = []
            for instruction in (0, 1):
                group_key = (unit, instruction)
                if not support[group_key]:
                    row = {'status': 'baseline-unsupported'}
                elif geometry[group_key]['status'] != 'matchable':
                    row = {'status': geometry[group_key]['status']}
                else:
                    clean = {r: baseline[(unit, instruction, r)]['margin'] for r in RECEIVERS}
                    high = {action: {r: output[(unit, instruction, r, action, 'high')]['margin']
                                     for r in RECEIVERS} for action in ACTIONS}
                    low = {action: {r: output[(unit, instruction, r, action, 'low')]['margin']
                                    for r in RECEIVERS} for action in ACTIONS}
                    analysis_before = runtime_numeric_probe()
                    row = group_decision(clean, high)
                    low_row = group_decision(clean, low)
                    row['low_dose'] = low_row
                    row['dose_comparison'] = {
                        action: {receiver: {'low': low_row['movements'][action][receiver],
                                            'high': row['movements'][action][receiver],
                                            'magnitude_change': (abs(row['movements'][action][receiver])
                                                                 - abs(low_row['movements'][action][receiver]))}
                                 for receiver in RECEIVERS}
                        for action in ACTIONS}
                    row['numeric_before'] = analysis_before
                    row['numeric_after'] = runtime_numeric_probe()
                pair.append(row)
                journal.record('group_outcome', {'unit': unit, 'instruction': instruction, **row})
            groups_out.append(pair)
        primary_before = runtime_numeric_probe()
        primary = primary_decision(groups_out, attempt_complete=True)
        journal.record('primary_numeric', {'before': primary_before, 'after': runtime_numeric_probe()})
        evaluator._check_runtime()
        result = {'groups': groups_out, 'primary': primary, 'testing_only': test_only}
        journal.finish('complete_candidate', result)
        return result
    except BaseException as primary:
        try:
            journal.finish('incomplete', {'error_type': type(primary).__name__, 'action': step,
                                          'testing_only': test_only})
        except BaseException as terminal_error:
            if isinstance(primary, Exception) and not isinstance(terminal_error, Exception):
                raise terminal_error from primary
        raise


def _run_fabricated(model, tokenizer, groups, journal, *, width, vocab_size, custody_check=None):
    """Run a test-only schedule over fabricated model and case objects."""
    return _run_core(model, tokenizer, groups, journal, width=width, vocab_size=vocab_size,
                     test_only=True, custody_check=custody_check)


def run_cpu_core(model, tokenizer, groups, journal, *, custody_check):
    """Execute a separately admitted CPU attempt, retaining provisional evidence.

    The owner must validate the committed selection, source, assets and custody,
    create the journal before model loading, and resolve process cleanup and the
    independent saved-output audit. The callback must raise on binding drift.
    """
    if type(journal) is not EvidenceJournal:
        raise ValueError('production execution requires a real EvidenceJournal')
    return _run_core(model, tokenizer, groups, journal, width=512, vocab_size=32128,
                     test_only=False, custody_check=custody_check)
