"""Offline CPU tests for the fixed pretrained smoke machinery."""
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch

if importlib.util.find_spec('transformers') is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipIf(GPT2Config is None, 'optional transformers package not installed')
class PretrainedSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_channels_match_uninstrumented_forward_and_reference(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.pretrained_smoke'))
        from looking_glass.pretrained_smoke import check_model
        torch.manual_seed(13)
        config = GPT2Config(vocab_size=7000, n_positions=16, n_ctx=16, n_embd=8,
                            n_layer=6, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0,
                            attn_pdrop=0.0, use_cache=False)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        result = check_model(model, ids, device='cpu')
        self.assertTrue(result['passed'])
        self.assertTrue(result['reference_logits_equal'])
        self.assertTrue(result['direct_gradient_equal'])
        self.assertEqual(len(result['baseline_logit_sha256']), 2)
        self.assertEqual(len(result['reference_value_sha256']), 18)
        for mode in ('direct', 'gradient', 'id'):
            cell = result['modes'][mode]
            self.assertTrue(cell['logits_equal_baseline'])
            self.assertTrue(cell['metadata_valid'])
            self.assertEqual(cell['event_count'], 18)
            self.assertEqual(cell['tensor_bytes'], 0 if mode == 'id' else 1152)
            self.assertEqual(len(cell['events']), 18)
            self.assertEqual(cell['values_equal_reference'], None if mode == 'id' else True)
            for event in cell['events']:
                self.assertEqual(event['value_sha256'] is None, mode == 'id')

    def test_subnormal_flush_is_not_reported_as_exact_capture(self):
        # Isolate the floating-point mode change from the parent test process.
        code = r'''
import sys
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from looking_glass.pretrained_smoke import (_cpu_subnormal_multiply_bits,
    _equal_float32_bits, check_model, SmokeFailure)
from looking_glass.probes import gradient_readout
torch.set_num_threads(1)
if not torch.set_flush_denormal(True) or _cpu_subnormal_multiply_bits() != 0:
    raise SystemExit(77)
x = torch.tensor([1, -2147483647], dtype=torch.int32).view(torch.float32)
y = gradient_readout(x)
assert not _equal_float32_bits(x, torch.zeros_like(x)), 'bit comparison missed subnormal'
if _equal_float32_bits(x, y):
    raise SystemExit(77)  # This backend did not flush the gradient readout.
config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
config._attn_implementation = 'eager'
model = GPT2LMHeadModel(config).eval()
def inject(value):
    bits = torch.zeros(value.shape, dtype=torch.int32, device=value.device)
    bits[..., 0] = 1
    return bits.view(torch.float32)
model.transformer.h[0].mlp.forward = inject
ids = torch.tensor([[464, 6766, 318, 4171, 13]])
try:
    check_model(model, ids, device='cpu')
except SmokeFailure as error:
    assert 'gradient saved value mismatch' in str(error), str(error)
else:
    raise AssertionError('smoke incorrectly accepted flushed gradient values')
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        if result.returncode == 77:
            self.skipTest('CPU denormal flushing is unsupported or differs on this host')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_output_directory_is_never_overwritten(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.pretrained_smoke'))
        from looking_glass.pretrained_smoke import main
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'already-there'
            destination.mkdir()
            marker = destination / 'sentinel'
            marker.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                main(['--device', 'cpu', '--cache-dir', tmp, '--output', str(destination)])
            self.assertEqual(marker.read_text(encoding='utf-8'), 'original')
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ['sentinel'])

    def test_equal_infinite_logits_are_not_a_successful_smoke(self):
        from looking_glass.pretrained_smoke import check_model, SmokeFailure
        class InfiniteHead(torch.nn.Module):
            def forward(self, hidden):
                return torch.full((*hidden.shape[:-1], 7000), float('inf'))
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        model.lm_head = InfiniteHead()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        with self.assertRaises(SmokeFailure):
            check_model(model, ids, device='cpu')

    def test_reference_phase_cannot_overlap_a_trace_owner(self):
        from looking_glass.pretrained_smoke import check_model
        from looking_glass.transformer_trace import GPT2TraceSession
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        with GPT2TraceSession(model, run_id='another-owner'):
            with self.assertRaises(RuntimeError):
                check_model(model, ids, device='cpu')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_reference_hook_registration_failure_quarantines_its_model(self):
        from looking_glass.pretrained_smoke import check_model
        from looking_glass.transformer_trace import GPT2TraceSession
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        with patch.object(model.transformer.h[0].attn, 'register_forward_hook',
                          side_effect=MemoryError('reference registration failed')):
            with self.assertRaises(MemoryError):
                check_model(model, ids, device='cpu')
        with self.assertRaises(RuntimeError):
            with GPT2TraceSession(model, run_id='another-owner'):
                pass

    def test_reference_entry_interruption_releases_lease(self):
        from looking_glass.pretrained_smoke import check_model
        from looking_glass.model_ownership import ModelLease
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        source, start = inspect.getsourcelines(check_model)
        acquire = next(i for i, line in enumerate(source) if 'lease.acquire()' in line)
        target = start + next(i for i, line in enumerate(source)
                              if i > acquire and 'lease.verify_tree()' in line)
        def cancel(frame, event, _arg):
            if frame.f_code is check_model.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('injected reference entry interruption')
            return cancel
        try:
            sys.settrace(cancel)
            with self.assertRaises(KeyboardInterrupt):
                check_model(model, ids, device='cpu')
        finally:
            sys.settrace(None)
        with ModelLease(model):
            pass

    def test_reference_interruption_before_handle_tracking_quarantines_model(self):
        from looking_glass.pretrained_smoke import check_model
        from looking_glass.model_ownership import ModelLease
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        source, start = inspect.getsourcelines(check_model)
        target = start + next(i for i, line in enumerate(source) if 'if not isinstance(handle,' in line)
        def cancel(frame, event, _arg):
            if frame.f_code is check_model.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('injected reference registration interruption')
            return cancel
        try:
            sys.settrace(cancel)
            with self.assertRaises(KeyboardInterrupt):
                check_model(model, ids, device='cpu')
        finally:
            sys.settrace(None)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()

    def test_reference_release_error_does_not_mask_forward_failure(self):
        from looking_glass.pretrained_smoke import check_model
        from looking_glass.model_ownership import ModelLease
        class PrimaryError(RuntimeError):
            pass
        class ReleaseError(RuntimeError):
            pass
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        with patch.object(model, 'forward', side_effect=PrimaryError('forward failed')):
            with patch.object(ModelLease, 'release', side_effect=ReleaseError('release failed')):
                with self.assertRaisesRegex(PrimaryError, 'forward failed') as caught:
                    check_model(model, ids, device='cpu')
        detail = '\n'.join(getattr(caught.exception, '__notes__', [])) or str(caught.exception.__cause__)
        self.assertIn('release failed', detail)

    def test_foreign_thread_reference_callback_cannot_contaminate_result(self):
        from looking_glass.pretrained_smoke import check_model, SmokeFailure
        config = GPT2Config(vocab_size=7000, n_positions=16, n_embd=8, n_layer=6, n_head=2)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[464, 6766, 318, 4171, 13]])
        owner = threading.get_ident()
        foreign_errors = []
        def inject(_module, _args, _output):
            if threading.get_ident() != owner:
                return
            def foreign():
                try:
                    model.transformer.h[0].mlp(torch.zeros(1, 5, 8))
                except BaseException as error:
                    foreign_errors.append(error)
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join(5)
        handle = model.transformer.h[0].mlp.register_forward_hook(inject)
        try:
            with self.assertRaises(SmokeFailure):
                check_model(model, ids, device='cpu')
            self.assertTrue(foreign_errors)
        finally:
            handle.remove()

    def test_failed_run_records_failure_but_never_success(self):
        from looking_glass import pretrained_smoke
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'new-run'
            with patch.object(pretrained_smoke, '_source_identity',
                              side_effect=pretrained_smoke.SmokeFailure('dirty source')):
                with self.assertRaisesRegex(pretrained_smoke.SmokeFailure, 'dirty source'):
                    pretrained_smoke.main(['--cache-dir', tmp, '--output', str(destination)])
            self.assertTrue((destination / 'started.json').is_file())
            failure = json.loads((destination / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['error_type'], 'SmokeFailure')
            self.assertIn('dirty source', failure['error'])
            self.assertFalse((destination / 'result.json').exists())

    def test_failure_record_write_error_does_not_hide_primary_error(self):
        from looking_glass import pretrained_smoke
        real_write = pretrained_smoke._write_json
        def fail_terminal(path, payload):
            if path.name == 'failure.json':
                raise OSError('failure record unavailable')
            real_write(path, payload)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'new-run'
            with patch.object(pretrained_smoke, '_source_identity',
                              side_effect=pretrained_smoke.SmokeFailure('primary source failure')):
                with patch.object(pretrained_smoke, '_write_json', side_effect=fail_terminal):
                    with self.assertRaisesRegex(pretrained_smoke.SmokeFailure, 'primary source failure') as caught:
                        pretrained_smoke.main(['--cache-dir', tmp, '--output', str(destination)])
            self.assertTrue((destination / 'started.json').exists())
            self.assertFalse((destination / 'result.json').exists())
            detail = '\n'.join(getattr(caught.exception, '__notes__', [])) or str(caught.exception.__cause__)
            self.assertIn('Failure record could not be written', detail)

    def test_source_blob_mismatch_is_rejected_before_a_recorded_run(self):
        from looking_glass import pretrained_smoke
        guard = Path(pretrained_smoke.__file__).parent / 'model_ownership.py'
        if not guard.exists():
            self.skipTest('shared ownership module not yet present in this checkout')
        real_read = Path.read_bytes
        def changed_source(path):
            if path.name == '__init__.py' and path.parent.name == 'looking_glass':
                return b'intentionally modified source'
            return real_read(path)
        with patch.object(pretrained_smoke, '_SOURCE_PATHS', ('looking_glass/__init__.py',)):
            with patch.object(Path, 'read_bytes', changed_source):
                with self.assertRaisesRegex(pretrained_smoke.SmokeFailure, 'committed blob'):
                    pretrained_smoke._source_identity()

    def test_existing_terminal_file_is_never_replaced(self):
        from looking_glass import pretrained_smoke
        with tempfile.TemporaryDirectory() as tmp:
            terminal = Path(tmp) / 'result.json'
            terminal.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                pretrained_smoke._write_json(terminal, {'overwritten': True})
            self.assertEqual(terminal.read_text(encoding='utf-8'), 'original')

    def test_start_record_failure_still_attempts_a_failure_record(self):
        from looking_glass import pretrained_smoke
        real_write = pretrained_smoke._write_json
        def fail_start(path, payload):
            if path.name == 'started.json':
                raise OSError('start record failed')
            real_write(path, payload)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'new-run'
            with patch.object(pretrained_smoke, '_write_json', side_effect=fail_start):
                with self.assertRaisesRegex(OSError, 'start record failed'):
                    pretrained_smoke.main(['--cache-dir', tmp, '--output', str(destination)])
            failure = json.loads((destination / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['error'], 'start record failed')

    def test_published_failure_redacts_cache_path_and_gpu_identifier(self):
        from looking_glass import pretrained_smoke
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'new-run'
            message = f"failed in {tmp} on GPU-11111111-2222-3333-4444-555555555555"
            with patch.object(pretrained_smoke, '_source_identity', side_effect=RuntimeError(message)):
                with self.assertRaises(RuntimeError):
                    pretrained_smoke.main(['--cache-dir', tmp, '--output', str(destination)])
            text = (destination / 'failure.json').read_text(encoding='utf-8')
            self.assertNotIn(tmp.replace('\\', '\\\\'), text)
            self.assertNotIn('GPU-11111111', text)
            self.assertIn('<cache>', text)
            self.assertIn('<gpu-uuid>', text)

    def test_redaction_covers_repr_escaped_windows_path_in_os_error(self):
        from looking_glass import pretrained_smoke
        with tempfile.TemporaryDirectory() as tmp:
            try:
                (Path(tmp) / 'missing.dat').open('rb')
            except FileNotFoundError as error:
                raw = str(error)
            clean = pretrained_smoke._redact_failure(raw, tmp)
            self.assertIn('<cache>', clean)
            self.assertNotIn(tmp.replace('\\', '\\\\'), clean)

    def test_cuda_cleanup_error_does_not_replace_primary_failure(self):
        from looking_glass import pretrained, pretrained_smoke
        class PrimaryError(RuntimeError):
            pass
        class CleanupError(RuntimeError):
            pass
        diagnostics = {}
        total = 8 * 1024 ** 3
        with ExitStack() as stack:
            stack.enter_context(patch.object(pretrained_smoke, '_source_identity', return_value={}))
            stack.enter_context(patch.object(pretrained, 'load_distilgpt2', side_effect=PrimaryError('load')))
            stack.enter_context(patch.object(torch, 'manual_seed'))
            for name, value in (('is_available', True), ('is_initialized', True),
                                ('mem_get_info', (total, total)), ('memory_allocated', 0),
                                ('memory_reserved', 0)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ('init', 'set_per_process_memory_fraction', 'empty_cache'):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, 'synchronize', side_effect=CleanupError('sync')))
            stack.enter_context(patch.object(pretrained_smoke.gc, 'collect', side_effect=CleanupError('collection')))
            with self.assertRaisesRegex(PrimaryError, 'load') as caught:
                pretrained_smoke.run_smoke('cuda', '.', lease_id='test-lease', _diagnostics=diagnostics)
        self.assertIn('CleanupError: sync', diagnostics['cleanup_errors'])
        self.assertIn('CleanupError: collection', diagnostics['cleanup_errors'])
        detail = '\n'.join(getattr(caught.exception, '__notes__', [])) or str(caught.exception.__cause__)
        self.assertIn('cleanup also failed', detail)
        self.assertEqual(diagnostics['observations']['memory'][0]['label'], 'admission_initial')

    def test_collection_error_does_not_replace_cpu_checkpoint_failure(self):
        from looking_glass import pretrained, pretrained_smoke
        class PrimaryError(RuntimeError):
            pass
        class CleanupError(RuntimeError):
            pass
        diagnostics = {}
        with patch.object(pretrained_smoke, '_source_identity', return_value={}), \
             patch.object(pretrained, 'load_distilgpt2', side_effect=PrimaryError('checkpoint failure')), \
             patch.object(pretrained_smoke.gc, 'collect', side_effect=CleanupError('collection failure')):
            with self.assertRaisesRegex(PrimaryError, 'checkpoint failure') as caught:
                pretrained_smoke.run_smoke('cpu', '.', _diagnostics=diagnostics)
        self.assertIn('CleanupError: collection failure', diagnostics['cleanup_errors'])
        detail = '\n'.join(getattr(caught.exception, '__notes__', [])) or str(caught.exception.__cause__)
        self.assertIn('collection failure', detail)

    def test_gpu_identity_requires_unambiguous_mapping_and_omits_uuid(self):
        from looking_glass import pretrained_smoke
        fake_result = SimpleNamespace(stdout='Example GPU, 123.45, 8192\n')
        with patch.object(pretrained_smoke.subprocess, 'run', return_value=fake_result):
            with patch.object(torch.cuda, 'device_count', return_value=1):
                with patch.object(torch.cuda, 'get_device_properties',
                                  return_value=SimpleNamespace(name='Example GPU')):
                    identity = pretrained_smoke._gpu_identity()
        self.assertEqual(identity['logical_device'], 'cuda:0')
        self.assertNotIn('uuid', identity)
        with patch.object(pretrained_smoke.subprocess, 'run', return_value=fake_result):
            with patch.object(torch.cuda, 'device_count', return_value=2):
                with self.assertRaises(pretrained_smoke.SmokeFailure):
                    pretrained_smoke._gpu_identity()


if __name__ == '__main__':
    unittest.main()
