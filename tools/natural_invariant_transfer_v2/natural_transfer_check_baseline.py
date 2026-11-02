import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import os

import natural_transfer_baseline as b


class BaselineRecordTests(unittest.TestCase):
    def good(self):
        names = ('started_sha256','events_sha256','terminal_sha256','supervisor_sha256','prelaunch_sha256',
                 'entry_result_sha256','model_admission_sha256','child_custody_sha256','reservation_sha256','latest_exposure_sha256',
                 'owner_entry_sha256','owner_executor_sha256','model_state_sha256','audit_sha256')
        prior = {name: format(i+1, '064x') for i,name in enumerate(names)}
        prior.update({'source_revision':'a'*40,'source_tree':'b'*40})
        receipt = {'schema':'agreement-invariant-v2-owner-exit-observation-v1', 'child_exit_code':0,
                   'observer_exit_code':0, 'reported_forwards':612, 'event_count':613,
                   'storage_accounting_complete':True, 'terminal_status':'complete_candidate',
                   'owner_pid':123, 'run_id':'toy-id', 'operator_exposure_sha256':prior['latest_exposure_sha256']}
        for name in ('started_sha256','events_sha256','terminal_sha256','supervisor_sha256','prelaunch_sha256',
                     'entry_result_sha256','model_admission_sha256','child_custody_sha256','source_revision','source_tree','reservation_sha256'):
            receipt[name] = prior[name]
        paths = [('attempt/started.json','started_sha256'),('attempt/events.jsonl','events_sha256'),
                 ('attempt/terminal.json','terminal_sha256'),('entry-result.json','entry_result_sha256'),
                 ('model-admission.json','model_admission_sha256'),('prelaunch.json','prelaunch_sha256')]
        files = [{'path':path, 'sha256':prior[key], 'bytes':5} for path,key in paths]
        receipt['run_files'] = len(files)
        custody = {'schema':'agreement-direction-supervised-child-v1','mode':'study','owner_pid':123,
                   'run_id':'toy-id','prelaunch_sha256':prior['prelaunch_sha256']}
        supervisor = {'schema':'agreement-direction-gated-observation-v1','clean':True,'validation_only':False,
                      'child':custody, 'child_sha256':prior['child_custody_sha256'],
                      'child_binding_matches':True,'post_source_check':'matched','post_errors':[],'recovery':[],
                      'supervision_error_type':None,'run_inventory_error':None, 'run_files':files,
                      'source_revision':prior['source_revision'],'source_tree':prior['source_tree'],
                      'prelaunch_sha256':prior['prelaunch_sha256'],
                      'completion_statuses':{'entry_result':'complete_candidate','terminal':'complete_candidate'},
                      'supervision_result':{'clean':True,'exit_code':0,'timed_out':False,'pid':123,'run_id':'toy-id'}}
        binding = {'prelaunch_sha256':prior['prelaunch_sha256'],'reservation_sha256':prior['reservation_sha256'],
                   'latest_exposure_sha256':prior['latest_exposure_sha256']}
        entry = {'status':'complete_candidate','model_disposed':True,'binding':binding,'owner_pid':123,'owner_run_id':'toy-id',
                 'worker':{'status':'complete_candidate','forwards':612,'schedule_sha256':b.SCHEDULE_SHA,
                           'model_state_sha256_before_after':prior['model_state_sha256']}}
        admission = {'status':'loaded_before_forward','binding':binding,'case_sha256':b.CASE_SHA,
                     'checkpoint':{'model_id':'distilbert/distilgpt2','model_revision':b.REVISION,
                                   'weights_sha256':b.WEIGHTS_SHA,'asset_sha256':{'tokenizer.json':b.TOKENIZER_SHA},
                                   'device':'cpu','dtype':'float32','attention_implementation':'eager','use_cache':False},
                     'runtime':{'threads':1,'interop_threads':1,'deterministic_algorithms':True,
                                'float32_matmul_precision':'highest'}}
        launch = {'schema':'distilgpt2-agreement-invariant-noun-prelaunch-v2','cases':{'sha256':b.CASE_SHA},
                  'protocol':{'sha256':b.PROTOCOL_SHA},
                  'owner':{'source_revision':prior['source_revision'],'source_tree':prior['source_tree'],
                           'entry_script_sha256':prior['owner_entry_sha256'],'executor_sha256':prior['owner_executor_sha256']},
                  'checkpoint':{'revision':b.REVISION,'weights_sha256':b.WEIGHTS_SHA,'tokenizer_sha256':b.TOKENIZER_SHA}}
        terminal = {'retained_tensor_bytes':42}
        audit = {'journal':{'status':'consistent_candidate','events':613,**{k:prior[k] for k in
                          ('started_sha256','events_sha256','terminal_sha256')}},
                 'dimensions':{'width':768,'vocabulary':50257},'schedule':{'sha256':b.SCHEDULE_SHA,'actions':[{}]*612},
                 'cases':[{}]*68,'source_tensor_bytes':42,
                 'public_inputs':{'case_sha256':b.CASE_SHA,'protocol_sha256':b.PROTOCOL_SHA,
                                  'owner_source_revision':prior['source_revision'],'external_binding':binding}}
        return receipt, supervisor, custody, entry, admission, launch, terminal, audit, prior

    def test_complete_bound_baseline_passes(self):
        b.verify_records(*self.good())

    def test_incomplete_mislabeled_or_wrong_source_rejects(self):
        for section, key, value in ((0,'observer_exit_code',1),(0,'child_exit_code',False),
                                    (0,'reported_forwards',301),(0,'storage_accounting_complete',False),
                                    (1,'clean',False),(1,'post_source_check','unknown'),
                                    (2,'mode','validation'),(3,'model_disposed',False),(4,'case_sha256','x'*64),
                                    (5,'schema','wrong'),(7,'cases',[])):
            args = list(self.good())
            args[section] = dict(args[section], **{key:value})
            with self.subTest(section=section,key=key), self.assertRaises(ValueError):
                b.verify_records(*args)
        args = list(self.good())
        args[7] = copy.deepcopy(args[7])
        args[7]['journal']['status'] = 'failed'
        with self.assertRaises(ValueError):
            b.verify_records(*args)
        args = list(self.good())
        args[1] = copy.deepcopy(args[1])
        args[1]['run_files'][0]['sha256'] = 'f'*64
        with self.assertRaises(ValueError):
            b.verify_records(*args)

    def test_audit_observation_requires_matching_exit_and_source_pins(self):
        prior = self.good()[-1]
        prior.update(receipt_sha256='e'*64, audit_observation_sha256='f'*64,
                     audit_observation_public_sha256='d'*64)
        expected = {'cases':b.CASE_SHA, 'entry':b.AUDIT_ENTRY_SHA, 'reader':b.AUDIT_READER_SHA,
                    'numeric':b.AUDIT_NUMERIC_SHA, 'events':prior['events_sha256'],
                    'started':prior['started_sha256'], 'terminal':prior['terminal_sha256'],
                    'supervisor':prior['supervisor_sha256'], 'owner_exit':prior['receipt_sha256'],
                    'u':'d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
                    'v1':'4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
                    'v2':'4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}
        options = {'--source-revision':prior['source_revision'], '--prelaunch-sha256':prior['prelaunch_sha256'],
                   '--reservation-sha256':prior['reservation_sha256'], '--latest-exposure-sha256':prior['latest_exposure_sha256'],
                   '--events-sha256':prior['events_sha256'], '--start-scope':'public prospective agreement invariant-noun successor v2',
                   '--attempt':'x', '--cases':'x', '--directions':'x', '--numeric-source':'x', '--output':'x'}
        command = ['python','-I','-S','-B'] + [v for pair in options.items() for v in pair]
        common = {'status':'complete','exit_code':0,'timed_out':False,
                  'stderr_sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                  'report_sha256':prior['audit_sha256'],'input_sha256_before':expected,'input_sha256_after':expected}
        original = dict(common, schema='agreement-invariant-v2-science-replay-observation-v1', command=command)
        public = dict(common, schema='agreement-invariant-v2-science-replay-public-observation-v1',
                      original_observation_sha256=prior['audit_observation_sha256'])
        with tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP']) as name:
            root = Path(name)
            (root/'tools').mkdir()
            entry = root/'tools'/'audit.py'
            entry.write_text('fake')
            numeric = root/'numeric.py'
            numeric.write_text('fake')
            private = {'prior_audit_observation':root/'obs.json','prior_audit_observation_public':root/'pub.json',
                       'prior_audit_entry':entry,'prior_audit_numeric':numeric,'prior_audit':root/'audit.json',
                       'prior_directions':root}
            def fake_pin(path, expected_sha):
                if Path(path) == private['prior_audit_observation']:
                    return json.dumps(original).encode()
                if Path(path) == private['prior_audit_observation_public']:
                    return json.dumps(public).encode()
                return b'fake-source'
            with mock.patch.object(b, 'pinned', side_effect=fake_pin):
                b.verify_audit_observation(private, prior)
                for target,key,value in ((original,'exit_code',1),(original,'timed_out',True),
                                         (original,'report_sha256','0'*64),(public,'original_observation_sha256','0'*64)):
                    old=target[key]
                    target[key]=value
                    with self.assertRaises(ValueError):
                        b.verify_audit_observation(private, prior)
                    target[key]=old
                altered=copy.deepcopy(original['input_sha256_after'])
                altered['events']='0'*64
                original['input_sha256_after']=altered
                with self.assertRaises(ValueError):
                    b.verify_audit_observation(private, prior)

    def test_pinned_source_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP']) as name:
            source=Path(name)/'source.py'
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                b.pinned(source, '0'*64)

if __name__ == '__main__':
    unittest.main()
