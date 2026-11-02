"""Offline curated download controls; no model or dependency downloads."""
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from looking_glass import app_downloads as dl


def spec(data=b'{"model_type":"llama"}'):
    return dl.ModelSpec(
        id='tiny-test', repo='example/tiny', revision='a'*40,
        title='Synthetic test only', license='apache-2.0', family='llama',
        float32_element_count=1000, hidden_size=16, layers=2, attention_heads=2,
        intermediate_size=32, vocab_size=100, max_context=128,
        files=(dl.FileSpec('config.json', len(data), hashlib.sha256(data).hexdigest()),),
    )


def wait(manager):
    deadline = time.monotonic()+5
    while manager.status()['state'] in ('running','cancelling'):
        if time.monotonic()>deadline: raise AssertionError('download worker did not terminate')
        time.sleep(.005)
    return manager.status()


class DownloadTests(unittest.TestCase):
    def test_verified_unique_publication_and_no_overwrite(self):
        data=b'{"model_type":"llama"}'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'models'
            calls=[]
            def transport(url): calls.append(url); return io.BytesIO(data)
            manager=dl.DownloadManager(root, (spec(),), transport=transport)
            self.assertFalse(root.exists())
            self.assertEqual(manager.status()['state'],'idle')
            first=manager.start('tiny-test'); result=wait(manager)
            self.assertEqual(result['state'],'complete',result)
            path=Path(result['path']); self.assertEqual((path/'config.json').read_bytes(),data)
            self.assertEqual(result['received_bytes'],len(data))
            self.assertIn('/resolve/'+'a'*40+'/config.json',calls[0])
            manager.start('tiny-test'); second=wait(manager)
            self.assertEqual(second['state'],'complete')
            self.assertNotEqual(second['path'],str(path))
            self.assertEqual((path/'config.json').read_bytes(),data)
            self.assertNotEqual(first['job_id'],second['job_id'])

    def test_corrupt_short_overlong_streams_never_publish(self):
        for data in (b'x',b'x'*100,b'{"model_type":"wrong"}'):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'models'; keep=Path(temp)/'unrelated'; keep.write_text('keep')
                manager=dl.DownloadManager(root,(spec(),),transport=lambda url:io.BytesIO(data))
                manager.start('tiny-test'); result=wait(manager)
                self.assertEqual(result['state'],'failed'); self.assertIsNone(result['path'])
                self.assertEqual(list(root.iterdir()),[])
                self.assertEqual(keep.read_text(),'keep')

    def test_cancel_and_single_job_and_stale_cancel(self):
        entered=threading.Event(); release=threading.Event()
        class Slow(io.BytesIO):
            def read(self,n=-1): entered.set(); release.wait(2); return super().read(n)
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=lambda url:Slow(b'{}'))
            job=manager.start('tiny-test'); self.assertTrue(entered.wait(2))
            with self.assertRaises(ValueError): manager.start('tiny-test')
            with self.assertRaises(ValueError): manager.cancel('stale')
            manager.cancel(job['job_id']); release.set()
            self.assertEqual(wait(manager)['state'],'cancelled')
            self.assertEqual(list((Path(temp)/'models').iterdir()),[])

    def test_unlisted_id_and_symlink_root_no_network(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'target'; target.mkdir(); root=Path(temp)/'models'; root.symlink_to(target, target_is_directory=True)
            with patch.object(dl,'open_official') as network:
                manager=dl.DownloadManager(root,(spec(),))
                with self.assertRaises(ValueError): manager.start('../arbitrary')
                with self.assertRaises(ValueError): manager.start('tiny-test')
                network.assert_not_called()
            self.assertEqual(list(target.iterdir()),[])

    def test_file_manifest_rejects_executable_paths_and_invalid_pins(self):
        for name in ('../x','a/b','model.bin','model.py','x.pkl','/tmp/x','x\\y'):
            with self.subTest(name=name), self.assertRaises(ValueError): dl.FileSpec(name,1,'a'*64)
        for size in (True,0,-1):
            with self.assertRaises(ValueError): dl.FileSpec('config.json',size,'a'*64)
        with self.assertRaises(ValueError): dl.FileSpec('config.json',1,'missing')

    def test_url_redirects_restricted_to_official_https(self):
        for url in ('http://huggingface.co/a','https://evil.test/a','https://huggingface.co.evil.test/a',
                    'https://u:p@huggingface.co/a','https://huggingface.co:444/a','file:///tmp/x'):
            with self.subTest(url=url), self.assertRaises(ValueError): dl.validate_download_url(url)
        for host in ('huggingface.co','cdn-lfs.huggingface.co','cas-bridge.xethub.hf.co'):
            self.assertEqual(dl.validate_download_url('https://'+host+'/a'),'https://'+host+'/a')

    def test_fit_unknown_cpu_and_known_too_large_filter(self):
        item=spec()
        cpu=dl.catalog_view((item,),device='cpu',free_bytes=0,total_bytes=0)
        self.assertEqual(len(cpu['models']),1)
        self.assertEqual(cpu['models'][0]['fit']['status'],'cpu')
        unknown=dl.catalog_view((item,),device='cuda:0',free_bytes=None,total_bytes=None)
        self.assertEqual(unknown['models'][0]['fit']['status'],'unknown')
        small=dl.catalog_view((item,),device='cuda:0',free_bytes=100,total_bytes=100)
        self.assertEqual(small['models'],[]); self.assertEqual(small['hidden_too_large'],1)
        shown=dl.catalog_view((item,),device='cuda:0',free_bytes=100,total_bytes=100,show_all=True)
        self.assertEqual(shown['models'][0]['fit']['status'],'too_large')
        enough=dl.catalog_view((item,),device='cuda:0',free_bytes=2**30,total_bytes=2**30)
        self.assertEqual(enough['models'][0]['fit']['status'],'likely_fits')
        self.assertIn('not',enough['models'][0]['fit']['note'].lower())

class AdditionalControls(unittest.TestCase):
    def test_app_policy_limits_precede_advisory_gpu_fit(self):
        item=spec()
        row=dl.catalog_view((item,),device='cuda:0',free_bytes=2**30,total_bytes=2**30,
                            max_model_bytes=100,show_all=True)['models'][0]
        self.assertEqual(row['fit']['status'],'too_large')
        self.assertEqual(row['fit']['reason'],'app_model_limit')
        row=dl.catalog_view((item,),device='cuda:0',free_bytes=2**30,total_bytes=2**30,
                            max_vram_bytes=100,show_all=True)['models'][0]
        self.assertEqual(row['fit']['status'],'too_large')
        self.assertEqual(row['fit']['available_bytes'],100)
        with self.assertRaises(ValueError): dl.catalog_view((item,),max_model_bytes=True)

    def test_git_blob_digest_and_error_messages_do_not_leak_transport_urls(self):
        from dataclasses import replace
        data=b'{"model_type":"llama"}'
        blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        item=replace(spec(),files=(dl.FileSpec('config.json',len(data),blob,'git_blob_sha1'),))
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(item,),transport=lambda url:io.BytesIO(data))
            manager.start(item.id); self.assertEqual(wait(manager)['state'],'complete')
            def broken(url): raise ValueError('https://cdn-lfs.huggingface.co/file?Secret=token')
            manager=dl.DownloadManager(Path(temp)/'other',(item,),transport=broken)
            manager.start(item.id); result=wait(manager)
            self.assertEqual(result['state'],'failed')
            self.assertNotIn('Secret',result['error']); self.assertNotIn('https://',result['error'])

    def test_no_space_never_calls_transport_and_preserves_siblings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'models'; root.mkdir(); sibling=root/'keep'; sibling.mkdir()
            with patch.object(dl.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()), patch.object(dl,'open_official') as transport:
                manager=dl.DownloadManager(root,(spec(),)); manager.start('tiny-test')
                self.assertEqual(wait(manager)['state'],'failed'); transport.assert_not_called()
            self.assertEqual(list(root.iterdir()),[sibling])

    def test_connection_timeout_is_terminal_and_response_is_closed(self):
        class Broken(io.BytesIO):
            def read(self,n=-1): raise TimeoutError('private URL')
        stream=Broken()
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=lambda url:stream)
            manager.start('tiny-test'); result=wait(manager)
            self.assertEqual(result['state'],'failed'); self.assertTrue(stream.closed)
            self.assertNotIn('private',result['error'])
            self.assertEqual(list((Path(temp)/'models').iterdir()),[])

    def test_cancel_after_completion_does_not_remove_snapshot(self):
        data=b'{"model_type":"llama"}'
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=lambda url:io.BytesIO(data))
            job=manager.start('tiny-test'); done=wait(manager)
            self.assertEqual(manager.cancel(job['job_id'])['state'],'complete')
            self.assertEqual((Path(done['path'])/'config.json').read_bytes(),data)

    def test_redirect_handler_rejects_before_following_and_has_no_proxy_auth(self):
        from urllib.request import Request
        handler=dl._OfficialRedirect()
        with self.assertRaises(ValueError):
            handler.redirect_request(Request('https://huggingface.co/a'),None,302,'Found',{},'https://evil.test/secret')
        with patch.object(dl,'build_opener') as build:
            response=build.return_value.open.return_value
            response.status=200; response.headers={}; response.geturl.return_value='https://huggingface.co/a'
            self.assertIs(dl.open_official('https://huggingface.co/a')._response,response)
            args=build.call_args.args
            self.assertEqual(args[0].proxies,{})
            request=build.return_value.open.call_args.args[0]
            self.assertNotIn('Authorization',request.headers)
            self.assertEqual(build.return_value.open.call_args.kwargs['timeout'],dl.NETWORK_TIMEOUT)

class FinalizationControls(unittest.TestCase):
    def test_cancel_after_last_read_before_publication(self):
        data=b'{"model_type":"llama"}'
        final_read=threading.Event(); release=threading.Event()
        class AtEnd(io.BytesIO):
            def read(self,n=-1):
                chunk=super().read(n)
                if not chunk: final_read.set(); release.wait(2)
                return chunk
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=lambda url:AtEnd(data))
            job=manager.start('tiny-test'); self.assertTrue(final_read.wait(2))
            manager.cancel(job['job_id']); release.set()
            self.assertEqual(wait(manager)['state'],'cancelled')
            self.assertEqual(list((Path(temp)/'models').iterdir()),[])

    def test_cleanup_failure_reported_not_success_and_only_owned_path(self):
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=lambda url:io.BytesIO(b'bad'))
            real=dl.shutil.rmtree
            with patch.object(dl.shutil,'rmtree',side_effect=OSError('locked')) as removal:
                manager.start('tiny-test'); status=wait(manager)
                self.assertEqual(status['state'],'failed')
                self.assertIn('could not be cleaned',status['error'])
                owned=removal.call_args.args[0]
                self.assertEqual(owned.parent,Path(temp)/'models')
                self.assertTrue(owned.name.startswith('tiny-test-'))
            real(owned)

    def test_deadline_and_http_encoding_limits(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dl,'JOB_TIMEOUT',-1), patch.object(dl,'open_official') as network:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),)); manager.start('tiny-test')
            self.assertEqual(wait(manager)['state'],'failed'); network.assert_not_called()
        with patch.object(dl,'build_opener') as build:
            response=build.return_value.open.return_value
            response.status=200; response.geturl.return_value='https://huggingface.co/a'
            response.headers={'Content-Encoding':'gzip'}
            with self.assertRaises(ValueError): dl.open_official('https://huggingface.co/a')
            response.close.assert_called_once()

    def test_streaming_http_read_uses_one_network_read_not_fill_entire_chunk(self):
        with patch.object(dl,'build_opener') as build:
            response=build.return_value.open.return_value
            response.status=200; response.geturl.return_value='https://huggingface.co/a'; response.headers={}
            response.read1.return_value=b'part'
            with dl.open_official('https://huggingface.co/a') as stream:
                self.assertEqual(stream.read(dl.CHUNK_BYTES),b'part')
            response.read1.assert_called_once_with(dl.CHUNK_BYTES)
            response.read.assert_not_called(); response.close.assert_called_once()

class CatalogContract(unittest.TestCase):
    def test_pinned_model_card_is_allowed_but_arbitrary_markdown_is_not(self):
        self.assertEqual(dl.FileSpec('README.md',1,'a'*64).name,'README.md')
        with self.assertRaises(ValueError): dl.FileSpec('arbitrary.md',1,'a'*64)

    def test_geometry_and_float32_fit_do_not_depend_on_storage_precision(self):
        from dataclasses import replace
        plain=spec(); grouped=replace(plain,head_dim=64,key_value_heads=1)
        a=dl.estimate_fit(plain); b=dl.estimate_fit(grouped)
        self.assertEqual(a['estimated_float32_weight_bytes'],4000)
        self.assertGreater(b['estimated_workspace_bytes'],a['estimated_workspace_bytes'])
        with self.assertRaises(ValueError): replace(plain,family='vit')
        with self.assertRaises(ValueError): replace(plain,key_value_heads=3)
        with self.assertRaises(ValueError): replace(plain,revision='main')
        with self.assertRaises(ValueError): replace(plain,files=(plain.files[0],plain.files[0]))

    def test_invalid_or_partial_device_information_stays_unknown(self):
        for free,total in ((None,100),(100,None),(-1,100),(101,100),(True,100),(0,0)):
            with self.subTest(free=free,total=total):
                row=dl.catalog_view((spec(),),device='cuda:0',free_bytes=free,total_bytes=total)['models'][0]
                self.assertEqual(row['fit']['status'],'unknown')

class TransportErrors(unittest.TestCase):
    def test_http_access_denial_is_actionable_without_credentials_or_signed_url(self):
        from urllib.error import HTTPError
        def denied(url): raise HTTPError('https://huggingface.co/private?secret=token',403,'denied',{},None)
        with tempfile.TemporaryDirectory() as temp:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),),transport=denied)
            manager.start('tiny-test'); result=wait(manager)
            self.assertEqual(result['state'],'failed')
            self.assertIn('403',result['error']); self.assertIn('credentials',result['error'])
            self.assertNotIn('secret',result['error'])

class ResponseCleanup(unittest.TestCase):
    def test_http_error_response_closed_by_official_transport(self):
        from urllib.error import HTTPError
        body=io.BytesIO(b'forbidden')
        error=HTTPError('https://huggingface.co/a',403,'Forbidden',{},body)
        with patch.object(dl,'build_opener') as build:
            build.return_value.open.side_effect=error
            with self.assertRaises(HTTPError): dl.open_official('https://huggingface.co/a')
        self.assertTrue(body.closed)

class ContextLimits(unittest.TestCase):
    def test_requested_context_is_not_silently_shortened_to_fit(self):
        for device in ('cpu','cuda:0'):
            with self.subTest(device=device):
                rows=dl.catalog_view((spec(),),device=device,free_bytes=2**30,total_bytes=2**30,
                                     context_tokens=256,show_all=True)
                fit=rows['models'][0]['fit']
                self.assertEqual(fit['status'],'too_large')
                self.assertEqual(fit['reason'],'context_limit')
                self.assertEqual(fit['requested_context_tokens'],256)
                self.assertEqual(fit['model_context_limit'],128)
                self.assertEqual(dl.catalog_view((spec(),),device=device,context_tokens=256)['models'],[])

class VerifiedCatalog(unittest.TestCase):
    def test_public_catalog_uses_exact_reviewed_snapshot_pins(self):
        entries={x.id:x for x in dl.CATALOG}
        self.assertEqual(set(entries),{'smollm2-135m','distilgpt2'})
        smol=entries['smollm2-135m']; distil=entries['distilgpt2']
        self.assertEqual(smol.revision,'93efa2f097d58c2a74874c7e644dbc9b0cee75a2')
        self.assertEqual(distil.revision,'2290a62682d06624634c1f46a6ad5be0f47f38aa')
        self.assertEqual(smol.download_bytes,272443805)
        self.assertEqual(len(smol.files),9); self.assertEqual(len(distil.files),8)
        expected={'smollm2-135m':'80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1',
                  'distilgpt2':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'}
        for item in entries.values():
            weight=next(f for f in item.files if f.name=='model.safetensors')
            self.assertEqual(weight.digest,expected[item.id]); self.assertEqual(weight.digest_kind,'sha256')
            self.assertEqual(item.license,'apache-2.0')
            self.assertTrue({'config.json','tokenizer.json','tokenizer_config.json','README.md'}<={f.name for f in item.files})
        view=dl.catalog_view(device='cpu')['models']
        self.assertEqual([x['id'] for x in view],['distilgpt2','smollm2-135m'])
        row=view[0]
        self.assertEqual(row['validation'],'metadata_only')
        self.assertEqual(row['fit']['estimated_float32_weight_bytes'],88204032*4)
        self.assertIn(distil.revision,row['license_url'])
        self.assertEqual(view[1]['validation'],'prior_cpu_verified')

    def test_stale_requested_revision_fails_without_network_or_writes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dl,'open_official') as network:
            manager=dl.DownloadManager(Path(temp)/'models',(spec(),))
            with self.assertRaises(ValueError): manager.start('tiny-test',expected_revision='b'*40)
            self.assertEqual(manager.status()['state'],'idle')
            self.assertFalse((Path(temp)/'models').exists()); network.assert_not_called()
