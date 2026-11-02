"""Explicit, curated public model downloads. No runtime/model ownership operations.

The managed directory is trusted local application storage, not a sandbox against
another process running as the same OS user. Catalog pins are developer-reviewed,
not supplied by an HTTP request. Fit estimates are advisory FP32 planning only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import tempfile
import threading
import time
from urllib.parse import urlsplit, quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler, HTTPSHandler
import uuid

MiB = 1024**2
CHUNK_BYTES = 256*1024
NETWORK_TIMEOUT = 20
JOB_TIMEOUT = 4*60*60
MAX_DOWNLOAD_BYTES = 32*1024**3
OFFICIAL_HOSTS = frozenset({
    'huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.huggingface.co',
    'cdn-lfs-eu-1.huggingface.co', 'cas-bridge.xethub.hf.co',
})
_ALLOWED_FILES = frozenset({
    'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'vocab.txt', 'tokenizer.model', 'spiece.model', 'model.safetensors',
    'model.safetensors.index.json', 'preprocessor_config.json', 'LICENSE', 'LICENSE.txt', 'README.md',
})


def _positive(value, name, maximum=None):
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a positive bounded integer')
    return value


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    digest: str
    digest_kind: str = 'sha256'

    def __post_init__(self):
        if not isinstance(self.name, str) or not (self.name in _ALLOWED_FILES or re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', self.name)):
            raise ValueError('File is not an allowed model input filename')
        _positive(self.size, 'File size', MAX_DOWNLOAD_BYTES)
        digits = {'sha256': 64, 'git_blob_sha1': 40}.get(self.digest_kind)
        if digits is None or not isinstance(self.digest, str) or not re.fullmatch('[0-9a-f]{'+str(digits)+'}',self.digest):
            raise ValueError('File requires an exact SHA256 or Git blob SHA1 pin')


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    revision: str
    title: str
    license: str
    family: str
    # Verified serialized tensor element count; may include non-learned buffers.
    float32_element_count: int
    hidden_size: int
    layers: int
    attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_context: int
    files: tuple[FileSpec, ...]
    head_dim: int | None = None
    key_value_heads: int | None = None
    validation: str = "metadata_only"

    def __post_init__(self):
        if not isinstance(self.id,str) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}',self.id):
            raise ValueError('Invalid catalog ID')
        if not isinstance(self.repo,str) or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+',self.repo):
            raise ValueError('Invalid fixed public repository')
        if not isinstance(self.revision,str) or not re.fullmatch('[0-9a-f]{40}',self.revision):
            raise ValueError('An immutable 40-character revision is required')
        if self.family not in ('gpt2','llama'):
            raise ValueError('Unsupported model family')
        for name in ('float32_element_count','hidden_size','layers','attention_heads','intermediate_size','vocab_size','max_context'):
            _positive(getattr(self,name),name)
        if self.validation not in ('metadata_only','prior_cpu_verified'): raise ValueError('Unknown validation scope')
        if self.head_dim is not None: _positive(self.head_dim, 'head_dim')
        if self.key_value_heads is not None:
            _positive(self.key_value_heads, 'key_value_heads')
            if self.attention_heads % self.key_value_heads: raise ValueError('Invalid GQA head geometry')
        if self.head_dim is None and self.hidden_size % self.attention_heads:
            raise ValueError('Explicit head_dim required for nonstandard projection width')
        if type(self.files) is not tuple or not 1 <= len(self.files) <= 128 or any(type(f) is not FileSpec for f in self.files):
            raise ValueError('Catalog needs a bounded immutable file manifest')
        if len({f.name for f in self.files}) != len(self.files):
            raise ValueError('Duplicate manifest filename')
        if self.download_bytes > MAX_DOWNLOAD_BYTES:
            raise ValueError('Snapshot exceeds download limit')
        for value in (self.title,self.license):
            if not isinstance(value,str) or not 1 <= len(value) <= 200:
                raise ValueError('Catalog requires bounded title and license')

    @property
    def download_bytes(self):
        return sum(f.size for f in self.files)


# Public/ungated Apache-2.0 registry metadata checked 2026-11-01. Immutable
# per-file pins only; no live registry lookup occurs during application startup.
# SmolLM2 pins also match results/local-app-smollm2-135m-cpu-20261031/record.json.
# DistilGPT2 is metadata-verified, not a new runtime or GPU validation claim.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id='smollm2-135m', repo='HuggingFaceTB/SmolLM2-135M',
        revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
        title='SmolLM2 135M', license='apache-2.0', family='llama',
        float32_element_count=134515008, hidden_size=576, layers=30, attention_heads=9,
        key_value_heads=3, head_dim=64, intermediate_size=1536, vocab_size=49152,
        max_context=8192, validation='prior_cpu_verified', files=(
            FileSpec('README.md', 6340, 'd1ba68cae64a89b6b434b11526e6e2271ee5ffd2c914ec35ed515f9d84c6085c'),
            FileSpec('config.json', 704, '1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843'),
            FileSpec('generation_config.json', 111, '2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059'),
            FileSpec('merges.txt', 466391, '0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510'),
            FileSpec('model.safetensors', 269060552, '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'),
            FileSpec('special_tokens_map.json', 831, 'e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3'),
            FileSpec('tokenizer.json', 2104556, '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'),
            FileSpec('tokenizer_config.json', 3658, '4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424'),
            FileSpec('vocab.json', 800662, '82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79'),
        ),
    ),
    ModelSpec(
        id='distilgpt2', repo='distilbert/distilgpt2',
        revision='2290a62682d06624634c1f46a6ad5be0f47f38aa',
        title='DistilGPT2', license='apache-2.0', family='gpt2',
        float32_element_count=88204032, hidden_size=768, layers=6, attention_heads=12,
        head_dim=64, intermediate_size=3072, vocab_size=50257, max_context=1024,
        # Registry FP32 element total includes legacy buffers; conservatively
        # budget it too, without calling those buffers learned parameters.
        validation='metadata_only', files=(
            FileSpec('README.md', 10992, '86f7c9ae4422de194e6b6104b0acc9c8b9d2cd0a', 'git_blob_sha1'),
            FileSpec('config.json', 762, 'fed02aafb76b46aa5da200966f9e42262e758023', 'git_blob_sha1'),
            FileSpec('generation_config.json', 124, '057e43033439e068b325f32af95dde9efc9552ae', 'git_blob_sha1'),
            FileSpec('merges.txt', 456318, '226b0752cac7789c48f0cb3ec53eda48b7be36cc', 'git_blob_sha1'),
            FileSpec('model.safetensors', 352824413, 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b', 'sha256'),
            FileSpec('tokenizer.json', 1355256, '4b988bccc9dc5adacd403c00b4704976196548f8', 'git_blob_sha1'),
            FileSpec('tokenizer_config.json', 26, 'be4d21d94f3b4687e5a54d84bf6ab46ed0f8defd', 'git_blob_sha1'),
            FileSpec('vocab.json', 1042301, '1f1d9aaca301414e7f6c9396df506798ff4eb9a6', 'git_blob_sha1'),
        ),
    ),
)


def estimate_fit(spec, *, device='cpu', free_bytes=None, total_bytes=None, context_tokens=128, max_model_bytes=None, max_vram_bytes=None):
    """Estimate FP32 weights plus eager work/capture/headroom, not peak VRAM."""
    _positive(context_tokens,'Context',4096)
    for name, limit in (('max_model_bytes',max_model_bytes),('max_vram_bytes',max_vram_bytes)):
        if limit is not None: _positive(limit,name)
    n=min(context_tokens,spec.max_context)
    weights=spec.float32_element_count*4
    # One full eager forward, all three boundaries; cross-step traces offload to RAM.
    captures=3*spec.layers*n*spec.hidden_size*4
    projection_width=spec.attention_heads*(spec.head_dim or spec.hidden_size//spec.attention_heads)
    workspace=4*(8*n*spec.hidden_size + 6*n*projection_width + 4*n*spec.intermediate_size +
                 2*spec.attention_heads*n*n + 2*n*spec.vocab_size)
    headroom=max(256*MiB,math.ceil((weights+captures+workspace)*.20))
    required=weights+captures+workspace+headroom
    valid=lambda v:type(v) is int and v>=0
    reason=None
    if device=='cpu': status='cpu'; available=None
    elif not isinstance(device,str) or not re.fullmatch(r'cuda:\d+',device):
        raise ValueError('Select cpu or an exact cuda:N device for fit estimates')
    elif not valid(free_bytes) or not valid(total_bytes) or total_bytes<=0 or free_bytes>total_bytes:
        status='unknown'; available=None
    else:
        available=min(free_bytes,total_bytes,max_vram_bytes if max_vram_bytes is not None else total_bytes)
        status='too_large' if required>available else 'likely_fits'
        if status=='too_large': reason='device_memory_or_policy_limit'
    if max_model_bytes is not None and weights>max_model_bytes:
        status='too_large'; reason='app_model_limit'
    if context_tokens>spec.max_context:
        status='too_large'; reason='context_limit'
    return dict(status=status,reason=reason,device=device,available_bytes=available,
                requested_context_tokens=context_tokens,model_context_limit=spec.max_context,
                max_model_bytes=max_model_bytes,max_vram_bytes=max_vram_bytes,
                estimated_float32_weight_bytes=weights,estimated_capture_bytes=captures,
                estimated_workspace_bytes=workspace,headroom_bytes=headroom,
                estimated_required_bytes=required,context_tokens=n,
                note='Advisory FP32 estimate, not a fit or peak-memory guarantee. Free memory can change; CPU RAM and longer contexts need separate planning.')


def catalog_view(catalog=CATALOG, *, device='cpu',free_bytes=None,total_bytes=None,context_tokens=128,show_all=False,max_model_bytes=None,max_vram_bytes=None):
    if type(show_all) is not bool: raise ValueError('show_all must be boolean')
    rows=[]; hidden=0
    for spec in catalog:
        fit=estimate_fit(spec,device=device,free_bytes=free_bytes,total_bytes=total_bytes,context_tokens=context_tokens,max_model_bytes=max_model_bytes,max_vram_bytes=max_vram_bytes)
        if fit['status']=='too_large' and not show_all: hidden+=1; continue
        rows.append(dict(id=spec.id,title=spec.title,repo=spec.repo,revision=spec.revision,
                         license=spec.license,family=spec.family,download_bytes=spec.download_bytes,
                         license_url=f'https://huggingface.co/{spec.repo}/blob/{spec.revision}/README.md',
                         validation=spec.validation,
                         validation_note=('Previously checked with this app on CPU; GPU execution is not validated.' if spec.validation=='prior_cpu_verified' else 'Public metadata and file pins checked; inspect and load explicitly to validate local compatibility.'),
                         source_url=f'https://huggingface.co/{spec.repo}/tree/{spec.revision}',fit=fit))
    rows.sort(key=lambda row:(row['fit']['status']=='too_large',row['fit']['estimated_required_bytes'],row['id']))
    return dict(models=rows,hidden_too_large=hidden,device=device,
                note='Only curated immutable public snapshots; downloading is explicit and does not load a model.')


def validate_download_url(url):
    if not isinstance(url,str): raise ValueError('Invalid download URL')
    parsed=urlsplit(url)
    if (parsed.scheme!='https' or parsed.hostname not in OFFICIAL_HOSTS or parsed.username is not None or
            parsed.password is not None or parsed.port not in (None,443) or parsed.fragment or
            any(ord(c)<32 for c in url)):
        raise ValueError('Download redirected outside approved official HTTPS hosts')
    return url


class _OfficialRedirect(HTTPRedirectHandler):
    max_redirections=5
    max_repeats=2

    def redirect_request(self,req,fp,code,msg,headers,newurl):
        validate_download_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


class _HTTPStream:
    """One socket read per chunk permits cancellation/deadline checks between reads."""
    def __init__(self, response): self._response=response
    def read(self, count): return self._response.read1(count)
    def __enter__(self): return self
    def __exit__(self, *exc): self._response.close()


def open_official(url):
    """No cookies, auth handlers, netrc, environment tokens or proxy configuration."""
    validate_download_url(url)
    opener=build_opener(ProxyHandler({}),_OfficialRedirect(),HTTPSHandler())
    try:
        response=opener.open(Request(url,headers={'User-Agent':'model-looking-glass-curated-download/1','Accept-Encoding':'identity'}),timeout=NETWORK_TIMEOUT)
    except HTTPError as exc:
        exc.close()
        raise
    try:
        validate_download_url(response.geturl())
        if response.status != 200: raise ValueError('Expected complete HTTP 200 response')
        if response.headers.get('Content-Encoding','identity').lower()!='identity':
            raise ValueError('Encoded download response rejected')
        return _HTTPStream(response)
    except BaseException:
        response.close(); raise


def _ordinary_path(path):
    """Reject redirected components. Managed storage is trusted local, not hostile-user isolation."""
    path=Path(os.path.abspath(path))
    for part in (path,*path.parents):
        if part.is_symlink() or (hasattr(part,'is_junction') and part.is_junction()):
            raise ValueError('Managed download path contains a symlink or junction')
        if part.exists() and not part.is_dir():
            raise ValueError('Managed download path component is not a directory')
    return path


class _Cancelled(Exception): pass


class _DownloadError(ValueError):
    """Safe locally authored public error; never wrap a remote message verbatim."""
    pass


def _network_error(exc):
    """Describe known transport failures without exposing remote exception text."""
    reason=exc.reason if isinstance(exc,URLError) else exc
    if isinstance(reason,ssl.SSLCertVerificationError):
        return ('TLS certificate verification failed. Check the Python environment\'s trusted '
                'certificates and network security settings; no model was published.')
    if isinstance(reason,ssl.SSLError):
        return 'A secure connection could not be established. Check TLS and network security settings; no model was published.'
    if isinstance(reason,socket.gaierror):
        return 'The model host could not be resolved. Check DNS and connectivity; no model was published.'
    if isinstance(reason,TimeoutError):
        return 'Download timed out. No model was published; retry explicitly when ready.'
    if isinstance(reason,ConnectionError):
        return 'The connection to the model host failed. Check connectivity and network restrictions; no model was published.'
    return 'The model host could not be reached. Check connectivity and network restrictions; no model was published.'


class DownloadManager:
    """One explicit job; only start() writes/uses the network. No automatic retries.

    Each job owns a new mode-0700 directory. Its private .partial tree is renamed
    to model only after all pins pass. Prior downloads and loaded models are never
    overwritten, selected, unloaded or otherwise touched. Cancellation is cooperative
    between network reads (socket timeout applies), not instantaneous. Process exit
    can leave an owned partial directory; there is no automatic resume or deletion.
    """
    def __init__(self, root, catalog=CATALOG, *, transport=None):
        self.root=Path(os.path.abspath(root))
        self._catalog={entry.id:entry for entry in catalog}
        if len(self._catalog)!=len(catalog): raise ValueError('Duplicate catalog IDs')
        self._transport=transport or open_official
        self._lock=threading.Lock(); self._cancel=threading.Event(); self._thread=None
        self._status=dict(state='idle',job_id=None,model_id=None,received_bytes=0,total_bytes=0,file=None,path=None,error=None)

    def status(self):
        with self._lock: return dict(self._status)

    def start(self,model_id, *, expected_revision=None):
        if not isinstance(model_id,str) or model_id not in self._catalog:
            raise ValueError('Choose an existing curated model ID')
        if expected_revision is not None and expected_revision!=self._catalog[model_id].revision:
            raise ValueError('Catalog revision changed; refresh before starting a download')
        with self._lock:
            if self._status['state'] in ('running','cancelling'):
                raise ValueError('A download is already active; cancel it or wait')
            _ordinary_path(self.root)
            spec=self._catalog[model_id]; job_id=uuid.uuid4().hex
            self._cancel=threading.Event()
            self._status=dict(state='running',job_id=job_id,model_id=model_id,received_bytes=0,
                              total_bytes=spec.download_bytes,file=None,path=None,error=None)
            self._thread=threading.Thread(target=self._work,args=(spec,self._cancel),name='mlg-download-'+job_id,daemon=True)
            try: self._thread.start()
            except BaseException:
                self._status.update(state='failed',error='Could not start download worker. No download was started.'); raise
            return dict(self._status)

    def cancel(self,job_id):
        with self._lock:
            if not isinstance(job_id,str) or job_id!=self._status['job_id']:
                raise ValueError('Download ID is stale; refresh download status')
            if self._status['state'] in ('running','cancelling'):
                self._cancel.set(); self._status['state']='cancelling'
            return dict(self._status)

    def _update(self,**changes):
        with self._lock: self._status.update(changes)

    def _work(self,spec,cancel):
        owned=None; published=False; started=time.monotonic()
        def check():
            if cancel.is_set(): raise _Cancelled()
            if time.monotonic()-started>JOB_TIMEOUT: raise TimeoutError('Download time limit')
        try:
            check(); root=_ordinary_path(self.root); root.mkdir(parents=True,exist_ok=True); _ordinary_path(root)
            if shutil.disk_usage(root).free < spec.download_bytes+64*MiB:
                raise _DownloadError('Not enough free disk space for this snapshot plus 64 MiB staging headroom')
            owned=Path(tempfile.mkdtemp(prefix=spec.id+'-',dir=root))
            staging=owned/'.partial'; staging.mkdir(mode=0o700)
            received=0
            for file in spec.files:
                check(); _ordinary_path(staging); self._update(file=file.name)
                url=f'https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{quote(file.name,safe="")}'
                hasher=hashlib.sha256() if file.digest_kind=='sha256' else hashlib.sha1()
                if file.digest_kind=='git_blob_sha1': hasher.update(f'blob {file.size}\0'.encode())
                count=0
                with self._transport(url) as response, (staging/file.name).open('xb') as out:
                    while True:
                        check(); chunk=response.read(min(CHUNK_BYTES,file.size-count+1)); check()
                        if not chunk: break
                        if not isinstance(chunk,bytes) or count+len(chunk)>file.size:
                            raise _DownloadError('Downloaded file exceeds its exact pinned byte size')
                        out.write(chunk); hasher.update(chunk); count+=len(chunk); received+=len(chunk)
                        self._update(received_bytes=received)
                    out.flush(); os.fsync(out.fileno())
                if count!=file.size or hasher.hexdigest()!=file.digest:
                    raise _DownloadError('Downloaded file did not match its pinned size and hash')
            check(); _ordinary_path(owned); _ordinary_path(staging)
            destination=owned/'model'
            # Exclusive private parent prevents collisions with any prior managed download.
            if destination.exists() or destination.is_symlink(): raise _DownloadError('Destination already exists')
            with self._lock:
                check()
                os.rename(staging,destination)
                published=True
                self._status.update(state='complete',path=str(destination),file=None,error=None)
        except BaseException as exc:
            cleanup_error=False
            if owned is not None and not published:
                try:
                    _ordinary_path(owned)
                    # Remove only this job's new private tree, never other downloads.
                    shutil.rmtree(owned)
                except (OSError,ValueError): cleanup_error=True
            # Never return raw HTTP exception text: CDN signed URLs can contain secrets.
            if cleanup_error: error='Download stopped; its private partial directory could not be cleaned. Inspect managed storage before retrying.'
            elif isinstance(exc,_Cancelled): error=None
            elif isinstance(exc,_DownloadError): error=str(exc)
            elif isinstance(exc,HTTPError): error=f'Public snapshot request returned HTTP {exc.code}. No credentials were used; check that the pinned public source remains available.'
            elif isinstance(exc,(URLError,ssl.SSLError,socket.gaierror,ConnectionError)): error=_network_error(exc)
            elif isinstance(exc,TimeoutError): error='Download timed out. No model was published; retry explicitly when ready.'
            else: error=f'Download failed ({type(exc).__name__}). No model was published; check connectivity, disk space and source availability.'
            self._update(state='cancelled' if isinstance(exc,_Cancelled) and not cleanup_error else 'failed',error=error,file=None,path=None)
