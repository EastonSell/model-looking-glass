import hashlib
import socket
import ssl
from urllib.error import HTTPError, URLError

import pytest

from looking_glass.app_downloads import DownloadManager, FileSpec, ModelSpec


def spec():
    return ModelSpec(id='test', repo='test/test', revision='a' * 40, title='Test',
                     license='test', family='gpt2', float32_element_count=1,
                     hidden_size=1, layers=1, attention_heads=1, intermediate_size=1,
                     vocab_size=1, max_context=1,
                     files=(FileSpec('config.json', 1, hashlib.sha256(b'x').hexdigest()),))


@pytest.mark.parametrize(('failure', 'expected'), [
    (URLError(ssl.SSLCertVerificationError('secret https://example.test/token')), 'TLS certificate verification failed'),
    (URLError(socket.gaierror('secret')), 'could not be resolved'),
    (URLError(TimeoutError('secret')), 'timed out'),
    (URLError(ConnectionRefusedError('secret')), 'connection to the model host failed'),
    (URLError('secret'), 'could not be reached'),
    (HTTPError('https://example.test/token', 404, 'secret', {}, None), 'HTTP 404'),
])
def test_network_failure_is_safe_and_never_publishes(tmp_path, failure, expected):
    def transport(_url):
        raise failure

    manager = DownloadManager(tmp_path, (spec(),), transport=transport)
    manager.start('test')
    manager._thread.join(timeout=5)
    assert not manager._thread.is_alive()
    status = manager.status()
    assert status['state'] == 'failed'
    assert status['path'] is None
    assert expected in status['error']
    assert 'secret' not in status['error']
    assert 'https://' not in status['error']
    assert list(tmp_path.iterdir()) == []
