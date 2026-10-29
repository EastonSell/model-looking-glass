"""Launch the tokenizer-only selector from a trusted checkout in isolated mode.

The caller creates and checks an empty bytecode-prefix directory before
starting this process. This script adds only the package, not the checkout
root, to the import system; it does not authenticate the checkout or Python.
"""
import os
import sys


def require_empty_prefix():
    prefix = sys.pycache_prefix
    if not prefix or not os.path.isdir(prefix) or os.listdir(prefix):
        raise ValueError('an independently prepared empty bytecode prefix is required')


def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('run with -I -B and a fresh -X pycache_prefix')
    require_empty_prefix()
    import importlib.util
    from pathlib import Path
    import subprocess

    root = Path(__file__).resolve().parents[1]
    package = root / 'looking_glass'
    source_paths = ('tools/run_flan_sentiment_selector_v1.py', 'looking_glass/__init__.py',
                    'looking_glass/flan_t5_sentiment_selector_v1.py',
                    'looking_glass/flan_t5_sentiment_screen.py', 'looking_glass/flan_t5_feasibility.py')
    def source_snapshot():
        revision = subprocess.check_output(['git', '--no-replace-objects', 'rev-parse', 'HEAD'], cwd=root).strip().decode('ascii')
        if len(revision) != 40 or subprocess.check_output(['git', '--no-replace-objects', 'cat-file', '-t', revision], cwd=root).strip() != b'commit':
            raise ValueError('selector checkout is not at a Git commit')
        for name in source_paths:
            saved = subprocess.check_output(['git', '--no-replace-objects', 'show', revision + ':' + name], cwd=root)
            if (root / name).read_bytes() != saved:
                raise ValueError('selector source differs from its checkout commit')
        return revision
    revision = source_snapshot()
    spec = importlib.util.spec_from_file_location(
        'looking_glass', package / '__init__.py', submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ValueError('cannot resolve trusted package')
    module = importlib.util.module_from_spec(spec)
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    try:
        from looking_glass.flan_t5_sentiment_selector_v1 import main as selector_main
        selector_main()
    finally:
        require_empty_prefix()
        if source_snapshot() != revision:
            raise ValueError('selector checkout changed during execution')


if __name__ == '__main__':
    main()
