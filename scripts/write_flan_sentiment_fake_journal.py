"""Write an exclusively created synthetic FLAN sentiment journal fixture.

The fake states, logits and rows are invented and contain no model checkpoint
or private study inputs. This is for exercising independent saved-record replay.
"""
import json
import importlib.util
from pathlib import Path
import sys

import torch


def main(checkout, output, mode):
    sys.path.insert(0, str(Path(checkout).resolve()))
    spec = importlib.util.spec_from_file_location('flan_linear_fake',
        Path(__file__).resolve().parent / 'review_flan_sentiment_linear_toy.py')
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    from looking_glass.flan_t5_sentiment_journal import EvidenceJournal, Limits
    from looking_glass.flan_t5_sentiment_runner import _run_fabricated

    output = Path(output).resolve()
    assert output.parent.is_dir() and not output.exists()
    torch.set_num_threads(1)
    journal = EvidenceJournal(output, {'schema': 'synthetic-reader-fixture-v1',
                                       'width': 2, 'vocab_size': 8},
                              limits=Limits(total_bytes=32_000_000, metadata_bytes=16_000_000,
                                            failure_bytes=1_000_000, forward_bytes=4096,
                                            max_forwards=2940, free_bytes=0))
    model = fake.PassingDecoder() if mode == 'pass' else fake.SkipDecoder(
        'rounding' if mode == 'rounding_unsupported' else mode)
    if mode == 'rounding_unsupported':
        model.force_negative = True
    result = _run_fabricated(model.eval(), fake.Tokenizer(), fake.groups(), journal,
                             width=2, vocab_size=8)
    assert result['primary']['status'] == ('supported' if mode == 'pass' else 'threshold-not-met')
    print(json.dumps({'attempt': str(output), 'components': journal.component_count,
                      'retained_bytes': journal.retained_bytes, 'primary': result['primary']}))


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] not in
                                      ('pass', 'zero', 'rounding', 'rounding_unsupported')):
        raise SystemExit('usage: write_flan_sentiment_fake_journal.py CHECKOUT NEW_OUTPUT_DIR '
                         '[pass|zero|rounding|rounding_unsupported]')
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else 'pass')
