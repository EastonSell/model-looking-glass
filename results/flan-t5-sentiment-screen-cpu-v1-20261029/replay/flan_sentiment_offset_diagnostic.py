"""Post-screen whitespace-inclusive token-alignment diagnostic; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

EVENT_SHA = 'eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b'
TERMINAL_SHA = '843a7140cdd7914fb2e5d6a0c382b05f4eb721b561f71f70d04fcdb4ff1d1e8f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    events_raw = (a.attempt / 'events.jsonl').read_bytes()
    terminal_raw = (a.attempt / 'terminal.json').read_bytes()
    if hashlib.sha256(events_raw).hexdigest() != EVENT_SHA or hashlib.sha256(terminal_raw).hexdigest() != TERMINAL_SHA:
        raise ValueError('saved attempt identity differs')
    events = [json.loads(line) for line in events_raw.splitlines()]
    fixed = [e for e in events if e.get('kind') == 'fixed_inputs']
    complete = [e for e in events if e.get('kind') == 'case_completed' and e.get('repeat') is False]
    if len(fixed) != 1 or len(complete) != 64:
        raise ValueError('fixed inputs or outcomes are incomplete')
    cases, tokens = fixed[0]['cases'], fixed[0]['token_rows']
    if len(cases) != 64 or len(tokens) != 64 or [e['case_index'] for e in complete] != list(range(64)):
        raise ValueError('screen case order differs')
    positions = []
    for case, token in zip(cases, tokens):
        word, text = case['adjective'], case['text']
        start = text.rfind(word)
        end = start + len(word)
        matches = [i for i, (left, right) in enumerate(token['offset_mapping']) if left < end and right > start]
        valid = (len(matches) == 1 and token['offset_mapping'][matches[0]] == [start - 1, end]
                 and text[start - 1:start] == ' ' and text[start:end] == word
                 and token['attention_mask'][matches[0]] == 1
                 and token['special_tokens_mask'][matches[0]] == 0)
        positions.append(matches[0] if valid else None)
    def same_context_except_word(i, j):
        x, y, pos = tokens[i], tokens[j], positions[i]
        return (pos is not None and pos == positions[j] and len(x['input_ids']) == len(y['input_ids'])
                and x['attention_mask'] == y['attention_mask'] and x['special_tokens_mask'] == y['special_tokens_mask']
                and [k for k, (v, w) in enumerate(zip(x['input_ids'], y['input_ids'])) if v != w] == [pos])
    details = []
    for i, case in enumerate(cases):
        donors = [j for j, other in enumerate(cases)
                  if other['instruction_index'] == case['instruction_index']
                  and other['subject'] == case['subject'] and other['side'] == case['side']
                  and other['pair_index'] != case['pair_index'] and same_context_except_word(i, j)
                  and complete[j]['classification']['correct'] is True]
        details.append({'index': i, 'word_position': positions[i],
                        'opposite_aligned': same_context_except_word(i, i ^ 1),
                        'same_polarity_clean_donors': donors})
    aligned = sum(details[i]['opposite_aligned'] for i in range(0, 64, 2))
    usable = sum(details[i]['opposite_aligned'] and all(
        complete[j]['classification']['correct'] is True and details[j]['same_polarity_clean_donors']
        for j in (i, i + 1)) for i in range(0, 64, 2))
    result = {'status': 'post_screen_diagnostic',
              'scope': 'Previously unregistered offset rule; no new model or tokenizer call.',
              'rule': 'One attended non-special token has offset [word_start-1,word_end] with exactly one preceding ASCII space; aligned donor rows differ only at that token.',
              'events_sha256': EVENT_SHA, 'terminal_sha256': TERMINAL_SHA,
              'aligned_opposite_pairs': aligned, 'behaviorally_usable_pairs': usable,
              'pair_denominator': 32, 'case_denominator': 64, 'adjective_contrast_families': 4,
              'cases': details}
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with a.output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short diagnostic write')
    print(json.dumps({'output_sha256': hashlib.sha256(data).hexdigest(),
                      'aligned_opposite_pairs': aligned, 'behaviorally_usable_pairs': usable}, sort_keys=True))


if __name__ == '__main__':
    main()
