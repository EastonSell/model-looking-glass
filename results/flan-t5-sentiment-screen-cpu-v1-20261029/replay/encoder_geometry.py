"""Recompute the post-screen descriptive encoder geometry; no model calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.attempt / 'events.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b':
        raise ValueError('unexpected journal')
    events = [json.loads(line) for line in raw.splitlines()]
    fixed = next(x for x in events if x['kind'] == 'fixed_inputs')
    cases, tokens = fixed['cases'], fixed['token_rows']
    complete = [x for x in events if x['kind'] == 'case_completed' and not x['repeat']]
    if len(cases) != 64 or len(tokens) != 64 or len(complete) != 64:
        raise ValueError('unexpected case inventory')

    def coords(index, kind):
        case, token = cases[index], tokens[index]
        text = case['text']
        word = case['adjective'] if kind == 'word' else case['subject']
        start = text.rfind(word)
        end = start + len(word)
        positions = [k for k, (a, b) in enumerate(token['offset_mapping']) if a < end and b > start]
        if (len(positions) != 1 or token['offset_mapping'][positions[0]] != [start - 1, end]
                or text[start - 1] != ' ' or token['attention_mask'][positions[0]] != 1):
            raise ValueError('unexpected token offset')
        return positions[0]

    def state(index, position):
        ref = complete[index]['encoder_ref']
        data = (args.attempt / ref['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256'] or ref['shape'][0] != 1 or ref['shape'][2] != 512:
            raise ValueError('unexpected encoder state')
        return struct.unpack_from('<512f', data, position * 512 * 4)

    def distance(a, b):
        return math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)))

    groups = []
    for instruction in range(2):
        for subject in range(4):
            for pair_group in range(2):
                ids = [instruction * 32 + subject * 8 + pair_group * 4 + k for k in range(4)]
                distances, word_distances = [], []
                for k, index in enumerate(ids):
                    opposite, same = ids[k ^ 1], ids[k ^ 2]
                    for kind, donor, position in (
                            ('opposite_word', opposite, coords(index, 'word')),
                            ('same_word', same, coords(index, 'word')),
                            ('opposite_instruction', opposite, 0),
                            ('opposite_subject', opposite, coords(index, 'subject'))):
                        if (len(tokens[index]['input_ids']) != len(tokens[donor]['input_ids'])
                                or coords(index, 'word') != coords(donor, 'word')):
                            raise ValueError('unaligned donor')
                        value = distance(state(index, position), state(donor, position))
                        distances.append(value)
                        if kind == 'opposite_word':
                            word_distances.append(value)
                q = min(distances)
                groups.append({'instruction': instruction, 'subject': cases[ids[0]]['subject'],
                               'group': pair_group, 'q': q, 'opposite_word_distances': word_distances,
                               'q_over_smallest_word_distance': q / min(word_distances),
                               'minimum_control': min(distances)})
    output = {'scope': 'post-screen descriptive encoder geometry from saved development data; no new forward and no prospective case selection',
              'events_sha256': hashlib.sha256(raw).hexdigest(), 'groups': groups,
              'summary': {'count': len(groups), 'q_min': min(v['q'] for v in groups),
                          'q_max': max(v['q'] for v in groups),
                          'ratio_min': min(v['q_over_smallest_word_distance'] for v in groups),
                          'ratio_max': max(v['q_over_smallest_word_distance'] for v in groups)}}
    payload = (json.dumps(output, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short write')
    print(json.dumps({'summary': output['summary'], 'sha256': hashlib.sha256(payload).hexdigest()}))


if __name__ == '__main__':
    main()
