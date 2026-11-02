import hashlib
import json
import unittest

from looking_glass.agreement_direction_cases_v1 import build_cases, has_relation, select_pairs, verify_inventory


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self):
        self.ids = {'The': 464, 'the': 262, 'is': 318, 'are': 389, 'was': 373, 'were': 547}
        self.next_id = 1000

    def encode(self, text, add_special_tokens=False):
        values = []
        for word in text.strip().split():
            if word not in self.ids:
                self.ids[word] = self.next_id
                self.next_id += 1
            values.append(self.ids[word])
        return FakeEncoding(values)


class CasesTests(unittest.TestCase):
    def test_complete_row_prefix_in_both_directions(self):
        self.assertTrue(has_relation((1, 2), (1, 2, 3)))
        self.assertTrue(has_relation((1, 2, 3), (1, 2)))
        self.assertTrue(has_relation((1, 2), (1, 2)))
        self.assertFalse(has_relation((1, 2), (1, 3)))

    def test_inventory_hash_binds_all_records_not_just_a_claimed_digest(self):
        obj = {'version': 'version-test', 'records': [{'token_ids': [3, 4]}], 'other': 2}
        raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(verify_inventory(raw, 'version-test', digest), obj)
        altered = raw.replace(b'"other":2', b'"other":3')
        with self.assertRaises(ValueError):
            verify_inventory(altered, 'version-test', digest)

    def test_eligibility_skips_shared_id_and_requires_four(self):
        tokenizer = FakeTokenizer()
        tokenizer.ids['duplicate'] = 42
        tokenizer.ids['duplicates'] = 42
        chosen, log = select_pairs(tokenizer, [('duplicate', 'duplicates'),
                                               ('a', 'aa'), ('b', 'bb'),
                                               ('c', 'cc'), ('d', 'dd')], set())
        self.assertEqual(chosen[0]['words'], ['a', 'aa'])
        self.assertEqual(log[0]['reason'], 'same_token')
        with self.assertRaises(ValueError):
            select_pairs(tokenizer, [('duplicate', 'duplicates')], set())

    def test_four_frames_cross_both_number_dimensions(self):
        cases, selection = build_cases(FakeTokenizer())
        targets = [case for case in cases if case['role'] == 'target']
        controls = [case for case in cases if case['role'] == 'context']
        self.assertEqual((len(targets), len(controls)), (32, 8))
        self.assertEqual(len({tuple(c['token_ids']) for c in cases}), 40)
        for frame in range(4):
            rows = [c for c in targets if c['frame_index'] == frame]
            self.assertEqual(len(rows), 8)
            self.assertEqual({(c['morphology'], c['subject_number'], c['attractor_number'])
                              for c in rows},
                             {(m, s, a) for m in ('regular', 'irregular')
                              for s in ('singular', 'plural') for a in ('singular', 'plural')})
        self.assertEqual(len(selection['regular']), 4)

    def test_continuation_boundary_must_be_preserved(self):
        class Bad(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens)
                if text.endswith(' is') and len(result.ids) > 1:
                    result.ids[-1] = 999999
                return result
        with self.assertRaises(ValueError):
            build_cases(Bad())


if __name__ == '__main__':
    unittest.main()
