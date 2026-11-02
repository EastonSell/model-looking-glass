"""Freeze a model-free, paired DistilBERT multisuffix agreement manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
TOKENIZER_SHA = {
    'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
    'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
    'config.json': '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83',
}
SUBJECTS = (('rider', 'riders'), ('hunter', 'hunters'),
            ('scholar', 'scholars'), ('runner', 'runners'))
FRAMES = (('At the park', 'near', ('guest', 'guests')),
          ('Beyond the village', 'beside', ('neighbor', 'neighbors')))
VARIANTS = (('terminal', ''), ('period', '.'), ('present', ' present.'),
            ('outside', ' outside.'), ('awake', ' awake.'))
SUFFIX_IDS = {'present':2556, 'outside':2648, 'awake':8300}


def build(snapshot):
    for name, expected in TOKENIZER_SHA.items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != expected:
            raise ValueError('tokenizer asset differs: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.mask_token_id != 103 or tokenizer.sep_token_id != 102:
        raise ValueError('unexpected special tokens')
    for pair in (*SUBJECTS, *(f[2] for f in FRAMES)):
        if any(len(tokenizer(x, add_special_tokens=False)['input_ids']) != 1 for x in pair):
            raise ValueError('noun must be one WordPiece: ' + repr(pair))
    for word, expected in SUFFIX_IDS.items():
        if tokenizer(word, add_special_tokens=False)['input_ids'] != [expected]:
            raise ValueError('suffix token differs: ' + word)
    cases = []
    for frame, (preamble, relation, attractors) in enumerate(FRAMES):
        for attractor_number, attractor in enumerate(attractors):
            for lemma, subjects in enumerate(SUBJECTS):
                for number, subject in enumerate(subjects):
                    index = len(cases)
                    stem = f'{preamble}, the {subject} {relation} the {attractor}'
                    variants = {}
                    for name, suffix in VARIANTS:
                        text = f'{stem} [MASK]{suffix}'
                        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
                        ids = encoded['input_ids']
                        expected_tail = ([102] if name == 'terminal' else [1012,102] if name == 'period'
                                         else [SUFFIX_IDS[name],1012,102])
                        if (ids[6] != tokenizer(subject, add_special_tokens=False)['input_ids'][0]
                                or ids[10] != 103 or ids[11:] != expected_tail):
                            raise ValueError('unexpected subject, mask or suffix alignment')
                        variants[name] = {'text':text, 'input_ids':ids,
                                          'attention_mask':encoded['attention_mask'],
                                          'subject_position':6, 'output_position':10}
                    prefix = variants['terminal']['input_ids'][:-1]
                    if any(row['input_ids'][:len(prefix)] != prefix for row in variants.values()):
                        raise ValueError('variant prefixes through mask differ')
                    cases.append({'index':index,'case_id':f'f{frame}-a{attractor_number}-l{lemma}-n{number}',
                                  'frame':frame,'attractor_number':attractor_number,'attractor':attractor,
                                  'lemma':lemma,'number':number,'subject':subject,
                                  'same_donor_index':index ^ 2,'opposite_donor_index':index ^ 3,
                                  'variants':variants})
    return {'schema':'bert-multisuffix-agreement-cases-v1',
            'model':'distilbert/distilbert-base-uncased','revision':REVISION,
            'tokenizer_asset_sha256':TOKENIZER_SHA,
            'targets':{'is':2003,'are':2024,'was':2001,'were':2020},
            'variants':[name for name, _ in VARIANTS], 'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=(json.dumps(build(a.snapshot),sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    with a.output.open('xb') as f:
        f.write(data)
    print('cases',32,'variant_rows',160,'sha256',hashlib.sha256(data).hexdigest())

if __name__=='__main__':
    main()
