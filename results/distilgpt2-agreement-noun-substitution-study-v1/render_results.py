"""Render the fixed substitution result note from the reviewed saved replay."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

REPORT_SHA256 = 'a3bc57be6506e1464e2cfb3d4fcc534a44c06a958c01086cdf98964aa4a09a94'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('independent replay bytes differ')
    report = json.loads(raw)
    if len(report['cells']) != 12 or len(report['cases']) != 54:
        raise ValueError('fixed case counts differ')
    lines = [
        '# Agreement noun-substitution study', '',
        'The preregistered descriptive comparison passed on these fixed public cases. '
        'Across 12 cells, the mean difference between opposite-number and same-number '
        'different-noun response distances was **0.219906** logit for the frozen '
        'agreement direction, versus **0.034990** and **0.050526** for the means of '
        'the individual absolute active-control differences. Eleven cells were '
        'strictly positive and one was exactly zero. All 48 target clean baselines '
        'had the expected strict is/are sign.', '',
        'This is prospective, public and nonblind evidence of differential response '
        'on a small fixed design. It does not identify a unique abstract grammatical '
        'representation: surface inflection, lexical differences and '
        'prompt-dependent or nonlinear gain remain viable explanations. The '
        'same nouns recur across frames and contexts, so the 12 cells are not '
        'independent replications and the regular/irregular comparison is '
        'lexically confounded.', '',
        '## Design and fixed decision rule', '',
        'The [protocol](../../docs/AGREEMENT_NOUN_SUBSTITUTION_V1.md) was published '
        'before the [54 fixed cases](../../experiments/distilgpt2_agreement_noun_substitution_cases_v1.json). '
        'The 48 targets cross two frames, three contexts (no attractor, singular '
        'attractor, plural attractor), two morphology groups, two lemmas and two '
        'subject forms; six subjectless fragments are separate descriptive '
        'controls. The regular pairs are hand/hands and arm/arms; the irregular '
        'pairs are foot/feet and tooth/teeth. These shared lexical forms are not '
        'a lexical holdout. The frozen layer-3 final-token attention-output '
        'direction u and active controls v1/v2 each used signed input dose ±2.', '',
        'For the is−are logit margin m, each response is '
        '`O_a(x) = (m(x,+2a) − m(x,−2a))/2`. Within a cell let A and B be '
        'different lemmas and s/p their singular/plural forms. '
        '`D_opp = (|O(A_s)−O(B_p)| + |O(B_s)−O(A_p)|)/2`, '
        '`D_same = (|O(A_s)−O(B_s)| + |O(A_p)−O(B_p)|)/2`, and '
        '`Δ = D_opp − D_same`. The signed contrasts are '
        '`N = ((A_s−A_p)+(B_s−B_p))/2`, '
        '`L = ((A_s−B_s)+(A_p−B_p))/2`, and '
        '`I = (A_s−A_p)−(B_s−B_p)`, using the corresponding O values.', '',
        'The fixed rule required adequate clean baselines, overall mean Δ_u ≥0.05 '
        'and above each mean individual |Δ_control|; each morphology mean ≥0.02 '
        'and above its controls; at least 8/12 positive cells and 4/6 in each '
        'morphology; and positive means in both frames. All conditions passed. '
        'These thresholds were descriptive choices motivated by earlier exposed '
        'data, not validated practical thresholds or a statistical test.', '',
        '## All 12 primary cells', '',
        'Values are is−are logits. Control columns are signed cell Δ, whereas '
        'the decision compares the means of their individual absolute values. '
        'Frame 0 is “Before sunrise,” and frame 1 is “After sunset,”.', '',
        '| Frame | Context | Morphology | D_opp u | D_same u | Δ u | Δ v1 | Δ v2 | N u | L u | I u |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for cell in report['cells']:
        axes = cell['contrasts']['present']
        u = axes['u']
        vals = [u['opposite'], u['same'], u['delta'], axes['v1']['delta'],
                axes['v2']['delta'], u['number'], u['lemma'], u['interaction']]
        lines.append('| %s | %s | %s | %s |' %
                     (cell['frame_index'], cell['condition'], cell['morphology'],
                      ' | '.join('%.6f' % v for v in vals)))
    lines += ['',
        'The zero Δ is frame 1 / singular attractor / regular morphology: '
        'both distances are 0.109648, while N is 0.046196. The irregular '
        'cell in that same frame/context has N = −0.084254 despite positive Δ. '
        'This matters because the absolute-distance statistic can be zero even '
        'when a number-form term is nonzero, and a positive value can reflect '
        'surface or lexical gain.', '',
        '| Group | Mean Δ u | Mean abs(Δ v1) | Mean abs(Δ v2) |',
        '|---|---:|---:|---:|',
    ]
    groups = [('all 12', report['aggregates']['overall'])]
    for kind in ('by_morphology', 'by_frame', 'by_condition'):
        groups.extend((kind + ': ' + name, value)
                      for name, value in report['aggregates'][kind].items())
    for label, value in groups:
        axes = value['present']
        lines.append('| %s | %.6f | %.6f | %.6f |' %
                     (label, axes['u']['delta'], axes['v1']['mean_absolute_delta'],
                      axes['v2']['mean_absolute_delta']))
    lines += ['',
        'Every baseline subgroup was fully correct: 24/24 in each frame and '
        'morphology, 16/16 in each context, 12/12 for each lemma, and all '
        '12 cells had four correct forms. No case was filtered.', '',
        '## Subjectless and output controls', '',
        'The six subjectless fragments had positive u odd responses in the '
        'is−are margin, mean **0.269007** and range **0.234711–0.315760**. '
        'Their syntax, final token and position differ from the target sentences; '
        'they are not a controlled subject deletion or a neutral null. Their '
        'responses leave shared output, attractor and context effects plausible.', '',
        '| Fragment | O u | O v1 | O v2 |', '|---|---:|---:|---:|']
    for row in report['fragments']:
        odd = row['odd']
        lines.append('| %s | %.6f | %.6f | %.6f |' %
                     (row['case_id'], odd['u']['present'], odd['v1']['present'], odd['v2']['present']))
    lines += ['',
        '| Margin | Mean Δ u | Mean abs(Δ v1) | Mean abs(Δ v2) |', '|---|---:|---:|---:|']
    for key, label in [('present', 'is−are'), ('past', 'was−were'),
                       ('the_vs_a', 'the−a'), ('in_vs_on', 'in−on')]:
        axes = report['aggregates']['overall'][key]
        lines.append('| %s | %.6f | %.6f | %.6f |' %
                     (label, axes['u']['delta'], axes['v1']['mean_absolute_delta'],
                      axes['v2']['mean_absolute_delta']))
    targets = [x for x in report['cases'] if x['role'] == 'target']
    lines += ['',
        'For the 48 targets, median full-vocabulary KL(clean‖arm) and top-one '
        'changes are shown below. Equal input L2 is not equal functional '
        'strength. The [full replay](science-replay.json) retains every case, '
        'four margins, odd/even values, signed factorial differences, top-20 '
        'lists, target ranks, KL values and delivery diagnostics. The raw '
        'witness tensors are in the archive.', '',
        '| Arm | Median KL | Top-one changes |', '|---|---:|---:|']
    for arm in ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus'):
        vs = [x['vocabulary'][arm] for x in targets]
        median = statistics.median(x['kl_clean_to_arm'] for x in vs)
        changes = sum(x['top_clean'][0] != x['top_arm'][0] for x in vs)
        lines.append('| %s | %.6f | %d/48 |' % (arm, median, changes))
    deliveries = [d for case in report['cases'] for d in case['deliveries'].values()]
    lines += ['',
        'All %d nonzero deliveries passed the fixed norm and signed-projection '
        'checks. Realized norms ranged from %.9f to %.9f; the maximum relative '
        'norm error was %.3g. These checks concern input displacement, not '
        'matched output strength.' % (len(deliveries), min(x['realized_norm'] for x in deliveries),
                                      max(x['realized_norm'] for x in deliveries),
                                      max(x['relative_error'] for x in deliveries)), '',
        '## Evidence and scope', '',
        'The single supervised CPU attempt recorded 486 forwards and 487 journal '
        'events. The direct observer exited zero and reported a complete '
        'candidate; its cooperative observation does not authenticate execution '
        'or prove descendant drain. The [independent source-first replay](science-replay.json) '
        'checked the 1,134 retained tensor payloads, schedule, full saved '
        'last-position logits, no-op equality, landing bits and doses. A '
        '[separate selected-logit replay](selected-replay.json) agrees on all '
        'primary margins and factorial values. The owner recorded full-sequence '
        'finiteness live; unsaved sequence positions cannot be independently '
        'replayed from these records.', '',
        'The exact [owner source](https://github.com/EastonSell/model-looking-glass/tree/ca6c67f30b90f5d79df68e379cbb7cafe21acb08) '
        'was public before the run; the reviewed owner was later merged at '
        '`1d9b97cd31ebe13effbe844e9668567c58ce7283`. The child reconstructed '
        'the v35 reservation; a separate operator observation bound the '
        'independently reviewed v36 successor. The [prelaunch](prelaunch.json), '
        '[boundary](launch-boundary.json), [parent exit](parent-exit.json), '
        '[custodian reports](retrospective-custody.json), and '
        '[redacted replay observation](replay-observation-public.json) are '
        'cooperative observations, not proof of complete exposure history '
        'or private membership. The original natural-split support implication '
        'is conditional on its published preparation; no private comparison '
        'was performed.', '',
        'After the direct exit, Windows reported 91,825,856 allocated bytes '
        'for ordinary files in the attempt; directory metadata and alternate '
        'streams were outside that observation. The inherited supervisor field '
        '`physical_bytes_before_receipt` sums apparent file lengths, not '
        'allocated bytes. Neither observation is an OS quota or descendant '
        'quiescence guarantee.', '',
        'The [parts manifest](parts-manifest.json) binds nine archive parts '
        '(70,545,740 bytes total, SHA-256 '
        '`2db2b35d1ed61aad125ea089d22429f92c635fa5b657c654b1a5e21ae453540b`). '
        'The [file manifest](archive-manifest.json) lists all 1,158 files and '
        'their hashes; the original case-bound claim stays outside the archive. '
        'Run `python -I -S -B join_evidence.py --parts-dir parts '
        '--manifest parts-manifest.json --output study-attempt-v1.tar.gz` '
        'to verify and join them. A failed write or final hash check can leave '
        'a partial output file. No full archive copy was created during '
        'publication staging.', '',
    ]
    result = '\n'.join(lines).encode('utf-8')
    with args.output.open('xb') as stream:
        stream.write(result)
    print(hashlib.sha256(result).hexdigest())


if __name__ == '__main__':
    main()
