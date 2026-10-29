"""Record completed locality inputs and reserve public year/label prompts."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v24 as v24

INVENTORY_VERSION = 'distilgpt2-development-exposures-v25'
PREDECESSOR_SHA256 = '877fa0376925be34ac3de14ff066e5465a67953660bce81011e9c8fd87810bd3'
INVENTORY_SHA256 = '32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144'
FORBIDDEN_PREFIX_COUNT = 4190
PROTOCOL_HEAD = 'd35d685950e36e91fc5cf576c03fe65e8d02aac3'
PROTOCOL_PATH = 'docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md'
PROTOCOL_SHA256 = '5da05c4d35708b420e5843b1eb77fda8699c74755233b510a01683ead2081f4a'
CASES_COMMIT = '469f13747bb90c67cdc3a9a1aa2a867ff69b1616'
CASE_PATH = 'experiments/distilgpt2_year_label_control_v1.json'
CASE_SHA256 = '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
LOCALITY_CASE_SHA256 = 'fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3'
LOCALITY_EVIDENCE_COMMIT = '475228289efa7840fa577833d17573a6a73e001e'
LOCALITY_PHASE_PATH = 'results/targeted-edit-screened-matched-v1-20261029/evaluation/phase-record.json'
LOCALITY_PHASE_SHA256 = '79faacd17bac96039729e241e0ef92b853f70b8b8a9fb326b09ff1e4fbc4bd78'
LOCALITY_JOURNAL_PATH = 'results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/observations.jsonl'
LOCALITY_JOURNAL_SHA256 = '76477d15ec279b28827591b3d643767b51321e68b54196eee49134c35dd4c809'
CANDIDATE_GROUP_KEYS = ('above_ids', 'middle_ids', 'local_above_ids', 'local_below_ids')

for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS',
              'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS',
              'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS',
              'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES', 'SCREENED_ROWS',
              'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES',
              'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES',
              'EXPLICIT_YEAR_DIRECTION_ROWS'):
    globals()[_name] = getattr(v24, _name)

# Exact reviewed public case rows and locality inputs are embedded below.
YEAR_LABEL_CONTROL_CASES = (('year-label-01-01-after-alpha-low',
  'The event tagged alpha took place after the year 1709 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 706, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-after-alpha-high',
  'The event tagged alpha took place after the year 1718 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 706, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-01-after-beta-low',
  'The event tagged beta took place after the year 1709 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 706, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-after-beta-high',
  'The event tagged beta took place after the year 1718 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 706, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-01-before-alpha-low',
  'The event tagged alpha took place before the year 1709 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 878, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-before-alpha-high',
  'The event tagged alpha took place before the year 1718 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 878, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-01-before-beta-low',
  'The event tagged beta took place before the year 1709 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 878, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-before-beta-high',
  'The event tagged beta took place before the year 1718 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 878, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-01-near-alpha-low',
  'The event tagged alpha took place near the year 1709 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 1474, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-near-alpha-high',
  'The event tagged alpha took place near the year 1718 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 1474, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-01-near-beta-low',
  'The event tagged beta took place near the year 1709 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 1474, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-01-near-beta-high',
  'The event tagged beta took place near the year 1718 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 1474, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-after-red-low',
  'The event tagged red took place after the year 1709 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 706, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-after-red-high',
  'The event tagged red took place after the year 1718 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 706, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-after-blue-low',
  'The event tagged blue took place after the year 1709 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 706, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-after-blue-high',
  'The event tagged blue took place after the year 1718 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 706, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-before-red-low',
  'The event tagged red took place before the year 1709 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 878, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-before-red-high',
  'The event tagged red took place before the year 1718 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 878, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-before-blue-low',
  'The event tagged blue took place before the year 1709 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 878, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-before-blue-high',
  'The event tagged blue took place before the year 1718 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 878, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-near-red-low',
  'The event tagged red took place near the year 1709 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 1474, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-near-red-high',
  'The event tagged red took place near the year 1718 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 1474, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-01-02-near-blue-low',
  'The event tagged blue took place near the year 1709 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 1474, 262, 614, 1596, 2931, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (940, 1157, 1065), (3312, 2998, 2919))),
 ('year-label-01-02-near-blue-high',
  'The event tagged blue took place near the year 1718 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 1474, 262, 614, 1596, 1507, 287, 262, 614, 1596),
  ((2481, 1828, 1954), (1065, 1485, 1415), (1129, 1238, 2481), (1314, 1433, 1558))),
 ('year-label-02-01-after-alpha-low',
  'The event tagged alpha took place after the year 1754 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 706, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-after-alpha-high',
  'The event tagged alpha took place after the year 1763 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 706, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-01-after-beta-low',
  'The event tagged beta took place after the year 1754 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 706, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-after-beta-high',
  'The event tagged beta took place after the year 1763 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 706, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-01-before-alpha-low',
  'The event tagged alpha took place before the year 1754 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 878, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-before-alpha-high',
  'The event tagged alpha took place before the year 1763 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 878, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-01-before-beta-low',
  'The event tagged beta took place before the year 1754 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 878, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-before-beta-high',
  'The event tagged beta took place before the year 1763 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 878, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-01-near-alpha-low',
  'The event tagged alpha took place near the year 1754 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 1474, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-near-alpha-high',
  'The event tagged alpha took place near the year 1763 in the year 17',
  (464, 1785, 30509, 17130, 1718, 1295, 1474, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-01-near-beta-low',
  'The event tagged beta took place near the year 1754 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 1474, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-01-near-beta-high',
  'The event tagged beta took place near the year 1763 in the year 17',
  (464, 1785, 30509, 12159, 1718, 1295, 1474, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-after-red-low',
  'The event tagged red took place after the year 1754 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 706, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-after-red-high',
  'The event tagged red took place after the year 1763 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 706, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-after-blue-low',
  'The event tagged blue took place after the year 1754 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 706, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-after-blue-high',
  'The event tagged blue took place after the year 1763 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 706, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-before-red-low',
  'The event tagged red took place before the year 1754 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 878, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-before-red-high',
  'The event tagged red took place before the year 1763 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 878, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-before-blue-low',
  'The event tagged blue took place before the year 1754 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 878, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-before-blue-high',
  'The event tagged blue took place before the year 1763 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 878, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-near-red-low',
  'The event tagged red took place near the year 1754 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 1474, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-near-red-high',
  'The event tagged red took place near the year 1763 in the year 17',
  (464, 1785, 30509, 2266, 1718, 1295, 1474, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))),
 ('year-label-02-02-near-blue-low',
  'The event tagged blue took place near the year 1754 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 1474, 262, 614, 1596, 4051, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2816, 3980, 3553), (4349, 4309, 4310))),
 ('year-label-02-02-near-blue-high',
  'The event tagged blue took place near the year 1763 in the year 17',
  (464, 1785, 30509, 4171, 1718, 1295, 1474, 262, 614, 1596, 5066, 287, 262, 614, 1596),
  ((2791, 3134, 3104), (3553, 3365, 3270), (2414, 2996, 2791), (1899, 5333, 5237))))
LOCALITY_INPUTS = (('screen-locality-other-country-capital-01',
  'The national capital of the United Kingdom is',
  (464, 2260, 3139, 286, 262, 1578, 7526, 318)),
 ('screen-locality-other-country-capital-02',
  'The national capital of Portugal is',
  (464, 2260, 3139, 286, 19101, 318)),
 ('screen-locality-other-country-capital-03', 'The national capital of Poland is', (464, 2260, 3139, 286, 12873, 318)),
 ('screen-locality-other-country-capital-04', 'The national capital of Sweden is', (464, 2260, 3139, 286, 10710, 318)),
 ('screen-locality-other-france-fact-01',
  'The Louvre Museum, formerly a royal palace, is in',
  (464, 4768, 43933, 9594, 11, 15734, 257, 15100, 20562, 11, 318, 287)),
 ('screen-locality-other-france-fact-02', "France's official language is", (28572, 338, 1743, 3303, 318)),
 ('screen-locality-other-france-fact-03',
  'Metropolitan France uses the currency called the',
  (9171, 14823, 4881, 3544, 262, 7395, 1444, 262)),
 ('screen-locality-other-france-fact-04',
  'Metropolitan France lies on the continent of',
  (9171, 14823, 4881, 7363, 319, 262, 15549, 286)),
 ('screen-locality-unrelated-fact-01',
  'The number of sides on a triangle is',
  (464, 1271, 286, 5389, 319, 257, 22950, 318)),
 ('screen-locality-unrelated-fact-02',
  'Water consists chemically of hydrogen and',
  (19184, 10874, 49966, 286, 17669, 290)),
 ('screen-locality-unrelated-fact-03', 'A full hour consists of sixty', (32, 1336, 1711, 10874, 286, 24742)),
 ('screen-locality-unrelated-fact-04',
  'The number of days in a fortnight is',
  (464, 1271, 286, 1528, 287, 257, 46327, 318)))
YEAR_LABEL_CONTROL_ROWS = tuple(row for _, _, row, _ in YEAR_LABEL_CONTROL_CASES)
LOCALITY_IDS = tuple('planned-' + case_id for case_id, _, _ in LOCALITY_INPUTS)
_canonical = v24._canonical
_prefixes = v24._prefixes


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    expected_locality = {f'planned-{case_id}': (text, row) for case_id, text, row in LOCALITY_INPUTS}
    ids = [record['record_id'] for record in records]
    if len(expected_locality) != 12 or len(set(ids)) != len(ids):
        raise ValueError('locality records or predecessor IDs are duplicated')
    if any(ids.count(key) != 1 for key in expected_locality):
        raise ValueError('expected planned locality records are missing')
    new_ids = ['planned-' + case_id for case_id, _, _, _ in YEAR_LABEL_CONTROL_CASES]
    if len(new_ids) != 48 or len(set(new_ids)) != 48 or any(key in ids for key in new_ids):
        raise ValueError('year-label record ID is missing, duplicated or already present')
    for record in records:
        key = record['record_id']
        if key in expected_locality:
            text, row = expected_locality[key]
            if (record.get('category') != 'planned' or record.get('token_ids') != list(row)
                    or record.get('public_text') != text or ('observed-' + key[8:]) in ids):
                raise ValueError('locality input, status or observed identity differs')
            record['record_id'] = 'observed-' + key[8:]
            record['category'] = 'observed-saved'
            record['provenance'] = ('Public screened locality input with seven saved condition rows; '
                                    'the process and cleanup are cooperative observations, not authenticated proof. '
                                    f'Journal SHA-256 {LOCALITY_JOURNAL_SHA256}.')
            record['references'] = list(record.get('references', ())) + [LOCALITY_PHASE_PATH, LOCALITY_JOURNAL_PATH]
            record['reference_commit'] = LOCALITY_EVIDENCE_COMMIT
    cases = YEAR_LABEL_CONTROL_CASES
    rows = [case[2] for case in cases]
    if (len(set(rows)) != 48 or any(type(case_id) is not str or re.fullmatch(r'year-label-[a-z0-9-]+', case_id) is None
            or type(text) is not str or not text or type(row) is not tuple or len(row) != 15
            or any(type(token) is not int or not 0 <= token < 50257 for token in row)
            or type(groups) is not tuple or len(groups) != 4
            or any(type(group) is not tuple or len(group) != 3
                   or any(type(token) is not int or not 0 <= token < 50257 for token in group) for group in groups)
            for case_id, text, row, groups in cases)):
        raise ValueError('year-label identities, rows or candidate groups differ')
    old_rows = [tuple(record['token_ids']) for record in records]
    old_rows.extend(tuple(record['token_ids'] + [record['target_token_id']])
                    for record in records if type(record.get('target_token_id')) is int)
    old_rows.extend(tuple(record['token_ids'] + [token]) for record in records
                    for token in record.get('candidate_token_ids', ()))
    old_rows.extend(tuple(record['token_ids'] + [token]) for record in records
                    for group in record.get('candidate_token_groups', {}).values() for token in group)
    continuations = [row + (token,) for _, _, row, groups in cases for group in groups for token in group]
    if any(_related(candidate, old) for candidate in rows + continuations for old in old_rows):
        raise ValueError('year-label input or continuation has a known full-row relation')
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i+1:]):
        raise ValueError('year-label inputs have a mutual full-row relation')
    for case_id, text, row, groups in cases:
        records.append({'record_id': 'planned-' + case_id, 'category': 'planned',
                        'token_ids': list(row), 'public_text': text,
                        'candidate_token_groups': {key: list(group) for key, group in zip(CANDIDATE_GROUP_KEYS, groups)},
                        'provenance': ('Public non-blind year/label prompt, planned only; no model forward is asserted. '
                                       f'Case inventory SHA-256 {CASE_SHA256}. Old sealed split unchecked.'),
                        'references': [PROTOCOL_PATH, CASE_PATH], 'reference_commit': CASES_COMMIT})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': v24.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'observed_screened_locality_rows': 12, 'planned_year_label_rows': 48,
                         'locality_journal_sha256': LOCALITY_JOURNAL_SHA256,
                         'old_sealed_split_checked': False}
    if FORBIDDEN_PREFIX_COUNT is not None and len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('year-label forbidden-prefix count changed')
    return payload


def inventory_payload():
    predecessor = v24.inventory_payload()
    if (v24.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError('predecessor exposure inventory does not match')
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError('development-exposure inventory does not match its pinned digest')
    return payload, digest


def inventory_digest():
    return _checked_inventory()[1]


def _forbidden_rows():
    return _prefixes(inventory_payload()['records'])


def _has_known_relation(row, forbidden, complete_rows):
    return row in forbidden or any(row[:len(known)] == known for known in complete_rows)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    payload, digest = _checked_inventory()
    count = len(_prefixes(payload["records"]))
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck)
            or check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != digest
            or type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int
            or check.forbidden_row_count != count or check.complete_exposure_history_certified is not False
            or check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return count, "known-prefixes-excluded-from-both", 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only"
            and check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int
            and 0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return count, "known-prefix-discovery-reuse-disclosed", check.known_or_reserved_discovery_overlap_count
    raise ValueError("a matching known-split check is required")


def require_preseal_readiness(check):
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete("known-row checking alone does not authorize a study seal")


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    count, policy, overlap = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete("prior exposure uncertainty must be explicitly disclosed")
    if type(protocol_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("a canonical protocol SHA-256 is required")
    return v2.PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                                 check.candidate_count // 2, count, policy, overlap)
