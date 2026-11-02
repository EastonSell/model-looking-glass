"""Reserve the public explicit-year-direction prompts in the known inventory."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v20 as v20

INVENTORY_VERSION = "distilgpt2-development-exposures-v21"
PREDECESSOR_SHA256 = "3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd"
INVENTORY_SHA256 = "a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae"
FORBIDDEN_PREFIX_COUNT = 3888
TARGET = v20.TARGET
PUBLIC_DISCOVERY = v20.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v20.FACTORIAL_GROUPS
FACTORIAL_ROWS = v20.FACTORIAL_ROWS
RECIPIENT_CASES = v20.RECIPIENT_CASES
INJECTION_RECORD_IDS = v20.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v20.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v20.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v20.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v20.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v20.GROUPED_REPORTED_ROWS
SCREENED_CASES = v20.SCREENED_CASES
SCREENED_ROWS = v20.SCREENED_ROWS
AGREEMENT_CASES = v20.AGREEMENT_CASES
AGREEMENT_ROWS = v20.AGREEMENT_ROWS
AGREEMENT_ANSWER_IDS = v20.AGREEMENT_ANSWER_IDS

PROTOCOL_HEAD = "e5f71eb66e233a54f1171d1e6f489c0fca587156"
PROTOCOL_MERGE = "921c2f6f44fe6e2b32e69b822049840a78b6f6f7"
PROTOCOL_PATH = "docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md"
PROTOCOL_SHA256 = "68f028e401fd4c43d1cac89772da10c92cf5e5de7af30c4043c18c21678b902e"
CASES_COMMIT = "3425253a9479ee4d09241baaf0da6c499667f2c8"
CASE_PATH = "experiments/distilgpt2_explicit_year_direction_v1.json"
CASE_SHA256 = "419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9"
ORDERED_ROWS_SHA256 = "2454f4176696b53a8a7a6b395d5b594fd9ed3c081a5909b69af11e61c330cb66"
COMPARISON_INPUT_PATH = "results/distilgpt2-explicit-year-direction-cases-v1/natural-comparison-inputs.json"
COMPARISON_INPUT_SHA256 = "0ebd8c2de1126a21b31ee06a33a8c19cb29fe2938a09b45bf5fb1acdb48b1478"
RECEIPT_PATH = "results/distilgpt2-explicit-year-direction-cases-v1/natural-comparison.json"
RECEIPT_SHA256 = "ecdb68744d67c596365021ed7920922e69209a8abb9e02cb2ca45835762e9695"
CANDIDATE_GROUP_KEYS = ("above_ids", "middle_ids", "local_above_ids", "local_below_ids")

EXPLICIT_YEAR_DIRECTION_CASES = (('year-direction-01-war-after-low',
  'The war took place after the year 1707 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-war-after-high',
  'The war took place after the year 1716 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-01-war-before-low',
  'The war took place before the year 1707 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-war-before-high',
  'The war took place before the year 1716 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-01-war-near-low',
  'The war took place near the year 1707 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-war-near-high',
  'The war took place near the year 1716 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-01-reign-after-low',
  'The reign took place after the year 1707 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-reign-after-high',
  'The reign took place after the year 1716 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-01-reign-before-low',
  'The reign took place before the year 1707 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-reign-before-high',
  'The reign took place before the year 1716 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-01-reign-near-low',
  'The reign took place near the year 1707 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 2998, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (2919, 2931, 940), (3023, 2713, 3312))),
 ('year-direction-01-reign-near-high',
  'The reign took place near the year 1716 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 1433, 287, 262, 614, 1596),
  ((1129, 1238, 2481), (940, 1157, 1065), (1558, 1507, 1129), (1485, 1415, 1314))),
 ('year-direction-02-war-after-low',
  'The war took place after the year 1729 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-war-after-high',
  'The war took place after the year 1738 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-02-war-before-low',
  'The war took place before the year 1729 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-war-before-high',
  'The war took place before the year 1738 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-02-war-near-low',
  'The war took place near the year 1729 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-war-near-high',
  'The war took place near the year 1738 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-02-reign-after-low',
  'The reign took place after the year 1729 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-reign-after-high',
  'The reign took place after the year 1738 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-02-reign-before-low',
  'The reign took place before the year 1729 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-reign-before-high',
  'The reign took place before the year 1738 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-02-reign-near-low',
  'The reign took place near the year 1729 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 1959, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (1270, 3132, 2624), (2075, 1983, 2078))),
 ('year-direction-02-reign-near-high',
  'The reign took place near the year 1738 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 2548, 287, 262, 614, 1596),
  ((3901, 3682, 3559), (2624, 2091, 2682), (2670, 1821, 3901), (2327, 2623, 2718))),
 ('year-direction-03-war-after-low',
  'The war took place after the year 1751 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-war-after-high',
  'The war took place after the year 1760 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-03-war-before-low',
  'The war took place before the year 1751 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-war-before-high',
  'The war took place before the year 1760 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-03-war-near-low',
  'The war took place near the year 1751 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-war-near-high',
  'The war took place near the year 1760 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-03-reign-after-low',
  'The reign took place after the year 1751 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-reign-after-high',
  'The reign took place after the year 1760 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-03-reign-before-low',
  'The reign took place before the year 1751 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-reign-before-high',
  'The reign took place before the year 1760 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-03-reign-near-low',
  'The reign took place near the year 1751 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 4349, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (4309, 4310, 4051), (2780, 2920, 1120))),
 ('year-direction-03-reign-near-high',
  'The reign took place near the year 1760 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 1899, 287, 262, 614, 1596),
  ((5066, 2414, 2996), (4051, 2816, 3980), (5333, 5237, 5066), (3553, 3365, 3270))),
 ('year-direction-04-war-after-low',
  'The war took place after the year 1773 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-war-after-high',
  'The war took place after the year 1782 in the year 17',
  (464, 1175, 1718, 1295, 706, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))),
 ('year-direction-04-war-before-low',
  'The war took place before the year 1773 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-war-before-high',
  'The war took place before the year 1782 in the year 17',
  (464, 1175, 1718, 1295, 878, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))),
 ('year-direction-04-war-near-low',
  'The war took place near the year 1773 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-war-near-high',
  'The war took place near the year 1782 in the year 17',
  (464, 1175, 1718, 1295, 1474, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))),
 ('year-direction-04-reign-after-low',
  'The reign took place after the year 1773 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-reign-after-high',
  'The reign took place after the year 1782 in the year 17',
  (464, 13580, 1718, 1295, 706, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))),
 ('year-direction-04-reign-before-low',
  'The reign took place before the year 1773 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-reign-before-high',
  'The reign took place before the year 1782 in the year 17',
  (464, 13580, 1718, 1295, 878, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))),
 ('year-direction-04-reign-near-low',
  'The reign took place near the year 1773 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 4790, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (4524, 2425, 4304), (2154, 4869, 4761))),
 ('year-direction-04-reign-near-high',
  'The reign took place near the year 1782 in the year 17',
  (464, 13580, 1718, 1295, 1474, 262, 614, 1596, 6469, 287, 262, 614, 1596),
  ((5332, 4521, 5774), (4304, 3324, 3695), (5999, 5705, 5332), (3720, 1795, 6659))))
EXPLICIT_YEAR_DIRECTION_ROWS = tuple(row for _, _, row, _ in EXPLICIT_YEAR_DIRECTION_CASES)
_canonical = v20._canonical
_prefixes = v20._prefixes


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    predecessor = v20.inventory_payload()
    if (v20.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v20.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_explicit_year_direction_rows": 48}
    cases = EXPLICIT_YEAR_DIRECTION_CASES
    if (type(cases) is not tuple or len(cases) != 48
            or any(type(case) is not tuple or len(case) != 4 for case in cases)):
        raise ValueError("expected the 48 fixed public explicit-year prompts")
    ids, rows = [case[0] for case in cases], [case[2] for case in cases]
    if (len(set(ids)) != 48 or len(set(rows)) != 48
            or any(type(case_id) is not str or re.fullmatch(r"year-direction-[a-z0-9-]+", case_id) is None
                   for case_id in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _, _ in cases)
            or any(type(row) is not tuple or len(row) != 13
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)
            or any(type(groups) is not tuple or len(groups) != 4
                   or any(type(group) is not tuple or len(group) != 3
                          or any(type(token) is not int or not 0 <= token < 50257 for token in group)
                          for group in groups) for _, _, _, groups in cases)):
        raise ValueError("explicit-year case identities, inputs or candidates are invalid")
    old_rows = [tuple(record["token_ids"]) for record in payload["records"]]
    old_rows.extend(tuple(record["token_ids"] + [record["target_token_id"]])
                    for record in payload["records"] if type(record.get("target_token_id")) is int)
    old_rows.extend(tuple(record["token_ids"] + [token]) for record in payload["records"]
                    for token in record.get("candidate_token_ids", ()))
    old_rows.extend(tuple(record["token_ids"] + [token]) for record in payload["records"]
                    for group in record.get("candidate_token_groups", {}).values() for token in group)
    complete = [row + (token,) for _, _, row, groups in cases for group in groups for token in group]
    if any(_related(candidate, prior) for candidate in rows + complete for prior in old_rows):
        raise ValueError("explicit-year input or candidate continuation has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("explicit-year prompts contain a full-row prefix relation")
    prior_ids = {record["record_id"] for record in payload["records"]}
    for case_id, text, row, groups in cases:
        record_id = "planned-" + case_id
        if record_id in prior_ids:
            raise ValueError("explicit-year prompt record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "planned", "token_ids": list(row), "public_text": text,
            "candidate_token_groups": {key: list(group) for key, group in zip(CANDIDATE_GROUP_KEYS, groups)},
            "provenance": "Public non-blind explicit-year prompt; planned only and no model forward is asserted. "
                          f"Case inventory SHA-256 {CASE_SHA256}.",
            "references": [PROTOCOL_PATH, CASE_PATH], "reference_commit": CASES_COMMIT,
        })
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("explicit-year prompt forbidden-prefix count changed")
    return payload

def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return payload, digest


def inventory_digest():
    return _checked_inventory()[1]


def _forbidden_rows():
    return _prefixes(inventory_payload()["records"])


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
