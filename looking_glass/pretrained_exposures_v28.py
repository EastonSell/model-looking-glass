"""Reserve 64 fixed answer-format inputs; no forward is asserted."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v27 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v28"
PREDECESSOR_SHA256 = "ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5"
INVENTORY_SHA256 = "2e883a9694e42f0eee8c2a82448f8b0a68b28ab91ffefa12ef57fd74b0ebb9f2"
FORBIDDEN_PREFIX_COUNT = 5047
CASES_COMMIT = "46842b08338fef9fc9aefc2c90aab3a8af63c66c"
CASE_PATH = "experiments/answer_format_feasibility_cases_v1.json"
CASE_SHA256 = "cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a"
REFERENCES = ('docs/ANSWER_FORMAT_FEASIBILITY_CPU_V1.md', 'docs/ANSWER_FORMAT_FEASIBILITY_SCOPE_V1.md', 'docs/ANSWER_FORMAT_FEASIBILITY_CASES_V1.md', 'experiments/answer_format_feasibility_cases_v1.json', 'experiments/answer_format_feasibility_natural_comparison_v1.json')
for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS', 'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS', 'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS', 'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES', 'SCREENED_ROWS', 'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES', 'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES', 'EXPLICIT_YEAR_DIRECTION_ROWS', 'YEAR_LABEL_CONTROL_CASES', 'YEAR_LABEL_CONTROL_ROWS', 'FULL_TOP_CASES', 'FULL_TOP_ROWS', 'AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS', 'AGREEMENT_SUBJECT_FOLLOWUP_CASES', 'AGREEMENT_SUBJECT_FOLLOWUP_ROWS'):
    globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes
ANSWER_FORMAT_CASES = (('answer-format-development-france-statement-none',
  "In this exercise, France's national capital is",
  (818, 428, 5517, 11, 4881, 338, 2260, 3139, 318),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-statement-neutral',
  "A short geography prompt appears below.\nIn this exercise, France's national capital is",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 818, 428, 5517, 11, 4881, 338, 2260, 3139, 318),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-statement-spain',
  "The capital of Spain is Madrid.\nIn this exercise, France's national capital is",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 818, 428, 5517, 11, 4881, 338, 2260, 3139, 318),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-statement-sweden',
  "The capital of Sweden is Stockholm.\nIn this exercise, France's national capital is",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 818, 428, 5517, 11, 4881, 338, 2260, 3139, 318),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-question-none',
  'What city serves as the national capital of France? Answer:',
  (2061, 1748, 9179, 355, 262, 2260, 3139, 286, 4881, 30, 23998, 25),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-question-neutral',
  'A short geography prompt appears below.\nWhat city serves as the national capital of France? Answer:',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 4881, 30, 23998, 25),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-question-spain',
  'The capital of Spain is Madrid.\nWhat city serves as the national capital of France? Answer:',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 4881, 30, 23998, 25),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-france-question-sweden',
  'The capital of Sweden is Stockholm.\nWhat city serves as the national capital of France? Answer:',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 4881, 30, 23998, 25),
  (6342, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-japan-statement-none',
  "In this exercise, Japan's national capital is",
  (818, 428, 5517, 11, 2869, 338, 2260, 3139, 318),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-statement-neutral',
  "A short geography prompt appears below.\nIn this exercise, Japan's national capital is",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 818, 428, 5517, 11, 2869, 338, 2260, 3139, 318),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-statement-spain',
  "The capital of Spain is Madrid.\nIn this exercise, Japan's national capital is",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 818, 428, 5517, 11, 2869, 338, 2260, 3139, 318),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-statement-sweden',
  "The capital of Sweden is Stockholm.\nIn this exercise, Japan's national capital is",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 818, 428, 5517, 11, 2869, 338, 2260, 3139, 318),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-question-none',
  'What city serves as the national capital of Japan? Answer:',
  (2061, 1748, 9179, 355, 262, 2260, 3139, 286, 2869, 30, 23998, 25),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-question-neutral',
  'A short geography prompt appears below.\nWhat city serves as the national capital of Japan? Answer:',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 2869, 30, 23998, 25),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-question-spain',
  'The capital of Spain is Madrid.\nWhat city serves as the national capital of Japan? Answer:',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 2869, 30, 23998, 25),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-japan-question-sweden',
  'The capital of Sweden is Stockholm.\nWhat city serves as the national capital of Japan? Answer:',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 2869, 30, 23998, 25),
  (11790, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-italy-statement-none',
  "In this exercise, Italy's national capital is",
  (818, 428, 5517, 11, 8031, 338, 2260, 3139, 318),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-statement-neutral',
  "A short geography prompt appears below.\nIn this exercise, Italy's national capital is",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 818, 428, 5517, 11, 8031, 338, 2260, 3139, 318),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-statement-spain',
  "The capital of Spain is Madrid.\nIn this exercise, Italy's national capital is",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 818, 428, 5517, 11, 8031, 338, 2260, 3139, 318),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-statement-sweden',
  "The capital of Sweden is Stockholm.\nIn this exercise, Italy's national capital is",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 818, 428, 5517, 11, 8031, 338, 2260, 3139, 318),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-question-none',
  'What city serves as the national capital of Italy? Answer:',
  (2061, 1748, 9179, 355, 262, 2260, 3139, 286, 8031, 30, 23998, 25),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-question-neutral',
  'A short geography prompt appears below.\nWhat city serves as the national capital of Italy? Answer:',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 8031, 30, 23998, 25),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-question-spain',
  'The capital of Spain is Madrid.\nWhat city serves as the national capital of Italy? Answer:',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 8031, 30, 23998, 25),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-italy-question-sweden',
  'The capital of Sweden is Stockholm.\nWhat city serves as the national capital of Italy? Answer:',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 2061, 1748, 9179, 355, 262, 2260, 3139, 286, 8031, 30, 23998, 25),
  (10598, 3576, 14708, 29679),
  'development'),
 ('answer-format-development-uk-statement-none',
  "In this exercise, the United Kingdom's national capital is",
  (818, 428, 5517, 11, 262, 1578, 7526, 338, 2260, 3139, 318),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-statement-neutral',
  "A short geography prompt appears below.\nIn this exercise, the United Kingdom's national capital is",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 818, 428, 5517, 11, 262, 1578, 7526, 338, 2260, 3139, 318),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-statement-spain',
  "The capital of Spain is Madrid.\nIn this exercise, the United Kingdom's national capital is",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 818, 428, 5517, 11, 262, 1578, 7526, 338, 2260, 3139, 318),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-statement-sweden',
  "The capital of Sweden is Stockholm.\nIn this exercise, the United Kingdom's national capital is",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 818, 428, 5517, 11, 262, 1578, 7526, 338, 2260, 3139, 318),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-question-none',
  'What city serves as the national capital of the United Kingdom? Answer:',
  (2061, 1748, 9179, 355, 262, 2260, 3139, 286, 262, 1578, 7526, 30, 23998, 25),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-question-neutral',
  'A short geography prompt appears below.\n'
  'What city serves as the national capital of the United Kingdom? Answer:',
  (32,
   1790,
   27876,
   6152,
   3568,
   2174,
   13,
   198,
   2061,
   1748,
   9179,
   355,
   262,
   2260,
   3139,
   286,
   262,
   1578,
   7526,
   30,
   23998,
   25),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-question-spain',
  'The capital of Spain is Madrid.\nWhat city serves as the national capital of the United Kingdom? Answer:',
  (464,
   3139,
   286,
   8602,
   318,
   14708,
   13,
   198,
   2061,
   1748,
   9179,
   355,
   262,
   2260,
   3139,
   286,
   262,
   1578,
   7526,
   30,
   23998,
   25),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-development-uk-question-sweden',
  'The capital of Sweden is Stockholm.\nWhat city serves as the national capital of the United Kingdom? Answer:',
  (464,
   3139,
   286,
   10710,
   318,
   29679,
   13,
   198,
   2061,
   1748,
   9179,
   355,
   262,
   2260,
   3139,
   286,
   262,
   1578,
   7526,
   30,
   23998,
   25),
  (3576, 6342, 14708, 29679),
  'development'),
 ('answer-format-heldout-france-statement-none',
  'The national capital city belonging to France is',
  (464, 2260, 3139, 1748, 16686, 284, 4881, 318),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-statement-neutral',
  'A short geography prompt appears below.\nThe national capital city belonging to France is',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 4881, 318),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-statement-spain',
  'The capital of Spain is Madrid.\nThe national capital city belonging to France is',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 4881, 318),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-statement-sweden',
  'The capital of Sweden is Stockholm.\nThe national capital city belonging to France is',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 4881, 318),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-question-none',
  "Identify France's national capital city. City:",
  (33234, 1958, 4881, 338, 2260, 3139, 1748, 13, 2254, 25),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-question-neutral',
  "A short geography prompt appears below.\nIdentify France's national capital city. City:",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 33234, 1958, 4881, 338, 2260, 3139, 1748, 13, 2254, 25),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-question-spain',
  "The capital of Spain is Madrid.\nIdentify France's national capital city. City:",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 33234, 1958, 4881, 338, 2260, 3139, 1748, 13, 2254, 25),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-france-question-sweden',
  "The capital of Sweden is Stockholm.\nIdentify France's national capital city. City:",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 33234, 1958, 4881, 338, 2260, 3139, 1748, 13, 2254, 25),
  (6342, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-statement-none',
  'The national capital city belonging to Japan is',
  (464, 2260, 3139, 1748, 16686, 284, 2869, 318),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-statement-neutral',
  'A short geography prompt appears below.\nThe national capital city belonging to Japan is',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 2869, 318),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-statement-spain',
  'The capital of Spain is Madrid.\nThe national capital city belonging to Japan is',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 2869, 318),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-statement-sweden',
  'The capital of Sweden is Stockholm.\nThe national capital city belonging to Japan is',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 2869, 318),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-question-none',
  "Identify Japan's national capital city. City:",
  (33234, 1958, 2869, 338, 2260, 3139, 1748, 13, 2254, 25),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-question-neutral',
  "A short geography prompt appears below.\nIdentify Japan's national capital city. City:",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 33234, 1958, 2869, 338, 2260, 3139, 1748, 13, 2254, 25),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-question-spain',
  "The capital of Spain is Madrid.\nIdentify Japan's national capital city. City:",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 33234, 1958, 2869, 338, 2260, 3139, 1748, 13, 2254, 25),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-japan-question-sweden',
  "The capital of Sweden is Stockholm.\nIdentify Japan's national capital city. City:",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 33234, 1958, 2869, 338, 2260, 3139, 1748, 13, 2254, 25),
  (11790, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-statement-none',
  'The national capital city belonging to Italy is',
  (464, 2260, 3139, 1748, 16686, 284, 8031, 318),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-statement-neutral',
  'A short geography prompt appears below.\nThe national capital city belonging to Italy is',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 8031, 318),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-statement-spain',
  'The capital of Spain is Madrid.\nThe national capital city belonging to Italy is',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 8031, 318),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-statement-sweden',
  'The capital of Sweden is Stockholm.\nThe national capital city belonging to Italy is',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 8031, 318),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-question-none',
  "Identify Italy's national capital city. City:",
  (33234, 1958, 8031, 338, 2260, 3139, 1748, 13, 2254, 25),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-question-neutral',
  "A short geography prompt appears below.\nIdentify Italy's national capital city. City:",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 33234, 1958, 8031, 338, 2260, 3139, 1748, 13, 2254, 25),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-question-spain',
  "The capital of Spain is Madrid.\nIdentify Italy's national capital city. City:",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 33234, 1958, 8031, 338, 2260, 3139, 1748, 13, 2254, 25),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-italy-question-sweden',
  "The capital of Sweden is Stockholm.\nIdentify Italy's national capital city. City:",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 33234, 1958, 8031, 338, 2260, 3139, 1748, 13, 2254, 25),
  (10598, 3576, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-statement-none',
  'The national capital city belonging to the United Kingdom is',
  (464, 2260, 3139, 1748, 16686, 284, 262, 1578, 7526, 318),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-statement-neutral',
  'A short geography prompt appears below.\nThe national capital city belonging to the United Kingdom is',
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 262, 1578, 7526, 318),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-statement-spain',
  'The capital of Spain is Madrid.\nThe national capital city belonging to the United Kingdom is',
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 262, 1578, 7526, 318),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-statement-sweden',
  'The capital of Sweden is Stockholm.\nThe national capital city belonging to the United Kingdom is',
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 464, 2260, 3139, 1748, 16686, 284, 262, 1578, 7526, 318),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-question-none',
  "Identify the United Kingdom's national capital city. City:",
  (33234, 1958, 262, 1578, 7526, 338, 2260, 3139, 1748, 13, 2254, 25),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-question-neutral',
  "A short geography prompt appears below.\nIdentify the United Kingdom's national capital city. City:",
  (32, 1790, 27876, 6152, 3568, 2174, 13, 198, 33234, 1958, 262, 1578, 7526, 338, 2260, 3139, 1748, 13, 2254, 25),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-question-spain',
  "The capital of Spain is Madrid.\nIdentify the United Kingdom's national capital city. City:",
  (464, 3139, 286, 8602, 318, 14708, 13, 198, 33234, 1958, 262, 1578, 7526, 338, 2260, 3139, 1748, 13, 2254, 25),
  (3576, 6342, 14708, 29679),
  'heldout'),
 ('answer-format-heldout-uk-question-sweden',
  "The capital of Sweden is Stockholm.\nIdentify the United Kingdom's national capital city. City:",
  (464, 3139, 286, 10710, 318, 29679, 13, 198, 33234, 1958, 262, 1578, 7526, 338, 2260, 3139, 1748, 13, 2254, 25),
  (3576, 6342, 14708, 29679),
  'heldout'))
ANSWER_FORMAT_ROWS = tuple(row for _, _, row, _, _ in ANSWER_FORMAT_CASES)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = [r["record_id"] for r in records]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError("predecessor record identities repeat")
    cases = ANSWER_FORMAT_CASES
    if (type(cases) is not tuple or len(cases) != 64
            or any(type(x) is not tuple or len(x) != 5 for x in cases)):
        raise ValueError("expected 64 fixed answer-format cases")
    ids = [x[0] for x in cases]
    rows = [x[2] for x in cases]
    if (len(set(ids)) != 64 or len(set(rows)) != 64
            or any(type(i) is not str or re.fullmatch(r"answer-format-[a-z0-9-]+", i) is None for i in ids)
            or any("planned-" + i in old_ids for i in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _, _, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 64
                   or any(type(t) is not int or not 0 <= t < 50257 for t in row) for row in rows)
            or any(type(labels) is not tuple or len(labels) != 4 or len(set(labels)) != 4
                   or any(type(t) is not int or not 0 <= t < 50257 for t in labels) for _, _, _, labels, _ in cases)
            or any(role not in ("development", "heldout") for _, _, _, _, role in cases)):
        raise ValueError("answer-format identities or token rows differ")
    old_rows = [tuple(r["token_ids"]) for r in records]
    old_rows.extend(tuple(r["token_ids"] + [r["target_token_id"]]) for r in records if type(r.get("target_token_id")) is int)
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records for token in r.get("candidate_token_ids", ()))
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records
                    for group in r.get("candidate_token_groups", {}).values() for token in group)
    if any(_related(candidate, old) for candidate in rows for old in old_rows):
        raise ValueError("answer-format input has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i+1:]):
        raise ValueError("answer-format inputs contain a full-row prefix relation")
    for case_id, text, row, labels, role in cases:
        records.append({"record_id":"planned-" + case_id,"category":"planned","case_role":role,
                        "token_ids":list(row),"public_text":text,"candidate_token_ids":list(labels),
                        "provenance":"Public prompt planned for a baseline-only answer-format study; "
                                     f"no model forward is asserted. Case SHA-256 {CASE_SHA256}.",
                        "references":list(REFERENCES),"reference_commit":CASES_COMMIT})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version":prior.INVENTORY_VERSION,"sha256":PREDECESSOR_SHA256}
    payload["update"] = {"planned_answer_format_rows":64}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("answer-format input prefix count changed")
    return payload


def inventory_payload():
    predecessor = prior.inventory_payload()
    if (prior.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
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
