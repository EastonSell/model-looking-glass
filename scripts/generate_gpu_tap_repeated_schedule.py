"""Generate the fixed schedule for the proposed repeated two-token tap study."""

import json
import platform
import random


PERMUTATIONS = (
    ("none", "direct", "gradient"),
    ("none", "gradient", "direct"),
    ("direct", "none", "gradient"),
    ("direct", "gradient", "none"),
    ("gradient", "none", "direct"),
    ("gradient", "direct", "none"),
)
WARMUPS = (PERMUTATIONS[0], PERMUTATIONS[5])
SEEDS = (53101, 53102, 53103, 53104, 53105)


def make_schedule():
    sessions = []
    for index, seed in enumerate(SEEDS):
        rounds = list(PERMUTATIONS) * 2
        random.Random(seed).shuffle(rounds)
        sessions.append({"session": index, "seed": seed, "measured_rounds": rounds})
    return {"protocol": "gpu-tap-latency-repeated-v1",
            "generator_python_implementation": platform.python_implementation(),
            "generator_python_version": platform.python_version(),
            "warmup_rounds": WARMUPS, "sessions": sessions}


if __name__ == "__main__":
    print(json.dumps(make_schedule(), indent=2, sort_keys=True))
