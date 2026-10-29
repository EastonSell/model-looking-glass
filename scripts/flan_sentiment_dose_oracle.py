"""Model-free, independent float32 dose oracle for the proposed FLAN study."""
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import unittest

helper_path = Path(__file__).resolve().parent / "bert_taste_matched_independent.py"
if not helper_path.is_file():
    # Development layout: the pinned helper is one directory above reviews/.
    helper_path = Path(__file__).resolve().parents[1] / "bert_taste_matched_independent.py"
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != "4acab808e72946a5196d268dbc6cd816deb1a5d6662a2d28bf0e3890087ca2d1":
    raise ValueError("independent Fraction-rounding helper changed")
spec = importlib.util.spec_from_file_location("independent_f32_helper", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def norm(values):
    result = math.sqrt(math.fsum(value ** 2 for value in values))
    if not math.isfinite(result):
        raise ValueError("nonfinite norm")
    return result


def distance(own_bits, donor_bits):
    if not own_bits or len(own_bits) != len(donor_bits):
        raise ValueError("vector sizes differ")
    return norm([helper.f32(b) - helper.f32(a) for a, b in zip(own_bits, donor_bits)])


def proposed(own_bits, donor_bits, target):
    d = distance(own_bits, donor_bits)
    if not math.isfinite(target) or not 0 < target <= d:
        raise ValueError("invalid nominal interpolation")
    fraction = target / d
    own = [helper.f32(bits) for bits in own_bits]
    donor = [helper.f32(bits) for bits in donor_bits]
    # Separate binary64 division, subtraction, multiplication and addition;
    # the helper performs exact-rational binary32 nearest-even rounding.
    post_bits = [helper.f32_round(a + fraction * (b - a)) for a, b in zip(own, donor)]
    actual_delta = [helper.f32(bits) - a for bits, a in zip(post_bits, own)]
    natural_delta = [b - a for a, b in zip(own, donor)]
    delivered = norm(actual_delta)
    cosine = (math.fsum(a * b for a, b in zip(actual_delta, natural_delta)) / (delivered * d)
              if delivered > 0 else None)
    tolerance = min(max(1e-5, 1e-5 * target), 0.01 * target)
    valid = (delivered > 0 and abs(delivered - target) <= tolerance and cosine is not None
             and math.isfinite(cosine) and cosine >= 0.99)
    return {"post_bits": post_bits, "distance": d, "fraction": fraction,
            "delivered": delivered, "cosine": cosine, "valid": valid}


def fabricated_group():
    """Produce all 16 directions and 32 landings from 512-wide fake states."""
    rng = random.Random(602101)
    states = [[[helper.f32_round(rng.uniform(-2.0, 2.0)) for _ in range(512)]
               for _ in range(3)] for _ in range(4)]
    # Sites: adjective, instruction and subject. Receiver order is
    # P1,N1,P2,N2; donor mappings are exact reciprocals.
    directions = []
    for receiver in range(4):
        for site, donor in ((0, receiver ^ 1), (0, receiver ^ 2),
                            (1, receiver ^ 1), (2, receiver ^ 1)):
            own, source = states[receiver][site], states[donor][site]
            directions.append((receiver, site, donor, own, source, distance(own, source)))
    radius = min(item[-1] for item in directions)
    if not radius / 2 > 1e-6:
        raise ValueError("fabricated radius is too small")
    landings = [[proposed(own, donor, radius / 2), proposed(own, donor, radius)]
                for _, _, _, own, donor, _ in directions]
    return {"states": states, "directions": directions, "radius": radius, "landings": landings}


class GeometryChecks(unittest.TestCase):
    def test_simple_equal_dose(self):
        own = [0, 0]
        donor = [helper.f32_round(4.0), 0]
        actual = proposed(own, donor, 1.0)
        self.assertEqual(actual["post_bits"], [helper.f32_round(1.0), 0])
        self.assertEqual((actual["fraction"], actual["delivered"], actual["cosine"]), (0.25, 1.0, 1.0))
        self.assertTrue(actual["valid"])

    def test_norm_alone_cannot_establish_direction(self):
        own = [helper.f32_round(1e8)] * 2
        donor = [helper.f32_round(1e8 + 8), helper.f32_round(1e8 + 56)]
        actual = proposed(own, donor, 8.0)
        self.assertEqual(actual["delivered"], 8.0)
        self.assertAlmostEqual(actual["cosine"], 7.0 / math.sqrt(50.0))
        self.assertFalse(actual["valid"])

    def test_positive_and_negative_subnormal_coordinates(self):
        own = [1, 0x80000001, 0]
        donor = [1, 0x80000001, helper.f32_round(1.0)]
        actual = proposed(own, donor, 0.5)
        self.assertEqual(actual["post_bits"], [1, 0x80000001, helper.f32_round(0.5)])
        self.assertTrue(actual["valid"])

    def test_zero_distance_cannot_be_normalized(self):
        with self.assertRaises(ValueError):
            proposed([0, 0], [0, 0], 1.0)

    def test_full_width_group_and_shared_radius(self):
        group = fabricated_group()
        self.assertEqual(len(group["directions"]), 16)
        self.assertEqual(sum(len(item) for item in group["landings"]), 32)
        self.assertEqual(group["radius"], min(item[-1] for item in group["directions"]))
        for pair in group["landings"]:
            for item in pair:
                self.assertEqual(len(item["post_bits"]), 512)
                self.assertTrue(item["valid"])


if __name__ == "__main__":
    unittest.main()
