import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ChallengeContractTest(unittest.TestCase):
    def test_challenge_001_is_bounded_and_measurable(self):
        path = ROOT / "challenges" / "001-spin-in-place" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["lesson"]["task"], "spin")
        self.assertGreaterEqual(spec["success"]["minimum_turns"], 1.0)
        self.assertLessEqual(spec["success"]["maximum_drift_m"], 0.5)
        self.assertTrue(spec["success"]["must_stay_upright"])
        self.assertGreaterEqual(len(spec["training"]["seeds"]), 3)


if __name__ == "__main__":
    unittest.main()
