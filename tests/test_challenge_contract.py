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

    def test_challenge_002_requires_both_markers_and_balance(self):
        path = ROOT / "challenges" / "002-two-marker-sprint" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["lesson"]["task"], "goto")
        self.assertEqual(len(spec["lesson"]["targets"]), 2)
        self.assertEqual(spec["success"]["minimum_markers"], 2)
        self.assertTrue(spec["success"]["must_stay_upright"])

    def test_challenge_003_requires_object_displacement(self):
        path = ROOT / "challenges" / "003-ball-push" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["lesson"]["task"], "ball")
        self.assertGreaterEqual(spec["success"]["minimum_ball_m"], 0.25)
        self.assertTrue(spec["success"]["must_stay_upright"])

    def test_challenge_004_discloses_learned_and_scripted_parts(self):
        path = ROOT / "challenges" / "004-duck-swing-team" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(spec["success"]["minimum_peak_angle_deg"], 30)
        self.assertGreaterEqual(spec["success"]["minimum_sustained_peaks"], 6)
        self.assertIn("learned", spec["disclosure"].lower())
        self.assertIn("deterministic", spec["disclosure"].lower())

    def test_challenge_005_requires_a_real_jump_and_stable_two_foot_landing(self):
        path = ROOT / "challenges" / "005-duck-quadruped-jump" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        success = spec["success"]
        self.assertEqual(spec["quadruped"]["model"], "Unitree Go1")
        self.assertGreaterEqual(len(spec["training"]["seeds"]), 3)
        self.assertEqual(success["required_feet_on_back"], 2)
        self.assertGreaterEqual(success["minimum_airborne_time_s"], 0.1)
        self.assertGreaterEqual(success["minimum_hold_time_s"], 1.0)
        self.assertTrue(success["must_avoid_ground_contact_after_landing"])
        self.assertTrue(success["must_keep_quadruped_upright"])
        self.assertIn("fixed standing pose", spec["disclosure"].lower())
        self.assertFalse(spec["feasibility"]["unassisted_jump_feasible"])
        self.assertFalse(spec["assistance"]["midair_external_force_allowed"])
        self.assertIn("stated on-screen", spec["disclosure"].lower())

    def test_challenge_005_names_the_pinned_quadruped_asset(self):
        script = (ROOT / "scripts" / "validate_go1_platform.py").read_text(encoding="utf-8")
        self.assertIn("MENAGERIE_REVISION", script)
        self.assertIn("unitree_go1", script)

    def test_challenge_005_calibration_reaches_pad_but_not_hold_gate(self):
        path = ROOT / "artifacts" / "005-duck-quadruped-jump" / "launch-calibration.json"
        calibration = json.loads(path.read_text(encoding="utf-8"))
        best = calibration["best"]
        self.assertFalse(calibration["success_gate_met"])
        self.assertTrue(best["both_feet_simultaneous"])
        self.assertFalse(best["ground_contact_after_pad"])
        self.assertLess(best["longest_untrained_hold_s"], 1.5)
        self.assertGreater(best["longest_untrained_hold_s"], 0.5)

    def test_challenge_005_trained_policy_passes_every_seed(self):
        path = ROOT / "artifacts" / "005-duck-quadruped-jump" / "landing-result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["success"])
        self.assertEqual(result["evaluation_seeds"], [17, 71, 173])
        self.assertTrue(all(row["both_feet_simultaneous"] for row in result["evaluations"]))
        self.assertTrue(all(row["longest_hold_s"] >= 1.5 for row in result["evaluations"]))
        self.assertTrue(all(not row["ground_contact_after_pad"] for row in result["evaluations"]))


if __name__ == "__main__":
    unittest.main()
