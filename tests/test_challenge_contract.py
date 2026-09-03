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

    def test_challenge_006_defines_a_controlled_roll_and_handoff_search(self):
        path = ROOT / "challenges" / "006-controlled-roll" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        success = spec["success"]
        self.assertGreaterEqual(success["minimum_cumulative_body_rotation_deg"], 300)
        self.assertLessEqual(success["maximum_inverted_upright_score"], -0.8)
        self.assertGreaterEqual(success["minimum_final_upright_score"], 0.9)
        self.assertLessEqual(success["maximum_horizontal_displacement_m"], 0.25)
        self.assertGreaterEqual(len(spec["training"]["seeds"]), 3)
        self.assertIn("remains frozen", spec["disclosure"].lower())
        self.assertIn("handoff", spec["disclosure"].lower())

    def test_challenge_006_trained_handoff_passes_every_seed(self):
        result = json.loads(
            (ROOT / "artifacts" / "006-controlled-roll" / "result.json").read_text(
                encoding="utf-8"))
        self.assertTrue(result["success"])
        self.assertEqual(result["candidate_count"], 26)
        self.assertEqual(result["best"]["roll_steps"], 41)
        self.assertEqual(result["best"]["passing_seeds"], 3)
        self.assertTrue(all(
            row["horizontal_displacement_m"] <= 0.2
            for row in result["best"]["evaluations"]))

    def test_challenge_012_defines_variable_speed_boat_balance(self):
        path = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        profiles = spec["environment"]["profiles"]
        success = spec["success"]
        self.assertEqual(len(profiles), 3)
        self.assertGreaterEqual(spec["environment"]["motion_ramp_s"], 2.0)
        self.assertEqual(success["required_profiles"], 3)
        self.assertGreaterEqual(success["minimum_duration_per_profile_s"], 20.0)
        self.assertGreaterEqual(success["minimum_deck_contact_ratio"], 0.9)
        self.assertLessEqual(success["maximum_relative_deck_displacement_m"], 0.35)
        self.assertTrue(success["must_avoid_floor_contact"])
        self.assertTrue(success["must_remain_inside_deck_bounds"])
        self.assertTrue(spec["baseline"]["policy_frozen"])
        self.assertEqual(len(spec["training"]["learned_parameters"]), 8)
        self.assertIn("held-out", spec["disclosure"].lower())

    def test_challenge_012_baseline_is_measured_before_training(self):
        path = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "baseline.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["policy_frozen"])
        self.assertEqual(result["evaluation_count"], 9)
        self.assertFalse(result["success"])
        self.assertTrue(all(row["duration_s"] == 20.0 for row in result["evaluations"]))
        self.assertTrue(all("survival_time_s" in row for row in result["evaluations"]))


if __name__ == "__main__":
    unittest.main()
