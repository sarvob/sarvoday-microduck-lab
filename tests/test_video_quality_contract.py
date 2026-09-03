import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VideoQualityContractTest(unittest.TestCase):
    def test_high_resolution_daily_output_contract(self):
        profile = json.loads(
            (ROOT / "stream" / "video-quality-profile.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(profile["daily_publish_target"], 3)
        self.assertGreaterEqual(profile["youtube_minimum_duration_seconds"], 180)
        self.assertEqual(profile["youtube_format"], "normal educational landscape video")
        self.assertFalse(profile["youtube_shorts_allowed"])
        self.assertGreaterEqual(profile["landscape"]["width"], 2560)
        self.assertGreaterEqual(profile["landscape"]["height"], 1440)
        self.assertGreaterEqual(profile["vertical_short"]["width"], 1440)
        self.assertGreaterEqual(profile["vertical_short"]["height"], 2560)
        self.assertFalse(profile["source_rules"]["upscaled_720p_allowed"])
        self.assertTrue(profile["source_rules"]["full_playback_qa_required"])
        self.assertTrue(profile["source_rules"]["moving_simulator_footage_required"])
        self.assertFalse(profile["source_rules"]["runtime_padding_allowed"])
        self.assertGreaterEqual(profile["minimum_free_disk_gb_for_rendering"], 20)


if __name__ == "__main__":
    unittest.main()
