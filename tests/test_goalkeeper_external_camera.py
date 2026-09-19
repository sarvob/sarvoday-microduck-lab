import unittest

import numpy as np

from scripts import evaluate_goalkeeper_external_camera as external


class ExternalGoalkeeperCameraTest(unittest.TestCase):
    def test_screen_is_visual_only(self):
        sim = external.setup_sim()
        geom = sim.model.geom("vision_screen")
        self.assertEqual(int(geom.contype[0]), 0)
        self.assertEqual(int(geom.conaffinity[0]), 0)

    def test_external_camera_estimate_comes_from_pixels(self):
        sim = external.setup_sim()
        sim.place_ball(-1.0, 0.18)
        sim.renderer.update_scene(sim.data, camera=external.EXTERNAL_CAMERA, scene_option=sim.opt)
        estimate = external.visual_ball_position(
            sim, sim.renderer.render(), external.EXTERNAL_CAMERA)
        self.assertIsNotNone(estimate)
        self.assertTrue(np.allclose(estimate[:2], sim.ball_xy(), atol=0.035))


if __name__ == "__main__":
    unittest.main()
