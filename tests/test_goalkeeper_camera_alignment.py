import unittest

import numpy as np

from scripts import train_goalkeeper as goalkeeper


class GoalkeeperCameraAlignmentTest(unittest.TestCase):
    def test_head_camera_is_on_lens_side_and_faces_incoming_ball(self):
        sim = goalkeeper.setup_sim(render=False)
        sim.place_ball(-1.22, 0.0)

        alignment = goalkeeper.camera_alignment(sim)

        self.assertTrue(np.isclose(abs(sim.yaw()), np.pi))
        self.assertGreater(alignment["camera_to_ball_dot"], 0.95)
        self.assertGreater(alignment["camera_to_lens_side_dot"], 0.95)

    def test_ball_is_in_front_of_goalkeeper_not_behind_it(self):
        sim = goalkeeper.setup_sim(render=False)
        sim.place_ball(-1.22, 0.0)
        camera_id = sim.model.camera("head_camera").id
        camera_x = float(sim.data.cam_xpos[camera_id, 0])

        self.assertLess(sim.ball_xy()[0], camera_x)
        self.assertLess(camera_x, goalkeeper.GOAL_X)

    def test_goalkeeper_controller_moves_toward_positive_crossing(self):
        sim = goalkeeper.setup_sim(render=False)
        for step in range(150):
            command = goalkeeper.waypoint_command(
                sim, target_y=0.12, progress=step / 150, weights=np.empty((0, 0)))
            sim.control_step("walk", command)

        self.assertGreater(float(sim.data.qpos[1]), 0.08)
        self.assertLess(float(sim.data.qpos[1]), 0.15)


if __name__ == "__main__":
    unittest.main()
