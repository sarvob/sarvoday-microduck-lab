import unittest

from scripts import evaluate_goalkeeper_camera_frame_loss as loss


class GoalkeeperCameraFrameLossTest(unittest.TestCase):
    def test_loss_mask_is_reproducible(self):
        first = loss.frame_scores(487, 15, 15015)
        second = loss.frame_scores(487, 15, 15015)
        self.assertEqual(first.tolist(), second.tolist())

    def test_higher_loss_keeps_subset(self):
        scores = loss.frame_scores(491, 15, 15015)
        kept_25 = {i for i in range(len(scores)) if loss.keep_frame(scores, i, 0.25)}
        kept_75 = {i for i in range(len(scores)) if loss.keep_frame(scores, i, 0.75)}
        self.assertTrue(kept_75 < kept_25)

    def test_zero_loss_keeps_every_frame(self):
        scores = loss.frame_scores(499, 15, 15015)
        self.assertTrue(all(loss.keep_frame(scores, i, 0.0) for i in range(len(scores))))


if __name__ == "__main__":
    unittest.main()
