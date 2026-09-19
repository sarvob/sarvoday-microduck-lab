import unittest

import numpy as np

from scripts import evaluate_goalkeeper_camera_latency as latency


class GoalkeeperCameraLatencyTest(unittest.TestCase):
    def test_packets_are_not_delivered_early(self):
        queued = [(1.5, 0.5, np.array([1.0, 2.0, 3.0]))]
        delivered = []
        self.assertEqual(latency.deliver_packets(queued, 1.49, delivered), 0)
        self.assertEqual(len(queued), 1)
        self.assertEqual(latency.deliver_packets(queued, 1.5, delivered), 1)

    def test_capture_timestamp_survives_delivery(self):
        measured = np.array([1.0, 2.0, 3.0])
        queued = [(2.5, 0.5, measured)]
        delivered = []
        latency.deliver_packets(queued, 2.5, delivered)
        self.assertEqual(delivered[0][0], 0.5)
        self.assertTrue(np.array_equal(delivered[0][1], measured))


if __name__ == "__main__":
    unittest.main()
