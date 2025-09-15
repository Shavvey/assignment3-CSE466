import unittest
from distance import Distance
from knns import KNearestNeigbors, VotingType
from point import Point
import numpy as np


class TestKNearestNeighbors(unittest.TestCase):
    # first, test w/ k = 1
    def test_knn_1(self):
        p1 = Point((5, 5), "1")
        p2 = Point((10, 10), "2")
        p3 = Point((0, 0), "3")
        pred1 = Point((1, 1))
        pred2 = Point((9, 9))
        pred3 = Point((4, 4))
        points = np.array([p1, p2, p3, pred1, pred2, pred3], dtype=Point)
        preds = KNearestNeigbors(points, Distance.euclidean, VotingType.MAJORITY, 1)
        labels: list[str] = []
        for pred in preds:
            labels.append(pred.label)
        self.assertEqual(labels, ["3", "2", "1"])


if __name__ == "__main__":
    unittest.main()
