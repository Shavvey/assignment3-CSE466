import unittest
from distance import Distance
from knns import KNearestNeigbors, VotingType
from point import Point
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random


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

    def test_knn_2(self):
        ks = 10
        for _ in range(ks):
            if ks % 2 != 0:
                labels = [str(k) for k in range(ks-1)]
                skPoints = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(30)]
                skLabels = [random.choice(labels) for _ in range(30)]
                points = [Point(point, label) for point, label in zip(skPoints, skLabels)]

                pred_points = [Point((1, 1)), Point((9, 9)), Point((4, 4))]
                points.extend(pred_points)
                points = np.array(points, dtype=Point)
                skPred = [(1, 1), (9, 9), (4, 4)]
                preds = KNearestNeigbors(points, Distance.euclidean, VotingType.MAJORITY, ks)
                predLabels = []

                for pred in preds:
                    predLabels.append(pred.label)
                knn = KNeighborsClassifier(n_neighbors=ks, weights="uniform", metric="euclidean")
                knn.fit(skPoints, skLabels)
                skPredLabels = knn.predict(skPred)
                np.testing.assert_array_equal(skPredLabels, predLabels)


if __name__ == "__main__":
    unittest.main()
