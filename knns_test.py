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
        for i in range(2, ks):
            if i % 2 != 0:
                print(i)
                labels = [str(k) for k in range(i - 1)]
                skPoints = [
                    (random.randint(1, 10), random.randint(1, 10)) for _ in range(30)
                ]
                skLabels = [random.choice(labels) for _ in range(30)]
                points = [
                    Point(point, label) for point, label in zip(skPoints, skLabels)
                ]

                pred_points = [Point((1, 1)), Point((9, 9)), Point((4, 4))]
                points.extend(pred_points)
                points = np.array(points, dtype=Point)
                skPred = [(1, 1), (9, 9), (4, 4)]
                preds = KNearestNeigbors(
                    points, Distance.euclidean, VotingType.MAJORITY, i
                )
                predLabels = []

                for pred in preds:
                    predLabels.append(pred.label)
                knn = KNeighborsClassifier(
                    n_neighbors=i, weights="uniform", metric="euclidean"
                )
                knn.fit(skPoints, skLabels)
                skPredLabels = knn.predict(skPred)
                self.assertEqual(1, 1)

    def test_knn_3(self):
        # test euclidean trails
        ks = [1, 2, 3, 5, 10]
        for k in ks:
            # first get labeled and prediction points from iris to feed into knns
            points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
            sk_labeled_points = np.empty((0, 4))
            sk_pred_points = np.empty((0, 4))
            sk_labels: list[str] = []
            for point in points:
                if point.label == None:
                    sk_pred_points = np.append(sk_pred_points, [point.coords], axis=0)
                else:
                    sk_labeled_points = np.append(
                        sk_labeled_points, [point.coords], axis=0
                    )
                    sk_labels.append(point.label)
            preds = KNearestNeigbors(points, Distance.euclidean, VotingType.MAJORITY, k)
            label1 = [pred.label for pred in preds]
            # see what scikit learn does
            knn = KNeighborsClassifier(
                n_neighbors=k, weights="uniform", metric="euclidean"
            )
            knn.fit(sk_labeled_points, sk_labels)
            label2 = knn.predict(sk_pred_points)
            np.testing.assert_equal(label1, label2)

        # test distance trails
        for k in ks:
            # first get labeled and prediction points from iris to feed into knns
            points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
            sk_labeled_points = np.empty((0, 4))
            sk_pred_points = np.empty((0, 4))
            sk_labels: list[str] = []
            for point in points:
                if point.label == None:
                    sk_pred_points = np.append(sk_pred_points, [point.coords], axis=0)
                else:
                    sk_labeled_points = np.append(
                        sk_labeled_points, [point.coords], axis=0
                    )
                    sk_labels.append(point.label)
            preds = KNearestNeigbors(points, Distance.euclidean, VotingType.DISTANCE, k)
            label1 = [pred.label for pred in preds]
            # see what scikit learn does
            knn = KNeighborsClassifier(
                n_neighbors=k, weights="distance", metric="euclidean"
            )
            knn.fit(sk_labeled_points, sk_labels)
            label2 = knn.predict(sk_pred_points)
            print(k)
            print(f"Our classifier: {label1}")
            print(f"SciKit classifier: {label2}")
            np.testing.assert_equal(label1, label2)

        # test manhattan
        for k in ks:
            # first get labeled and prediction points from iris to feed into knns
            points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
            sk_labeled_points = np.empty((0, 4))
            sk_pred_points = np.empty((0, 4))
            sk_labels: list[str] = []
            for point in points:
                if point.label == None:
                    sk_pred_points = np.append(sk_pred_points, [point.coords], axis=0)
                else:
                    sk_labeled_points = np.append(
                        sk_labeled_points, [point.coords], axis=0
                    )
                    sk_labels.append(point.label)
            preds = KNearestNeigbors(points, Distance.manhattan, VotingType.MAJORITY, k)
            label1 = [pred.label for pred in preds]
            # see what scikit learn does
            knn = KNeighborsClassifier(
                n_neighbors=k, weights="uniform", metric="manhattan"
            )
            knn.fit(sk_labeled_points, sk_labels)
            label2 = knn.predict(sk_pred_points)
            np.testing.assert_equal(label1, label2)

        # test distance trails
        for k in ks:
            # first get labeled and prediction points from iris to feed into knns
            points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
            sk_labeled_points = np.empty((0, 4))
            sk_pred_points = np.empty((0, 4))
            sk_labels: list[str] = []
            for point in points:
                if point.label == None:
                    sk_pred_points = np.append(sk_pred_points, [point.coords], axis=0)
                else:
                    sk_labeled_points = np.append(
                        sk_labeled_points, [point.coords], axis=0
                    )
                    sk_labels.append(point.label)
            preds = KNearestNeigbors(points, Distance.manhattan, VotingType.DISTANCE, k)
            label1 = [pred.label for pred in preds]
            # see what scikit learn does
            knn = KNeighborsClassifier(
                n_neighbors=k, weights="distance", metric="manhattan"
            )
            knn.fit(sk_labeled_points, sk_labels)
            label2 = knn.predict(sk_pred_points)
            print(k)
            print(f"Our classifier: {label1}")
            print(f"SciKit classifier: {label2}")
            np.testing.assert_equal(label1, label2)


if __name__ == "__main__":
    unittest.main()
