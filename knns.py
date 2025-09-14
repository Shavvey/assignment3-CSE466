from types import FunctionType
from enum import Enum
import numpy as np
import numpy.typing as npt
from point import Point
from distance import Distance

class VotingType(Enum):
    MAJORITY = 1
    DISTANCE = 2

def main():
    points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
    preds = KNearestNeigbors(points, Distance.Euclidean, VotingType.MAJORITY, 5)


def KNearestNeigbors(
    points: npt.NDArray, dist: FunctionType, vtype: VotingType, k: int
) -> npt.NDArray:
    "Given a set of points. Make predictions for the unlabeled points using kNNs"
    preds = np.array([], dtype=Point)
    ldata = np.array([], dtype=Point)
    # collect data points in predictions and labeled data
    for point in points:
        if point.label == None:
            preds = np.append(preds, [point])
        else:
            ldata = np.append(ldata, [point])
    # create predictions for each unlabeled piece of data
    for pred in preds:
        distances = (pred, ldata, dist, k)
    return preds


def make_distances(
    pred: Point, ldata: npt.NDArray, dist: FunctionType, k: int
) -> list[tuple[Point, float]]:
    # find all distances to labeled points
    pdist = []
    for ld in ldata:
        distance = dist(ld, pred)
        pdist.append((ld, distance))
    pdist.sort(key=lambda x: x[1])
    return pdist[0:k]

if __name__ == "__main__":
    main()
