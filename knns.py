from types import FunctionType
from enum import Enum
import numpy as np
import numpy.typing as npt
from point import Point
from distance import Distance

type PointDistance = tuple[Point, float]

class VotingType(Enum):
    MAJORITY = 1
    DISTANCE = 2

def main():
    points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
    preds = KNearestNeigbors(points, Distance.Euclidean, VotingType.MAJORITY, 5)
    for i, pred in enumerate(preds):
        print(f"{i}: {pred}")


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
        distances = make_distances(pred, ldata, dist, k)
        label = make_prediction(vtype, distances)
        pred.label = label
    return preds


def make_distances(
    pred: Point, ldata: npt.NDArray, dist: FunctionType, k: int
) -> list[PointDistance]:
    "Given a prediction point, make a list of minimum distances to `k` points the data"
    # find all distances to labeled points
    pdist = []
    for ld in ldata:
        distance = dist(ld, pred)
        pdist.append((ld, distance))
    pdist.sort(key=lambda x: x[1])
    return pdist[0:k]

def make_prediction(vtype: VotingType, distances: list[PointDistance]) -> str:
    # map each label to voting power
    match vtype:
        case VotingType.MAJORITY:
            vote_map: dict[str, int] = {}
            for pdist in distances:
                label = pdist[0].label
                if label != None:
                    if vote_map.get(label) == None:
                        vote_map[label] = 1
                    else:
                        vote_map[label] += 1
            pred = sorted(vote_map, key=lambda x: x[0])[0]
            return pred
        case _:
            raise ValueError("[ERROR]: Voting type not implemented!")

if __name__ == "__main__":
    main()
