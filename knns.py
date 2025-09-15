from types import FunctionType
from enum import Enum
import numpy as np
import numpy.typing as npt
from point import Point

type PointDistance = tuple[Point, float]

class VotingType(Enum):
    MAJORITY = 1
    DISTANCE = 2

def KNearestNeigbors(
    points: npt.NDArray, dist: FunctionType, vtype: VotingType, k: int, reuse: bool | None = None
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
        if reuse == True:
            # reuse predicted point by appending to train data
            ldata = np.append(ldata, [pred])
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
            vote_mapm: dict[str, int] = {}
            for pdist in distances:
                label = pdist[0].label
                if label != None:
                    # here voting power is simple majority, each label gets equal weighting
                    if vote_mapm.get(label) == None:
                        vote_mapm[label] = 1
                    else:
                        vote_mapm[label] += 1
            # voting result in descending order, use label with largest voting power
            preds = sorted(vote_mapm.items(), key=lambda x: x[1], reverse=True)
            return preds[0][0]
        case VotingType.DISTANCE:
            vote_mapd: dict[str, float] = {}
            for pdist in distances:
                label = pdist[0].label
                if label != None:
                    # here voting power is determined by inverse square of distance
                    if vote_mapd.get(label) == None:
                        vote_mapd[label] = 1/(pdist[1]**2)
                    else:
                        vote_mapd[label] += 1/(pdist[1]**2)
            # voting result in descending order, use label with largest voting power
            preds = sorted(vote_mapd.items(), key=lambda x: x[1], reverse=True)
            return preds[0][0]
            
        case _:
            raise ValueError("[ERROR]: Voting type not implemented!")

