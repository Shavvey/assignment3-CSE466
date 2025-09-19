from point import Point
from knns import KNearestNeigbors, VotingType
from distance import Distance



def main():
    points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
    preds = KNearestNeigbors(points, Distance.euclidean, VotingType.DISTANCE, 10)
    for _, pred in enumerate(preds):
        print(f"{pred.label.capitalize()}")


if __name__ == "__main__":
    main()
