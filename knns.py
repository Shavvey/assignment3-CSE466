import numpy as np
import numpy.typing as npt
from point import Point


def main():
    print("Hello, World!")
    points = Point.points_from_iris_csv("data/iris_rnd_train.csv")
    for point in points:
        print(point)


class KNearestNeigbors:
    data = []

    def __init__(self, data: npt.NDArray):
        self.data = data


if __name__ == "__main__":
    main()
