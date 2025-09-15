import math as m
from point import Point


class Distance:
    @staticmethod
    def euclidean(x: Point, y: Point) -> float:
        assert x.dims == y.dims
        dist = lambda x, y: (x - y) ** 2
        sum = 0
        for i in range(x.dims):
            sum += dist(x.coords[i], y.coords[i])
        return m.sqrt(sum)

    @staticmethod
    def manhattan(x: Point, y: Point) -> float:
        assert x.dims == y.dims
        dist = lambda x, y: abs(x - y)
        sum = 0
        for i in range(x.dims):
            sum += dist(x.coords[i], y.coords[i])
        return sum
