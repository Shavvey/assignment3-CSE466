import math as m
from point import Point
import unittest


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


class TestDistances(unittest.TestCase):
    def test_euclidean(self):
        x = Point((3, 5))
        y = Point((6, 8))
        self.assertEqual(Distance.euclidean(x, y), m.sqrt(18))

    def test_manhattan(self):
        x = Point((3, 5))
        y = Point((6, 8))
        self.assertEqual(Distance.manhattan(x, y), 6)
