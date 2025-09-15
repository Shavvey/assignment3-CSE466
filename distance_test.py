from types import FunctionType
import unittest
from distance import Distance
from point import Point
import math as m


class TestDistances(unittest.TestCase):
    def test_euclidean(self):
        x = Point((3, 5))
        y = Point((6, 8))
        self.assertEqual(Distance.euclidean(x, y), m.sqrt(18))

    def test_manhattan(self):
        x = Point((3, 5))
        y = Point((6, 8))
        self.assertEqual(Distance.manhattan(x, y), 6)

# unit test just to ensure passing distance as functiontype works as expected
class TestFunctionType(unittest.TestCase):
    @staticmethod
    def distfunc(dist: FunctionType, x: Point, y: Point) -> float:
        return dist(x, y)

    def test_distfunc(self):
        x = Point((3, 5))
        y = Point((6, 8))
        res_euclid = TestFunctionType.distfunc(Distance.euclidean, x, y)
        self.assertEqual(res_euclid, m.sqrt(18))


if __name__ == "__main__":
    unittest.main()
