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

if __name__ == "__main__":
    unittest.main()
