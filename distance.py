import numpy.typing as npt
import math as m


class Distance:
    @staticmethod
    def Euclidean(x: npt.NDArray, y: npt.NDArray) -> float:
        assert len(x) == len(y)  # vectors must have the same dims
        sum = 0
        return sum
