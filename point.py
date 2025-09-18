import numpy.typing as npt
import numpy as np
import os as os
import csv as csv


class Point:
    dims: int
    coords: npt.NDArray[np.float32]
    label: str | None = None

    def __init__(self, coords: npt.ArrayLike, label: str | None = None):
        "Create a point in something that looks like a coordination system. An optional label can be provided"
        self.coords = np.array(coords, dtype=np.float32)
        self.dims = len(self.coords)
        if label != None:
            self.label = label

    def __eq__(self, other) -> bool:
        if self.dims != other.dims:
            return False
        return np.array_equal(other.coords, self.coords)

    def __str__(self) -> str:
        sb = "("
        if self.label != None:
            sb += f"{self.label}, "
        for i, coord in enumerate(self.coords):
            if i < len(self.coords) - 1:
                sb += f"{str(coord)}, "
            else:
                sb += f"{str(coord)}"
        sb += ")"
        return sb

    @staticmethod
    def points_from_iris_csv(csv_path: str) -> npt.NDArray:
        "Given data from iris csv, convert into list of labeled points"
        points: npt.NDArray = np.array([], dtype=Point)
        with open(csv_path, "r") as file:
            csv_reader = csv.reader(file)
            # consume header
            _ = next(csv_reader)
            for row in csv_reader:
                rlen = len(row)
                coords = [float(s) for s in row[0 : rlen - 1]]
                label = row[rlen - 1]
                if label == "":
                    p = Point(coords, None)
                else:
                    p = Point(coords, label=label)
                points = np.append(points, np.array(p))
        return points
