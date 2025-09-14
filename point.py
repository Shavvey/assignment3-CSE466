import numpy.typing as npt
import numpy as np
import os as os

class Point:
    dims: int
    # NOTE: could use generics, but it's just easier to make everything a float by default
    coords: npt.NDArray[np.float32]

    def __init__(self, coords: npt.ArrayLike):
        "Create a point in something that looks like a coordination system"
        self.coords = np.array(coords, dtype = np.float32)
        self.dims = len(self.coords)

    @staticmethod
    def points_from_csv(csv_path: os.PathLike) -> npt.NDArray:
        "Given a csv file, convert into list of points"
        with open(csv_path, "r") as file:
            for line in file:
                        
     
    
