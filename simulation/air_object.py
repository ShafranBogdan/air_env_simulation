from .unit import Unit
from .trajectory import Trajectory
from typing import Callable
import numpy as np
import pandas as pd

class AirObject(Unit):
    def __init__(self, track: Trajectory) -> None:
        super().__init__()

        if track.get_position(self.time.get_time()).shape != (3,):
            raise RuntimeError(f'Track function should return numpy array with (3,) shape.')

        self.__track = track
        self.__data = pd.DataFrame(columns=['x_true', 'y_true', 'z_true'])

    def trigger(self) -> None:
        self.__data.loc[len(self.__data)] = {
            'x_true': self.position()[0],
            'y_true': self.position()[1],
            'z_true': self.position()[2]
        }

    def position(self) -> np.array:
        return list(map(float, self.__track.get_position(self.time.get_time())))

    def get_data(self) -> pd.DataFrame:
        return self.__data.copy()
