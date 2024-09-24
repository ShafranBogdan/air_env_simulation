from .unit import Unit
from typing import Callable
import numpy as np

class AirObject(Unit):
    def __init__(self, track: Callable[[int], np.array]) -> None:
        super().__init__()

        if track(self.time.get_time()).shape != (3,):
            raise RuntimeError(f'Track function should return numpy array with (3,) shape.')

        self.__track = track

    def trigger(self) -> None:
        pass

    def position(self) -> np.array:
        return list(map(float, self.__track(self.time.get_time())))
