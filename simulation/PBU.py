import numpy as np
import pandas as pd

from .unit import Unit
from .radar_system import RadarSystem
from typing import List


class PBU(Unit):
    def __init__(self, radars: List[RadarSystem] = None) -> None:
        super().__init__()
        self.__radars = dict()
        self.__radar_id_next = 0

        if radars is not None:
            self.__radars = {i: radars[i] for i in range(len(radars))}
            self.__radar_id_next = len(radars)

    def trigger(self) -> None:
        for radar in self.__radars.values():
            radar.trigger()


    def get_num_radars(self) -> int:
        return len(self.__radars)

    def get_data(self) -> List[pd.DataFrame]:
        return [radar.get_data() for radar in self.__radars.values()]

    def get_mean(self) -> List:
        return [radar.get_mean() for radar in self.__radars.values()]

    def get_errors(self) -> List:
        return [radar.get_error() for radar in self.__radars.values()]

    def get_radars_position(self) -> List[np.array]:
        return [radar.get_position() for radar in self.__radars.values()]

    def get_radars_detection_radius(self) -> List[np.array]:
        return [radar.get_detection_radius() for radar in self.__radars.values()]

