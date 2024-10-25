import pandas as pd
import numpy as np
from .unit import Unit
from typing import List
from .air_object import AirObject


class AirEnv(Unit):

    def __init__(self, air_objects: List[AirObject] = None) -> None:
        super().__init__()
        self.__air_objects = dict()
        self.__air_object_id_next = 0

        if air_objects is not None:
            self.__air_objects = {i: air_objects[i] for i in range(len(air_objects))}
            self.__air_object_id_next = len(air_objects)

    def trigger(self) -> None:
        for ao in self.__air_objects.values():
            ao.trigger()

    def is_attached(self, air_object: AirObject) -> bool:
        return air_object in self.__air_objects.values()

    def attach_air_object(self, air_object: AirObject) -> int:
        if self.is_attached(air_object):
            raise RuntimeError('AirObject already attached to AirEnv.')
        self.__air_objects[self.__air_object_id_next] = air_object
        self.__air_object_id_next += 1
        return self.__air_object_id_next - 1

    def detach_air_object(self, air_object: AirObject) -> int:
        if not self.is_attached(air_object):
            raise RuntimeError('AirObject is not attached to AirEnv.')
        for k, v in self.__air_objects.items():
            if v == air_object:
                self.__air_objects.pop(k, None)
                return k

    def air_objects_dataframe(self) -> pd.DataFrame:
        """
        Для текущего момента модельного времени формируется таблица положений всех ВО
        :return: pd.DataFrame - таблица
        """
        data = pd.DataFrame(columns=['id', 'x_true', 'y_true', 'z_true'])
        for ao_id, ao in self.__air_objects.items():
            x = ao.position()[0]
            y = ao.position()[1]
            z = ao.position()[2]
            data.loc[len(data)] = {
                'id': ao_id,
                'x_true': x,
                'y_true': y,
                'z_true': z
            }
        return data