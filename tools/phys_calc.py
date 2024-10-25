import numpy as np


class Physic:
    @staticmethod
    def calc_w(v: float, r: float):
        """
        Расчет угловой скорости рад / мс
        v - м / мс
        r - м
        """
        return v / r

    @staticmethod
    def convert_velocity(v: float):
        """
        V in meters/s
        return: V in meters/ms
        """
        return v / 1000

    @staticmethod
    def to_sphere_coord(x, y, z) -> tuple:
        """
        Перевод декартовой системы координат в сферическую
        :return: tuple
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        fi = np.arctan(y / x)
        psi = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
        return r, fi, psi

