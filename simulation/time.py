import time


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class Time:
    """
    Время задается в ms
    Стандартны шаг dt = 1 s
    """
    def __init__(self, dt: int = 1) -> None:
        print(f'Time initialized: 0s')
        self.__t = 0
        self.__dt = dt

    def get_time(self) -> int:
        return self.__t

    def get_dt(self) -> int:
        return self.__dt

    def set_dt(self, dt) -> None:
        self.__dt = dt

    def set(self, t: int) -> None:
        print(f'Time set: {self.__t} s -> {t} s')
        self.__t = t

    def step(self) -> None:
        self.__t += self.__dt