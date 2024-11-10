import numpy as np
import pandas as pd

from .unit import Unit
from .air_env import AirEnv


class RadarSystem(Unit):

    def __init__(self, position: np.array=np.array([0, 0, 0]), detection_radius: float=10000, error_r: float=5., error_fi: float=0.0001, 
                 error_psi: float=0.0001, air_env: AirEnv = None,
                 detection_fault_probability: float = 0., detection_period: int = 100,
                 detection_delay: int = 0) -> None:
        super().__init__()

        self.__detection_fault_probability = detection_fault_probability
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period

        self.__position = np.array(position, dtype=float)
        self.__detection_radius = detection_radius
        self.__error_r = error_r
        self.__error_fi = error_fi
        self.__error_psi = error_psi

        self.__air_env = air_env

        self.__data_dtypes = {
            'id': 'int64',
            'time': 'int64',
            'x_true': 'float64',
            'y_true': 'float64',
            'z_true': 'float64',
            'x_measure' : 'float64',
            'y_measure' : 'float64',
            'z_measure' : 'float64',
            'r_true' : 'float64',
            'fi_true' : 'float64',
            'psi_true' : 'float64',
            'r_measure' : 'float64',
            'fi_measure' : 'float64',
            'psi_measure' : 'float64',
            'v_x_true': 'float64',
            'v_y_true': 'float64',
            'v_z_true': 'float64',
            'v_r_true': 'float64',
            'v_fi_true': 'float64',
            'v_psi_true': 'float64',
            'v_x_measure': 'float64',
            'v_y_measure': 'float64',
            'v_z_measure': 'float64',
            'v_r_measure': 'float64',
            'v_fi_measure': 'float64',
            'v_psi_measure': 'float64',
            'r_err': 'float64',
            'fi_err': 'float64',
            'psi_err': 'float64',
        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)

    def trigger(self) -> None:
        if self.time.get_time() % self.__detection_period == self.__detection_delay:
            if np.random.choice([False, True],
                                p=[self.__detection_fault_probability, 1.0 - self.__detection_fault_probability]):
                self.detect_air_objects()

    def detect_air_objects(self) -> None:
        # Получение положений всех ВО в наблюдаемой AirEnv
        detections = self.__air_env.air_objects_dataframe()

        # Фильтрация ВО с координатами вне области наблюдения
        p = self.__position
        r = self.__detection_radius
        detections['is_observed'] = detections.apply(
            lambda row: np.sqrt((row['x_true'] - p[0]) ** 2 + (row['y_true'] - p[1]) ** 2 + (row['z_true'] - p[2]) ** 2) <= r,
            axis=1
        )
        detections = detections[detections['is_observed']]
        detections.drop(columns=['is_observed'], inplace=True)

        detections['time'] = self.time.get_time()
        detections['r_true'], detections['fi_true'], detections['psi_true'] = self.__to_sphere_coord(detections['x_true'], detections['y_true'], detections['z_true'])

        detections['r_measure'] = detections['r_true'] + np.random.normal(0, self.__error_r, len(detections))
        detections['fi_measure'] = detections['fi_true'] + np.random.normal(0, self.__error_fi, len(detections))
        detections['psi_measure'] = detections['psi_true'] + np.random.normal(0, self.__error_psi, len(detections))

        detections['x_measure'], detections['y_measure'], detections['z_measure'] = self.__to_cartesian_coord(detections['r_measure'], detections['fi_measure'], detections['psi_measure'])
        
        detections['r_err'] = self.__error_r
        detections['fi_err'] = self.__error_fi
        detections['psi_err'] = self.__error_psi

        # Вычисление скоростей
        for coord in (
            'x_true', 
            'y_true', 
            'z_true', 
            'x_measure',
            'y_measure',
            'z_measure',
            'r_true',
            'fi_true',
            'psi_true',
            'r_measure', 
            'fi_measure', 
            'psi_measure'
        ):
            detections[f'v_{coord}'] = None if len(self.__data) == 0 else (detections[coord] - self.__data.iloc[len(self.__data) - 1][coord]) / (detections['time'] - self.__data.iloc[len(self.__data) - 1]['time'])

        # Concat new detections with data
        self.__concat_data(detections)

    def __to_cartesian_coord(self, r, fi, psi) -> tuple:
        """
        Перевод сферической системы координат в декартову
        :return: tuple = (x, y, z)
        """
        x = r * np.sin(psi) * np.cos(fi)
        y = r * np.sin(psi) * np.sin(fi)
        z = r * np.cos(psi)
        return (x, y, z)

    def __to_sphere_coord(self, x, y, z) -> tuple:
        """
        Перевод декартовой системы координат в сферическую
        :return: tuple
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        fi = np.atan2(y, x)
        psi = np.arccos(z / r)
        return (r, fi, psi)

    def __concat_data(self, df: pd.DataFrame) -> None:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])
            self.__data.reset_index(inplace=True, drop=True)

    def get_data(self) -> pd.DataFrame:
        return self.__data.copy()

    def clear_data(self) -> None:
        self.__data = self.__data.iloc[0:0]

    def set_air_environment(self, air_env: AirEnv) -> None:
        self.__air_env = air_env

    def set_detection_fault_probability(self, detection_fault_probability: float) -> None:
        self.__detection_fault_probability = detection_fault_probability

    def set_detection_period(self, detection_period: int) -> None:
        self.__detection_period = detection_period

    def repr(self) -> str:
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )