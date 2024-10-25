import numpy as np
import pandas as pd

from .unit import Unit
from .air_env import AirEnv
from tools import Physic


class RadarSystem(Unit):

    def __init__(self, position: np.array = np.array([0, 0, 0]), detection_radius: float = 10000, mean: float = 0.,
                 error: float = 1.,
                 air_env: AirEnv = None,
                 detection_fault_probability: float = 0.0, detection_period: int = 1,
                 detection_delay: int = 0) -> None:
        super().__init__()
        self.__position = np.array(position, dtype=float)
        self.__detection_fault_probability = detection_fault_probability
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period
        self.__detection_radius = detection_radius

        self.__mean = mean
        self.__error = error
        self.__air_env = air_env

        self.__data_dtypes = {
            'id': 'int64',
            'time': 'int64',
            'x_true': 'float64',
            'y_true': 'float64',
            'z_true': 'float64',
            'x_measure': 'float64',
            'y_measure': 'float64',
            'z_measure': 'float64',
            'r_true': 'float64',
            'fi_true': 'float64',
            'psi_true': 'float64',
            'r_measure': 'float64',
            'fi_measure': 'float64',
            'psi_measure': 'float64',
            'v_x_true_extr': 'float64',
            'v_y_true_extr': 'float64',
            'v_z_true_extr': 'float64',
            'v_r_true_extr': 'float64',
            'v_fi_true_extr': 'float64',
            'v_psi_true_extr': 'float64',
            'v_x_measure_extr': 'float64',
            'v_y_measure_extr': 'float64',
            'v_z_measure_extr': 'float64',
            'v_r_measure_extr': 'float64',
            'v_fi_measure_extr': 'float64',
            'v_psi_measure_extr': 'float64',
            'x_err': 'float64',
            'y_err': 'float64',
            'z_err': 'float64',
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
            lambda row: np.sqrt(
                (row['x_true'] - p[0]) ** 2 + (row['y_true'] - p[1]) ** 2 + (row['z_true'] - p[2]) ** 2) <= r,
            axis=1
        )
        detections = detections[detections['is_observed']]
        detections.drop(columns=['is_observed'], inplace=True)

        detections['time'] = self.time.get_time()
        detections['r_true'], detections['fi_true'], detections['psi_true'] = Physic.to_sphere_coord(
            detections['x_true'], detections['y_true'], detections['z_true'])
        detections['x_measure'] = detections['x_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        detections['y_measure'] = detections['y_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        detections['z_measure'] = detections['z_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        detections['r_measure'], detections['fi_measure'], detections['psi_measure'] = Physic.to_sphere_coord(
            detections['x_measure'], detections['y_measure'], detections['z_measure'])

        detections['x_err'] = self.__error
        detections['y_err'] = self.__error
        detections['z_err'] = self.__error

        # Выичисление скоростей
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
            detections[f'v_{coord}_extr'] = None if len(self.__data) == 0 else (detections[coord] -
                                                                                self.__data.iloc[len(self.__data) - 1][
                                                                                    coord]) / self.time.get_dt()

        # Concat new detections with data
        self.__concat_data(detections)

    def __concat_data(self, df: pd.DataFrame) -> None:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])
            self.__data.reset_index(inplace=True, drop=True)

    def get_data(self) -> pd.DataFrame:
        return self.__data.copy()

    def get_position(self):
        return self.__position

    def get_detection_radius(self):
        return self.__detection_radius

    def get_mean(self):
        return self.__mean

    def get_error(self):
        return self.__error

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
