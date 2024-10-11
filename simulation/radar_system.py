import numpy as np
import pandas as pd

from .unit import Unit
from .air_env import AirEnv


class RadarSystem(Unit):

    def __init__(self, position: np.array=np.array([0, 0, 0]), detection_radius: float=10000, error: float=5., air_env: AirEnv = None,
                 detection_fault_probability: float = 0., detection_period: int = 100,
                 detection_delay: int = 0) -> None:
        super().__init__()

        self.__detection_fault_probability = detection_fault_probability
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period

        self.__position = np.array(position, dtype=float)
        self.__detection_radius = detection_radius
        self.__error = error
        self.__r_error, self.__fi_error, self.__psi_error = self.__to_sphere_coord(error, error, error)
        print(f'Sphere errors = {self.__r_error, self.__fi_error, self.__psi_error}')
        self.__air_env = air_env

        self.__data_dtypes = {
            'is_observed' : 'bool',
            'id': 'int64',
            'time': 'int64',
            'x_true': 'float64',
            'y_true': 'float64',
            'z_true': 'float64',
            'x_measure' : 'float64',
            'y_measure' : 'float64',
            'z_measure' : 'float64',
            'x_measure_extr' : 'float64',
            'y_measure_extr' : 'float64',
            'z_measure_extr' : 'float64',
            'r_true' : 'float64',
            'fi_true' : 'float64',
            'psi_true' : 'float64',
            'r_measure' : 'float64',
            'fi_measure' : 'float64',
            'psi_measure' : 'float64',
            'r_measure_extr' : 'float64',
            'fi_measure_extr' : 'float64',
            'psi_measure_extr' : 'float64',
            'r_measure_smooth' : 'float64',
            'fi_measure_smooth' : 'float64',
            'psi_measure_smooth' : 'float64',
            'v_x_true_extr': 'float64',
            'v_y_true_extr': 'float64',
            'v_z_true_extr': 'float64',
            'v_x_true': 'float64',
            'v_y_true': 'float64',
            'v_z_true': 'float64',
            'v_r_true_extr': 'float64',
            'v_fi_true_extr': 'float64',
            'v_psi_true_extr': 'float64',
            'v_r_true': 'float64',
            'v_fi_true': 'float64',
            'v_psi_true': 'float64',
            'v_x_measure_extr': 'float64',
            'v_y_measure_extr': 'float64',
            'v_z_measure_extr': 'float64',
            'v_x_measure': 'float64',
            'v_y_measure': 'float64',
            'v_z_measure': 'float64',
            'v_r_measure_extr': 'float64',
            'v_fi_measure_extr': 'float64',
            'v_psi_measure_extr': 'float64',
            'v_r_measure': 'float64',
            'v_fi_measure': 'float64',
            'v_psi_measure': 'float64',
            'v_r_measure_smooth': 'float64',
            'v_fi_measure_smooth': 'float64',
            'v_psi_measure_smooth': 'float64',
            'x_err': 'float64',
            'y_err': 'float64',
            'z_err': 'float64',
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
        # detections = detections[detections['is_observed']]
        # detections.drop(columns=['is_observed'], inplace=True)

        detections['time'] = self.time.get_time()
        detections['r_true'], detections['fi_true'], detections['psi_true'] = self.__to_sphere_coord(detections['x_true'], detections['y_true'], detections['z_true'])
        detections['x_measure'] = detections['x_true'] + np.random.normal(0, self.__error, len(detections))
        detections['y_measure'] = detections['y_true'] + np.random.normal(0, self.__error, len(detections))
        detections['z_measure'] = detections['z_true'] + np.random.normal(0, self.__error, len(detections))
        detections['r_measure'], detections['fi_measure'], detections['psi_measure'] = self.__to_sphere_coord(detections['x_measure'], detections['y_measure'], detections['z_measure'])
        
        detections['x_err'] = self.__error
        detections['y_err'] = self.__error
        detections['z_err'] = self.__error
        detections['r_err'], detections['fi_err'], detections['psi_err'] = self.__to_sphere_coord(detections['x_err'], detections['y_err'], detections['z_err'])
        
        prev_detect = None
        if len(self.__data) != 0: 
            air_objects_count = self.__air_env.get_air_objects_count()
            prev_detect = self.__data.tail(air_objects_count)
            #print(f'prev_detect = {prev_detect}')
            prev_detect = prev_detect.set_index(prev_detect['id'])
            #print(f'prev_detect_set = {prev_detect}')

        for coord in (
            'x_measure',
            'y_measure',
            'z_measure',
            'r_measure',
            'fi_measure',
            'psi_measure'
        ):
            # print(f'Prev detect = {None if prev_detect is None or np.isnan(prev_detect[f'v_{coord}_extr']) else prev_detect[f'v_{coord}_extr'] }')
            if prev_detect is None or prev_detect[f'v_{coord}_extr'].isna().any():
                detections[f'{coord}_extr'] = detections[f'{coord}']
            else:
                if f'{coord}_smooth' not in prev_detect.columns or f'v_{coord}_smooth' not in prev_detect.columns or prev_detect[f'v_{coord}_smooth'].isna().any() or prev_detect[f'{coord}_smooth'].isna().any():
                    detections[f'{coord}_extr'] = prev_detect[f'{coord}_extr'] + prev_detect[f'v_{coord}_extr'] * (detections['time'] - prev_detect['time'])
                else:
                    detections[f'{coord}_extr'] = prev_detect[f'{coord}_smooth'] + prev_detect[f'v_{coord}_smooth'] * (detections['time'] - prev_detect['time'])
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
            if prev_detect is None:
                detections[f'v_{coord}'] = None
                detections[f'v_{coord}_extr'] = None
            else:
                dt = (detections['time'] - prev_detect['time'])
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[coord]) / dt
                if f'v_{coord}_smooth' not in prev_detect.columns or prev_detect[f'v_{coord}_smooth'].isna().any():
                    detections[f'v_{coord}_extr'] = detections[f'v_{coord}']
                else:
                    detections[f'v_{coord}_extr'] = prev_detect[f'v_{coord}_smooth']
                # print(f'v_{coord} = {detections[f'v_{coord}']}')
                # print(f'detect = {detections[coord]}')
                # print(f'prev = {prev_detect[coord]}')
                # print(f'dif = {(detections[coord] - prev_detect[coord])}')

        # Расчет отфильтрованных координат и скоростей
        for coord in (
            'r_measure', 
            'fi_measure', 
            'psi_measure'
        ):
            if len(self.__data) == 0:
                detections[f'{coord}_smooth'] = None
                continue
            n = 1
            r = detections['r_measure']
            if coord == 'r_measure':
                coord_type = 'r'
            elif coord == 'fi_measure':
                coord_type = 'fi'
            elif coord == 'psi_measure':
                coord_type = 'psi'
            mu = self.__calc_mu(n, r, coord_type)
            smooth_coord = self.__calc_smooth_coord(detections[coord], detections[f'{coord}_extr'], mu)
            detections[f'{coord}_smooth'] = smooth_coord

        for v in (
            'v_r_measure',
            'v_fi_measure', 
            'v_psi_measure'
        ):
            if len(self.__data) == 0:
                detections[f'{v}_smooth'] = None
                continue
            n = 1
            r = detections['r_measure']
            if v == 'v_r_measure':
                coord_type = 'r'
            elif v == 'v_fi_measure':
                coord_type = 'fi'
            elif v == 'v_psi_measure':
                coord_type = 'psi'
            mu = self.__calc_mu(n, r, coord_type)
            dt = detections['time'] - self.__data.iloc[len(self.__data) - 1]['time']
            smooth_v = self.__calc_smooth_v(detections[v], detections[f'{v}_extr'], mu, dt)
            detections[f'{v}_smooth'] = smooth_v

        # Concat new detections with data
        #print(detections)
        self.__concat_data(detections)

    def __to_sphere_coord(self, x, y, z) -> tuple:
        """
        Перевод декартовой системы координат в сферическую
        :return: tuple
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        fi = np.atan(y / x)
        psi = np.atan(np.sqrt(x**2 + y**2) / z)
        return (r, fi, psi)

    def __concat_data(self, df: pd.DataFrame) -> None:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])
            self.__data.reset_index(inplace=True, drop=True)

    def __calc_smooth_coord(self, measure_coord, extr_coord, mu):
        error_signal = measure_coord - extr_coord
        alpha = self.__calc_alpha(mu)
        return extr_coord + alpha * error_signal
    
    def __calc_smooth_v(self, measure_v, extr_v, mu, dt):
        error_signal = measure_v - extr_v
        beta = self.__calc_beta(mu)
        return extr_v + beta / dt * error_signal

    def __calc_alpha(self, mu):
        """
        Расчет коэффициента α
        params:
        mu - интенсивность маневра
        """
        lg_series = np.log(mu)
        alpha = lg_series.apply(lambda lg: 0.5 * np.exp(-np.abs(lg - 0.15)**1.7 / (1.3 * np.e)) if lg <= 0.15 else
                                (0.5 * np.exp(-np.abs(lg - 0.15)**1.9 / (1. * np.e)) if lg > 0.15 and lg <= 0.65 else 0.5 * np.exp(-np.abs(0.65 - 0.15)**1.9 / (1. * np.e))))
        return alpha
    
    def __calc_beta(self, mu):
        """
        Расчет коэффициента β
        params:
        mu - интенсивность маневра
        """
        lg_series = np.log(mu)
        alpha_series = self.__calc_alpha(mu)
        beta = pd.Series(
            [
                2 * (1 - alpha - np.sqrt(1 - 2 * alpha)) if lg <= 0.15
                else 2 * (1 - alpha + np.sqrt(1 - 2 * alpha))
                for lg, alpha in zip(lg_series, alpha_series)
            ],
            index=lg_series.index
        )
        # print(f'beta = {beta}, \n lgmu = {lg_series}')
        return beta

    def __calc_mu(self, n, r, coord_type):
        """
        Расчет интенсивности маневра для сферических координат при движении по окружности
        params:
        n — перегрузка при маневре цели
        r - дальность цели
        coord_type - Oneof(r, fi, psi)
        """
        if coord_type == 'r':
            std = self.__r_error
        elif coord_type == 'fi':
            std = self.__fi_error
        elif coord_type == 'psi':
            std = self.__psi_error
        else:
            raise ValueError(f'coord_type should be one of r, fi, psi')
        
        t0 = self.__detection_period / 1000 # период сопровождени в секундах
        g = 9.8
        tmp = 2 / np.pi * (n * g * t0**2) / std
        if coord_type == 'r':
            return tmp / r
        return tmp / (r / r)

    def get_data(self) -> pd.DataFrame:
        cp = self.__data.copy()
        cp = cp[cp['is_observed']]
        cp.drop(columns=['is_observed'], inplace=True)
        return cp

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
