import numpy as np
import pandas as pd

from .unit import Unit
from .air_env import AirEnv


class RadarSystem(Unit):

    def __init__(self, position: np.array=np.array([0, 0, 0]), detection_radius: float=10000, error: np.array=np.array([0., 0., 0.]), air_env: AirEnv = None,
                 detection_fault_probability: float = 0., detection_period: int = 100,
                 detection_delay: int = 0) -> None:
        """
        position: позиция радара
        detection_radius: радиус обнаружения в метрах
        error: вектор ошибок локатора по сферическим координатам (r_error (м), theta_error (градусы), fi_error (градусы))
        air_env: объект воздушной обстановки
        detection_fault_probability: вероятность ошибки обнаружения 
        detection_period: частота обращения локатора к цели (мс)
        detection_delay: задержка обрашения (мс)
        """
        super().__init__()

        self.__detection_fault_probability = detection_fault_probability
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period

        self.__position = np.array(position, dtype=float)
        self.__detection_radius = detection_radius
        self.__r_error, self.__theta_error, self.__fi_error = error
        self.__theta_error = self.__to_radians(self.__theta_error)
        self.__fi_error = self.__to_radians(self.__fi_error)
        print(f'Sphere errors = {self.__r_error, self.__theta_error, self.__fi_error}')
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
            'theta_true' : 'float64',
            'r_measure' : 'float64',
            'fi_measure' : 'float64',
            'theta_measure' : 'float64',
            'r_measure_extr' : 'float64',
            'fi_measure_extr' : 'float64',
            'theta_measure_extr' : 'float64',
            'r_measure_smooth' : 'float64',
            'fi_measure_smooth' : 'float64',
            'theta_measure_smooth' : 'float64',
            'v_x_true_extr': 'float64',
            'v_y_true_extr': 'float64',
            'v_z_true_extr': 'float64',
            'v_x_true': 'float64',
            'v_y_true': 'float64',
            'v_z_true': 'float64',
            'v_x_true_from_sphere': 'float64',
            'v_y_true_from_sphere': 'float64',
            'v_z_true_from_sphere': 'float64',
            'v_x_measure_extr': 'float64',
            'v_y_measure_extr': 'float64',
            'v_z_measure_extr': 'float64',
            'v_x_measure': 'float64',
            'v_y_measure': 'float64',
            'v_z_measure': 'float64',
            'v_r_true_extr': 'float64',
            'v_fi_true_extr': 'float64',
            'v_theta_true_extr': 'float64',
            'v_r_true': 'float64',
            'v_fi_true': 'float64',
            'v_theta_true': 'float64',
            'v_r_measure': 'float64',
            'v_fi_measure': 'float64',
            'v_theta_measure': 'float64',
            'v_r_measure_smooth': 'float64',
            'v_fi_measure_smooth': 'float64',
            'v_theta_measure_smooth': 'float64',
            'v_r_measure_extr': 'float64',
            'v_fi_measure_extr': 'float64',
            'v_theta_measure_extr': 'float64',
            'v_r_true_extr_from_cart': 'float64',
            'v_fi_true_extr_from_cart': 'float64',
            'v_theta_true_extr_from_cart': 'float64',
            'v_r_measure_extr_from_cart': 'float64',
            'v_fi_measure_extr_from_cart': 'float64',
            'v_theta_measure_extr_from_cart': 'float64',
            'r_error': 'float64',
            'fi_error': 'float64',
            'theta_error': 'float64',
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
        detections['r_true'], detections['theta_true'], detections['fi_true'] = self.__to_sphere_coord(detections['x_true'], detections['y_true'], detections['z_true'])
        detections['r_measure'] = detections['r_true'] + np.random.normal(0, self.__r_error, len(detections))
        detections['theta_measure'] = detections['theta_true'] + np.random.normal(0, self.__theta_error, len(detections))
        detections['fi_measure'] = detections['fi_true'] + np.random.normal(0, self.__fi_error, len(detections))
        detections['x_measure'], detections['y_measure'], detections['z_measure'] = self.__to_cartesian_coord(detections['r_measure'], detections['theta_measure'], detections['fi_measure'])
        detections['r_error'] = self.__r_error
        detections['theta_error'] = self.__theta_error
        detections['fi_error'] = self.__fi_error

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
            'theta_measure'
        ):
            # print(f'Prev detect = {None if prev_detect is None or np.isnan(prev_detect[f'v_{coord}_extr']) else prev_detect[f'v_{coord}_extr'] }')
            if prev_detect is None or prev_detect[f'v_{coord}_extr'].isna().any():
                detections[f'{coord}_extr'] = detections[f'{coord}']
            else:
                dt = (detections['time'] - prev_detect['time']) / 1000
                if f'{coord}_smooth' not in prev_detect.columns or f'v_{coord}_smooth' not in prev_detect.columns or prev_detect[f'v_{coord}_smooth'].isna().any() or prev_detect[f'{coord}_smooth'].isna().any():
                    detections[f'{coord}_extr'] = prev_detect[f'{coord}_extr'] + prev_detect[f'v_{coord}_extr'] * dt
                else:
                    # Вычисляем экстраполированные сферические координаты
                    print(f'For sphere cast prev_v_r_measure_smooth = {prev_detect[f'v_r_measure_smooth'][0]}, prev_v_r_measure = {prev_detect[f'v_r_measure'][0]}, prev_v_r_measure_extr = {prev_detect[f'v_r_measure_extr'][0]}')
                    print(f'For sphere cast prev_r_measure_smooth = {prev_detect[f'r_measure_smooth'][0]}, prev_r_measure = {prev_detect[f'r_measure'][0]}, prev_r_measure_extr = {prev_detect[f'r_measure_extr'][0]}')
                    print(f'For sphere cast prev_r_measure_smooth = {prev_detect[f'r_measure_smooth'][0]}, prev_r_measure = {prev_detect[f'r_measure'][0]}, prev_r_measure_extr = {prev_detect[f'fi_measure_extr'][0]}')
                    pd_v_x, pd_v_y, pd_v_z = self.__spherical_to_cartesian_velocity(prev_detect[f'v_r_measure_smooth'], 
                                                                                    prev_detect[f'v_theta_measure_smooth'],
                                                                                    prev_detect[f'v_fi_measure_smooth'],
                                                                                    prev_detect[f'r_measure_smooth'],
                                                                                    prev_detect[f'theta_measure_smooth'],
                                                                                    prev_detect[f'fi_measure_smooth']
                                                                                    )
                    print(f'Cast vel = {pd_v_x[0], pd_v_y[0], pd_v_z[0]}, true vel = {prev_detect[f'v_x_true'][0], prev_detect[f'v_y_true'][0], prev_detect[f'v_z_true'][0]}')
                    pd_x, pd_y, pd_z = self.__to_cartesian_coord(prev_detect[f'r_measure_smooth'],
                                                                 prev_detect[f'theta_measure_smooth'],
                                                                 prev_detect[f'fi_measure_smooth'],
                                                                )
                    print(f'Cast coord = {pd_x[0], pd_y[0], pd_z[0]}, true coord = {prev_detect[f'x_true'][0], prev_detect[f'y_true'][0], prev_detect[f'z_true'][0]}')
                    new_x = pd_x + pd_v_x * dt
                    new_y = pd_y + pd_v_y * dt
                    new_z = pd_z + pd_v_z * dt

                    detections[f'r_measure_extr'], detections[f'theta_measure_extr'], detections[f'fi_measure_extr'] = self.__to_sphere_coord(new_x, new_y, new_z)
                    break
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
            'theta_true',
            'r_measure', 
            'fi_measure', 
            'theta_measure'
        ):
            if prev_detect is None:
                detections[f'v_{coord}'] = None
                detections[f'v_{coord}_extr'] = None
            else:
                dt = (detections['time'] - prev_detect['time']) / 1000 # шаг по врмени в секундах
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[coord]) / dt
                if f'v_{coord}_smooth' not in prev_detect.columns or prev_detect[f'v_{coord}_smooth'].isna().any():
                    if 'v_{coord}_smooth' not in prev_detect.columns:
                        detections[f'v_{coord}_extr'] = detections[f'v_{coord}']
                    else:
                        detections[f'v_{coord}_extr'] = prev_detect[f'v_{coord}'] # экстраполированя скорость = разнице предыдущих двух измерений / dt
                else:
                    detections[f'v_{coord}_extr'] = prev_detect[f'v_{coord}_smooth']

        sphere_coord = (
                'r_true',
                'fi_true',
                'theta_true',
                'r_measure', 
                'fi_measure', 
                'theta_measure'
            )
        if prev_detect is None:
            for coord in sphere_coord:
                detections[f'v_{coord}'] = None
                detections[f'v_{coord}_extr'] = None
                detections[f'v_{coord}_extr_from_cart'] = None
            detections[f'v_x_true_from_sphere'] = None
            detections[f'v_y_true_from_sphere'] = None
            detections[f'v_z_true_from_sphere'] = None
        else:
            detections[f'v_r_true_extr_from_cart'], detections[f'v_fi_true_extr_from_cart'], detections[f'v_theta_true_extr_from_cart'] = self.__cartesian_to_spherical_velocity(
                                                                                                                                prev_detect[f'v_x_true'],
                                                                                                                                prev_detect[f'v_y_true'],
                                                                                                                                prev_detect[f'v_z_true'],
                                                                                                                                prev_detect[f'x_true'],
                                                                                                                                prev_detect[f'y_true'],
                                                                                                                                prev_detect[f'z_true']
                                                                                                                                )
            detections[f'v_x_true_from_sphere'], detections[f'v_y_true_from_sphere'], detections[f'v_z_true_from_sphere'] = self.__spherical_to_cartesian_velocity(
                                                                                                                                detections[f'v_r_true'],
                                                                                                                                detections[f'v_theta_true'],
                                                                                                                                detections[f'v_fi_true'],
                                                                                                                                detections[f'r_true'],
                                                                                                                                detections[f'theta_true'],
                                                                                                                                detections[f'fi_true']
                                                                                                                                )
            if prev_detect[f'v_r_measure_smooth'].isna().any(): 
                detections[f'v_r_measure_extr_from_cart'], detections[f'v_fi_measure_extr_from_cart'], detections[f'v_theta_measure_extr_from_cart'] = self.__cartesian_to_spherical_velocity(
                                                                                                                                prev_detect[f'v_x_measure'],
                                                                                                                                prev_detect[f'v_y_measure'],
                                                                                                                                prev_detect[f'v_z_measure'],
                                                                                                                                prev_detect[f'x_measure'],
                                                                                                                                prev_detect[f'y_measure'],
                                                                                                                                prev_detect[f'z_measure']
                                                                                                                                )
            else:
                detections[f'v_r_measure_extr_from_cart'] = prev_detect[f'v_r_measure_smooth']
                detections[f'v_theta_measure_extr_from_cart'] = prev_detect[f'v_theta_measure_smooth']
                detections[f'v_fi_measure_extr_from_cart'] = prev_detect[f'v_fi_measure_smooth']
        # Расчет отфильтрованных координат и скоростей
        for coord in (
            'r_measure', 
            'fi_measure', 
            'theta_measure'
        ):
            n = 1
            r = detections['r_measure']
            if coord == 'r_measure':
                coord_type = 'r'
            elif coord == 'fi_measure':
                coord_type = 'fi'
            elif coord == 'theta_measure':
                coord_type = 'theta'
            mu = self.__calc_mu(n, r, coord_type)
            smooth_coord = self.__calc_smooth_coord(detections[coord], detections[f'{coord}_extr'], mu)
            detections[f'{coord}_smooth'] = smooth_coord

        for v in (
            'v_r_measure',
            'v_fi_measure', 
            'v_theta_measure'
        ):
            if prev_detect is None:
                detections[f'{v}_smooth'] = None
                continue
            n = 1
            r = detections['r_measure']
            if v == 'v_r_measure':
                coord_type = 'r'
            elif v == 'v_fi_measure':
                coord_type = 'fi'
            elif v == 'v_theta_measure':
                coord_type = 'theta'
            mu = self.__calc_mu(n, r, coord_type)
            dt = (detections['time'] - prev_detect['time']) / 1000
            smooth_v = self.__calc_smooth_v(detections[f'{v}_extr'], detections[f'{v[2:]}'], detections[f'{v[2:]}_extr'], mu, dt)
            print(f'smooth_{v} = {smooth_v[0]}')
            print(f'for smooth {v}_extr = {detections[f'{v}_extr'][0]}, measure = {detections[f'{v[2:]}'][0]}, extr_coord = {detections[f'{v[2:]}_extr'][0]}')
            detections[f'{v}_smooth'] = smooth_v

        # Concat new detections with data
        self.__concat_data(detections)

    def __to_sphere_coord(self, x, y, z) -> tuple:
        """
        Перевод декартовой системы координат в сферическую
        :return: tuple = (r, theta, fi)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) # угол наклона относительно оси z [0, pi]
        fi = np.atan2(y, x) # угол в плоскости x, y [-pi, pi)
        return (r, theta, fi)
    
    def __to_cartesian_coord(self, r, theta, fi) -> tuple:
        """
        Перевод сферической системы координат в декартову
        :return: tuple = (x, y, z)
        """
        x = r * np.sin(theta) * np.cos(fi)
        y = r * np.sin(theta) * np.sin(fi)
        z = r * np.cos(theta)
        return (x, y, z)
    
    def __cartesian_to_spherical_velocity(self, v_x, v_y, v_z, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        fi = np.arctan2(y, x)
        
        v_r = v_x * np.sin(theta) * np.cos(fi) + v_y * np.sin(theta) * np.sin(fi) + v_z * np.cos(theta)
        v_theta = v_x * np.cos(theta) * np.cos(fi) + v_y * np.cos(theta) * np.sin(fi) - v_z * np.sin(theta)
        v_fi = -v_x * np.sin(fi) + v_y * np.cos(fi)
        
        return v_r, v_theta, v_fi
    
    def __spherical_to_cartesian_velocity(self, v_r, v_theta, v_fi, r, theta, fi):
        v_x = v_r * np.sin(theta) * np.cos(fi) + r * v_theta * np.cos(theta) * np.cos(fi) - r * v_fi * np.sin(theta) * np.sin(fi)
        v_y = v_r * np.sin(theta) * np.sin(fi) + r * v_theta * np.cos(theta) * np.sin(fi) + r * v_fi * np.sin(theta) * np.cos(fi)
        v_z = v_r * np.cos(theta) - r * v_theta * np.sin(theta)
        
        return v_x, v_y, v_z

    def __to_radians(self, angle) -> float:
        """
        Перевод угла в градусах в радианы
        """
        return angle / 180 * np.pi

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
    
    def __calc_smooth_v(self, extr_v, measure_coord, extr_coord, mu, dt):
        error_signal = measure_coord - extr_coord
        beta = self.__calc_beta(mu)
        print(f'error_signal = {error_signal.values}, beta = {beta.values}')
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
        coord_type - Oneof(r, fi, theta)
        """
        eps = 1e-6
        if coord_type == 'r':
            std = self.__r_error
        elif coord_type == 'fi':
            std = self.__fi_error
        elif coord_type == 'theta':
            std = self.__theta_error
        else:
            raise ValueError(f'coord_type should be one of r, fi, theta')
        
        t0 = self.__detection_period / 1000 # период сопровождени в секундах
        g = 9.8
        tmp = 2 / np.pi * (n * g * t0**2) / (std + eps)
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
