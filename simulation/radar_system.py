import numpy as np
import pandas as pd
from enum import Enum
from scipy.stats import chi2

from .unit import Unit
from .air_env import AirEnv
from .logger import Logger

class CoordinateType(Enum):
    RADIUS = "radius"
    FI = "fi"
    THETA = "theta"

class RadarSystem(Unit):
    def __init__(self,
                logger = Logger(name='radar_system', log_file='log_file.txt'),
                position: np.array=np.array([0, 0, 0]), 
                detection_radius: float=10000,
                error: np.array=np.array([0., 0., 0.]),  
                air_env: AirEnv = None,
                detection_fault_probability: float = 0., 
                detection_period: int = 100,
                detection_delay: int = 0,
                sharp_fluctuation_prob: float = 0.,
                P_ray: float = 0.,
                G_trans: float = 0.,
                G_recv: float = 0.,
                lamda: float = 0.,
                rcs_mean: float = 0,
                tau: float = 0.,
                miss1: float = 0.,
                miss2: float = 0.,
                miss3: float = 0.,
                N: float = 1.,
                k1: float = 1.,
                k2: float = 1.,
                ) -> None:
        """
        position: позиция радара
        detection_radius: радиус обнаружения в метрах
        error: вектор ошибок локатора по сферическим координатам (r_error (м), theta_error (градусы), fi_error (градусы))
        air_env: объект воздушной обстановки
        detection_fault_probability: вероятность ошибки обнаружения 
        detection_period: частота обращения локатора к цели (мс)
        detection_delay: задержка обрашения (мс)
        P_ray: излучаемая мощность
        G_trans: коэффициент усиления передающей антенны
        G_recv:  коэффициент усиления приемной антенны
        lamda: длина волны
        rcs_mean: эффективная площадь рассеяния (ЭПР) цели(среднее значение)
        tau: длительность импульса
        miss1: Потери при передаче сигнала
        miss2: Потери при обработке сигнала
        miss3: Ширина диаграммы направленности антенны
        N: шум-фактор приемного устройства
        k1: ширина диаграммы направленности антенны
        k2: полоса сигнала
        """
        super().__init__()

        self.__logger = logger
        self.__detection_fault_probability = detection_fault_probability
        self.__sharp_fluctuation_prob = sharp_fluctuation_prob
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period
        self.__r_error, self.__theta_error, self.__fi_error = error
        self.__theta_error = self.__to_radians(self.__theta_error)
        self.__fi_error = self.__to_radians(self.__fi_error)
        self.__position = np.array(position, dtype=float)
        self.__detection_radius = detection_radius
        self.__air_env = air_env
        self.__P_ray = P_ray
        self.__G_trans = G_trans
        self.__G_recv = G_recv
        self.__lamda = lamda
        self.__rcs_mean = rcs_mean
        self.__tau = tau
        self.__miss1 = miss1
        self.__miss2 = miss2
        self.__miss3 = miss3
        self.__N = N
        self.__k1 = k1
        self.__k2 = k2

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
            'x_measure_smooth' : 'float64',
            'y_measure_smooth' : 'float64',
            'z_measure_smooth' : 'float64',
            'r_true' : 'float64',
            'fi_true' : 'float64',
            'theta_true' : 'float64',
            'r_measure' : 'float64',
            'fi_measure' : 'float64',
            'theta_measure' : 'float64',
            'r_measure_extr' : 'float64',
            'fi_measure_extr' : 'float64',
            'theta_measure_extr' : 'float64',
            'r_measure_extr_train' : 'float64',
            'fi_measure_extr_train' : 'float64',
            'theta_measure_extr_train' : 'float64',
            'r_measure_smooth' : 'float64',
            'fi_measure_smooth' : 'float64',
            'theta_measure_smooth' : 'float64',
            'v_x_true': 'float64',
            'v_y_true': 'float64',
            'v_z_true': 'float64',
            'v_x_measure': 'float64',
            'v_y_measure': 'float64',
            'v_z_measure': 'float64',
            'v_x_measure_smooth': 'float64',
            'v_y_measure_smooth': 'float64',
            'v_z_measure_smooth': 'float64',
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
            'v_r_measure_extr_train': 'float64',
            'v_fi_measure_extr_train': 'float64',
            'v_theta_measure_extr_train': 'float64',
            'noise/signal ratio': 'float64',
            'P_ray': 'float64',
            'G_trans': 'float64',
            'G_recv': 'float64',
            'lambda': 'float64',
            'tau': 'float64',
            'rcs': 'float64',
            'tau': 'float64',
            'sum_miss': 'float64',
            'N': 'float64',
            'k1': 'float64',
            'k2': 'float64',
            'r_error': 'float64',
            'fi_error': 'float64',
            'theta_error': 'float64',
            'true_alpha_r': 'float64',
            'true_alpha_theta': 'float64',
            'true_alpha_fi': 'float64',
            'true_beta_r': 'float64',
            'true_beta_theta': 'float64',
            'true_beta_fi': 'float64',
        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)

    def trigger(self) -> None:
        if self.time.get_time() % self.__detection_period == self.__detection_delay:
            if np.random.choice([False, True],
                                p=[self.__detection_fault_probability, 1.0 - self.__detection_fault_probability]):
                self.detect_air_objects()

    def detect_air_objects(self) -> None:
        prev_detect = None
        if len(self.__data) != 0: 
            air_objects_count = self.__air_env.get_air_objects_count()
            prev_detect = self.__data.tail(air_objects_count)
            #print(f'prev_detect = {prev_detect}')
            prev_detect = prev_detect.set_index(prev_detect['id'])
            #print(f'prev_detect_set = {prev_detect}')
        
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
        detections['time'] = self.time.get_time() / 1000
        detections['r_true'], detections['theta_true'], detections['fi_true'] = self.__to_sphere_coord(detections['x_true'], detections['y_true'], detections['z_true'])

        rcs = self.__calculate_rcs(self.__rcs_mean, detections.shape[0])
        ns = self.__calc_noise_signal(detections['r_true'], rcs)
        ns_db = 10 * np.log10(ns)
        detections['noise/signal ratio'] = ns_db

        self.__logger.debug(f'noise/signal in dB = {ns_db[0]}')
        c = 3 * 10**8
        self.__r_error_with_ns = c * np.sqrt(np.pi) / (2 * self.__k2 * np.sqrt(2 * ns))
        self.__fi_error_with_ns = np.sqrt(np.pi) / (self.__to_radians(self.__k1) * np.sqrt(2 * ns))
        self.__theta_error_with_ns = np.sqrt(np.pi) / (self.__to_radians(self.__k1) * np.sqrt(2 * ns))

        detections['P_ray'] = self.__P_ray
        detections['G_trans'] = self.__calc_linear_from_dB(self.__G_trans)
        detections['G_recv'] = self.__calc_linear_from_dB(self.__G_recv)
        detections['lambda'] = self.__lamda
        detections['tau'] = self.__tau
        detections['rcs'] = rcs
        detections['sum_miss'] = self.__calc_linear_from_dB(self.__miss1) * self.__calc_linear_from_dB(self.__miss2) * self.__calc_linear_from_dB(self.__miss3)
        detections['N'] = self.__calc_linear_from_dB(self.__N)
        detections['k1'] = self.__k1
        detections['k2'] = self.__k2

        self.__logger.debug(f'r_error_with_ns = {self.__r_error_with_ns[0]}')
        self.__logger.debug(f'fi_error_with_ns = {self.__fi_error_with_ns[0]}')
        self.__logger.debug(f'theta_error_with_ns = {self.__theta_error_with_ns[0]}')
        sharp_coef = np.random.choice([0, 1], p=[1 - self.__sharp_fluctuation_prob, self.__sharp_fluctuation_prob])
        detections['r_measure'] = detections['r_true'] + np.random.normal(0, self.__r_error_with_ns, len(detections)) + sharp_coef * np.random.uniform(10, 12) # добавим к нормальному шуму еще резкий скачок с какой-то вероятностью
        detections['theta_measure'] = self.normalize_theta(detections['theta_true'] + np.random.normal(0, self.__theta_error_with_ns, len(detections)))
        detections['fi_measure'] = self.normalize_fi(detections['fi_true'] + np.random.normal(0, self.__fi_error_with_ns, len(detections)))

        detections['x_measure'], detections['y_measure'], detections['z_measure'] = self.__to_cartesian_coord(detections['r_measure'], detections['theta_measure'], detections['fi_measure'])
        detections['r_error'] = self.__r_error_with_ns
        detections['theta_error'] = self.__theta_error_with_ns
        detections['fi_error'] = self.__fi_error_with_ns
        
        for coord in (
            'r_measure',
            'fi_measure',
            'theta_measure'
        ):
            # print(f'Prev detect = {None if prev_detect is None or np.isnan(prev_detect[f'v_{coord}_extr']) else prev_detect[f'v_{coord}_extr'] }')
            if prev_detect is None or prev_detect[f'v_{coord}_extr'].isna().any():
                detections[f'{coord}_extr'] = detections[f'{coord}']
                detections[f'{coord}_extr_train'] = detections[f'{coord}']
            else:
                dt = (detections['time'] - prev_detect['time']) / 1000
                # Вычисляем экстраполированные сферические координаты
                self.__logger.debug(f'For sphere cast prev_v_r_measure_smooth = {prev_detect[f'v_r_measure_smooth'][0]}, prev_v_r_measure = {prev_detect[f'v_r_measure'][0]}, prev_v_r_measure_extr = {prev_detect[f'v_r_measure_extr'][0]}')
                self.__logger.debug(f'For sphere cast prev_r_measure_smooth = {prev_detect[f'r_measure_smooth'][0]}, prev_r_measure = {prev_detect[f'r_measure'][0]}, prev_r_measure_extr = {prev_detect[f'r_measure_extr'][0]}')
                self.__logger.debug(f'For sphere cast prev_theta_measure_smooth = {prev_detect[f'theta_measure_smooth'][0]}, prev_theta_measure = {prev_detect[f'theta_measure'][0]}, theta_theta_measure_extr = {prev_detect[f'theta_measure_extr'][0]}')
                self.__logger.debug(f'For sphere cast prev_fi_measure_smooth = {prev_detect[f'fi_measure_smooth'][0]}, prev_fi_measure = {prev_detect[f'fi_measure'][0]}, prev_fi_measure_extr = {prev_detect[f'fi_measure_extr'][0]}')
                pd_v_x, pd_v_y, pd_v_z = self.__spherical_to_cartesian_velocity(
                    prev_detect[f'v_r_measure_smooth'], 
                    prev_detect[f'v_theta_measure_smooth'],
                    prev_detect[f'v_fi_measure_smooth'],
                    prev_detect[f'r_measure_smooth'],
                    prev_detect[f'theta_measure_smooth'],
                    prev_detect[f'fi_measure_smooth']
                )
                self.__logger.debug(f'Cast vel = {pd_v_x[0], pd_v_y[0], pd_v_z[0]}, true vel = {prev_detect[f'v_x_true'][0], prev_detect[f'v_y_true'][0], prev_detect[f'v_z_true'][0]}')
                pd_x, pd_y, pd_z = self.__to_cartesian_coord(
                    prev_detect[f'r_measure_smooth'],
                    prev_detect[f'theta_measure_smooth'],
                    prev_detect[f'fi_measure_smooth'],
                )
                self.__logger.debug(f'Cast coord = {pd_x[0], pd_y[0], pd_z[0]}, true coord = {prev_detect[f'x_true'][0], prev_detect[f'y_true'][0], prev_detect[f'z_true'][0]}')
                new_x = pd_x + pd_v_x * dt
                new_y = pd_y + pd_v_y * dt
                new_z = pd_z + pd_v_z * dt

                detections[f'r_measure_extr'], detections[f'theta_measure_extr'], detections[f'fi_measure_extr'] = self.__to_sphere_coord(new_x, new_y, new_z)
                detections[f'theta_measure_extr'] = self.normalize_theta(detections[f'theta_measure_extr'])
                detections[f'fi_measure_extr'] = self.normalize_fi(detections[f'fi_measure_extr'])

                # Считаем экстраполированные координаты для обучения как прошлая измеренная скорость на время + предыдщуюя измеренная координата 
                pd_v_x, pd_v_y, pd_v_z = self.__spherical_to_cartesian_velocity(
                    prev_detect[f'v_r_measure'], 
                    prev_detect[f'v_theta_measure'],
                    prev_detect[f'v_fi_measure'],
                    prev_detect[f'r_measure'],
                    prev_detect[f'theta_measure'],
                    prev_detect[f'fi_measure']
                )
                pd_x, pd_y, pd_z = self.__to_cartesian_coord(
                    prev_detect[f'r_measure'],
                    prev_detect[f'theta_measure'],
                    prev_detect[f'fi_measure'],
                )
                
                new_x = pd_x + pd_v_x * dt
                new_y = pd_y + pd_v_y * dt
                new_z = pd_z + pd_v_z * dt

                detections[f'r_measure_extr_train'], detections[f'theta_measure_extr_train'], detections[f'fi_measure_extr_train'] = self.__to_sphere_coord(new_x, new_y, new_z)
                # detections[f'theta_measure_extr_train'] = self.normalize_theta(detections[f'theta_measure_extr_train'])
                # detections[f'fi_measure_extr_train'] = self.normalize_fi(detections[f'fi_measure_extr_train'])
                break

        # Выичисление скоростей
        for coord in (
            'x_true', 
            'y_true', 
            'z_true', 
        ):
            if prev_detect is None:
                detections[f'v_{coord}'] = None
            else:
                dt = (detections['time'] - prev_detect['time']) / 1000 # шаг по времени в секундах
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[coord]) / dt

        if prev_detect is None:
            detections[f'v_r_true'] = None
            detections[f'v_theta_true'] = None
            detections[f'v_fi_true'] = None
        else:
            detections[f'v_r_true'], detections[f'v_theta_true'], detections[f'v_fi_true'] = self.__cartesian_to_spherical_velocity(detections[f'v_x_true'], detections[f'v_y_true'], detections[f'v_z_true'], detections['x_true'], detections[f'y_true'], detections[f'z_true'])

        for coord in (
            'x_measure',
            'y_measure',
            'z_measure',
        ):
            if prev_detect is None:
                detections[f'v_{coord}'] = None
            else:
                dt = (detections['time'] - prev_detect['time']) / 1000 # шаг по времени в секундах
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[coord]) / dt # Вычисление скорости на текущем цикле обзора

        if prev_detect is None:
            detections[f'v_r_measure'] = None
            detections[f'v_theta_measure'] = None
            detections[f'v_fi_measure'] = None
        else:
            detections[f'v_r_measure'], detections[f'v_theta_measure'], detections[f'v_fi_measure'] = self.__cartesian_to_spherical_velocity(detections[f'v_x_measure'], detections[f'v_y_measure'], detections[f'v_z_measure'], detections['x_measure'], detections[f'y_measure'], detections[f'z_measure'])

        if prev_detect is None or prev_detect['v_x_measure'].isna().any():
            detections[f'v_r_measure_extr_train'] = None
            detections[f'v_theta_measure_extr_train'] = None
            detections[f'v_fi_measure_extr_train'] = None
        else:
            detections[f'v_r_measure_extr_train'], detections[f'v_theta_measure_extr_train'], detections[f'v_fi_measure_extr_train'] = self.__cartesian_to_spherical_velocity(prev_detect[f'v_x_measure'], prev_detect[f'v_y_measure'], prev_detect[f'v_z_measure'], prev_detect['x_measure'], prev_detect[f'y_measure'], prev_detect[f'z_measure'])

        if prev_detect is None:
            detections[f'v_r_measure_extr'] = None
            detections[f'v_theta_measure_extr'] = None
            detections[f'v_fi_measure_extr'] = None
        elif prev_detect[f'v_r_measure_smooth'].isna().any():
            detections[f'v_r_measure_extr'], detections[f'v_theta_measure_extr'], detections[f'v_fi_measure_extr'] = self.__cartesian_to_spherical_velocity(prev_detect[f'v_x_measure'], prev_detect[f'v_y_measure'], prev_detect[f'v_z_measure'], prev_detect[f'x_measure'], prev_detect[f'y_measure'], prev_detect[f'z_measure'])
        else:
            # detections[f'v_r_measure_extr'], detections[f'v_theta_measure_extr'], detections[f'v_fi_measure_extr'] = self.__cartesian_to_spherical_velocity(prev_detect[f'v_x_measure_smooth'], prev_detect[f'v_y_measure_smooth'], prev_detect[f'v_z_measure_smooth'], prev_detect[f'x_measure_smooth'], prev_detect[f'y_measure_smooth'], prev_detect[f'z_measure_smooth'])
            detections[f'v_r_measure_extr'] = prev_detect[f'v_r_measure_smooth']
            detections[f'v_theta_measure_extr'] = prev_detect[f'v_theta_measure_smooth']
            detections[f'v_fi_measure_extr'] = prev_detect[f'v_fi_measure_smooth']


        # Расчет отфильтрованных координат и скоростей
        for coord in (
            'r_measure', 
            'fi_measure', 
            'theta_measure'
        ):
            n = 4
            r = detections['r_measure']
            if coord == 'r_measure':
                coord_type = CoordinateType.RADIUS
            elif coord == 'fi_measure':
                coord_type = CoordinateType.FI
            elif coord == 'theta_measure':
                coord_type = CoordinateType.THETA
            mu = self.__calc_mu(n, r, coord_type)
            self.__logger.debug(f'mu from alpha = {mu.values}, type = {coord_type}')

            smooth_coord = self.__calc_smooth_coord(detections[coord], detections[f'{coord}_extr'], mu, coord_type=coord_type)
            if coord_type == CoordinateType.FI:
                detections[f'{coord}_smooth'] = self.normalize_fi(smooth_coord)
            elif coord_type == CoordinateType.THETA:
                detections[f'{coord}_smooth'] = self.normalize_theta(smooth_coord)
            else:
                detections[f'{coord}_smooth'] = smooth_coord

        detections['x_measure_smooth'], detections['y_measure_smooth'], detections['z_measure_smooth'] = self.__to_cartesian_coord(detections['r_measure_smooth'], detections['theta_measure_smooth'], detections['fi_measure_smooth'])

        for v in (
            'v_r_measure',
            'v_fi_measure',
            'v_theta_measure'
        ):
            if prev_detect is None:
                detections[f'{v}_smooth'] = None
                continue
            n = 3
            r = detections['r_measure']
            if v == 'v_r_measure':
                coord_type = CoordinateType.RADIUS
            elif v == 'v_fi_measure':
                coord_type = CoordinateType.FI
            elif v == 'v_theta_measure':
                coord_type = CoordinateType.THETA
            mu = self.__calc_mu(n, r, coord_type)
            dt = (detections['time'] - prev_detect['time']) / 1000
            self.__logger.debug(f'for smooth {v}_extr = {detections[f'{v}_extr'][0]}, measure = {detections[f'{v[2:]}'][0]}, extr_coord = {detections[f'{v[2:]}_extr'][0]}')
            smooth_v = self.__calc_smooth_v(detections[f'{v}_extr'], detections[f'{v[2:]}'], detections[f'{v[2:]}_extr'], mu, dt, coord_type=coord_type)
            self.__logger.debug(f'smooth_{v} = {smooth_v[0]}')
            detections[f'{v}_smooth'] = smooth_v
        
        # Перевод сглаженных скоростей из сферических в декартовы
        if prev_detect is None:
            detections['v_x_measure_smooth'] = None
            detections['v_y_measure_smooth'] = None
            detections['v_z_measure_smooth'] = None
        else:
            detections['v_x_measure_smooth'], detections['v_y_measure_smooth'], detections['v_z_measure_smooth'] = self.__spherical_to_cartesian_velocity(
                detections[f'v_r_measure_smooth'],
                detections[f'v_theta_measure_smooth'],
                detections[f'v_fi_measure_smooth'],
                detections['r_measure_smooth'],
                detections['theta_measure_smooth'],
                detections['fi_measure_smooth']
            )

        # Расчет таргета для обучения
        detections['true_alpha_r'] = (detections['r_true'] - detections['r_measure_extr_train']) / (detections['r_measure'] - detections['r_measure_extr_train'])
        detections['true_alpha_theta'] = (self.normalize_theta(detections['theta_true']) - self.normalize_theta(detections['theta_measure_extr_train'])) / (self.normalize_theta(detections['theta_measure']) - self.normalize_theta(detections['theta_measure_extr_train']))
        detections['true_alpha_fi'] = (self.normalize_fi(detections['fi_true']) - self.normalize_fi(detections['fi_measure_extr_train'])) / (self.normalize_fi(detections['fi_measure']) - self.normalize_fi(detections['fi_measure_extr_train']))

        if detections['v_r_true'].isna().any() or detections['v_r_measure_extr'].isna().any():
            detections['true_beta_r'] = None
            detections['true_beta_theta'] = None
            detections['true_beta_fi'] = None
        else:
            dt = dt = (detections['time'] - prev_detect['time']) / 1000
            detections['true_beta_r'] = (detections['v_r_true'] - detections['v_r_measure_extr_train']) * dt / (detections['r_measure'] - detections['r_measure_extr_train'])
            detections['true_beta_theta'] = (detections['v_theta_true'] - detections['v_theta_measure_extr_train']) * dt / (self.normalize_theta(detections['theta_measure']) - self.normalize_theta(detections['theta_measure_extr_train']))
            detections['true_beta_fi'] = (detections['v_fi_true'] - detections['v_fi_measure_extr_train']) * dt / (self.normalize_fi(detections['fi_measure']) - self.normalize_fi(detections['fi_measure_extr_train']))
        # Concat new detections with data
        self.__concat_data(detections)

    def normalize_theta(self, theta):
        """Нормализация угла theta в пределах от 0 до pi."""
        return np.clip(theta, 0, np.pi)

    def normalize_fi(self, fi):
        """Нормализация угла fi в пределах от -pi до pi."""
        return (fi + np.pi) % (2 * np.pi) - np.pi

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
        
        v_r = (v_x * x + v_y * y + v_z * z) / r
        v_theta = (-v_x * np.sin(fi) + v_y * np.cos(fi) + v_z * np.sin(theta)) / r
        v_fi = (v_x * np.sin(fi) + v_y * np.cos(fi)) / (r * np.sin(theta))
        
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

    def __calc_smooth_coord(self, measure_coord, extr_coord, mu, coord_type=CoordinateType.RADIUS):
        if coord_type == CoordinateType.RADIUS:
            error_signal = measure_coord - extr_coord
        elif coord_type == CoordinateType.THETA:
            normalized_measure_coord = self.normalize_theta(measure_coord)
            normalized_extr_coord = self.normalize_theta(extr_coord)
            error_signal = normalized_measure_coord - normalized_extr_coord
        else:
            normalized_measure_coord = self.normalize_fi(measure_coord)
            normalized_extr_coord = self.normalize_fi(extr_coord)
            error_signal = normalized_measure_coord - normalized_extr_coord

        alpha = self.__calc_alpha(mu)
        self.__logger.debug(f'From calc smooth coord error_signal = {error_signal.values}, alpha = {alpha.values}, mu = {mu.values}')
        return extr_coord + alpha * error_signal
    
    def __calc_smooth_v(self, extr_v, measure_coord, extr_coord, mu, dt, coord_type):
        if coord_type == CoordinateType.RADIUS:
            error_signal = measure_coord - extr_coord
        elif coord_type == CoordinateType.THETA:
            normalized_measure_coord = measure_coord
            normalized_extr_coord = extr_coord
            error_signal = normalized_measure_coord - normalized_extr_coord
        else:
            normalized_measure_coord = measure_coord
            normalized_extr_coord = extr_coord
            error_signal = normalized_measure_coord - normalized_extr_coord

        beta = self.__calc_beta(mu)
        self.__logger.debug(f'From calc smooth velocity error_signal = {error_signal.values}, beta = {beta.values}, mu = {mu.values}')
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
        if coord_type == CoordinateType.RADIUS:
            std = self.__r_error_with_ns
        elif coord_type == CoordinateType.FI:
            std = self.__fi_error
        elif coord_type == CoordinateType.THETA:
            std = self.__theta_error
        else:
            raise ValueError(f'coord_type should be one of r, fi, theta')
        
        t0 = self.__detection_period / 1000 # период сопровождени в секундах
        g = 9.8
        tmp = 2 * (n * g * t0**2) / (np.pi * (std + eps))
        self.__logger.debug(f'tmp = {tmp}, coord_type = {coord_type}')
        if coord_type != CoordinateType.RADIUS:
            return tmp / r
        return tmp / (r / r)

    def __calc_noise_signal(self, r, rcs):
        k = 1.38 * 10**(-23)
        T = 290
        miss = self.__calc_linear_from_dB(self.__miss1) * self.__calc_linear_from_dB(self.__miss2) * self.__calc_linear_from_dB(self.__miss3)
        linear_G_trans = self.__calc_linear_from_dB(self.__G_trans)
        linear_G_recv = self.__calc_linear_from_dB(self.__G_recv)
        linear_N = self.__calc_linear_from_dB(self.__N)
        numerator = self.__P_ray * self.__tau * linear_G_trans * linear_G_recv * self.__lamda**2 * rcs * miss
        denominator = (4 * np.pi)**3 * r**4 * linear_N * k * T
        return numerator / denominator

    def __calc_linear_from_dB(self, d):
        return 10**(0.1 * d) # d = 10 * lg(ratio)

    def __calculate_rcs(self, mean_rcs, num_samples):
        df = 4 # степени свободы
        scale = mean_rcs / 2
        rcs_values = chi2.rvs(df, scale=scale, size=num_samples)
        return rcs_values

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
