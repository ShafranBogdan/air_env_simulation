import numpy as np
from .air_env import AirEnv
from .air_object import AirObject
from .trajectory import Trajectory, TrajectorySegment
from .unit import Unit
from tools import Physic


class Generator(Unit):
    def __init__(self, detection_radius: float, start_time: float, end_time: float, neg_v_prob: float = 0.5,
                 num_samples: float = 1, num_seg: float = 2, velocity_pool=np.arange(200, 401, 50),
                 radius_pool=np.arange(5000, 10001, 500)):
        super().__init__()
        self.__detection_radius = detection_radius
        self.__num_samples = num_samples
        self.__num_seg = num_seg
        self.velocity_pool = velocity_pool
        self.radius_pool = radius_pool
        self.neg_v_prob = neg_v_prob
        self.time_intervals = np.arange(start_time, end_time + 1, (end_time - start_time) / num_seg)

    def trigger(self, **kwargs) -> None:
        pass

    def __get_random_position(self, r, z_min=10 ** 3, z_max=1.2 * 10 ** 4) -> np.array:
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)  # Случайный единичный вектор

        radius = np.random.uniform(0, 1) ** (1 / 3)  # Случайный радиус внутри объема единичной сферы

        vec = vec * r * radius  # Множим вектор на радиус

        # Ограничиваем третью координату z_max
        vec[2] = np.clip(vec[2], z_min, z_max)

        return np.array([0, 0, 0])
        # return vec

    def __get_time_interval(self, num_seg) -> tuple:
        start_time = self.time_intervals[num_seg] + 1 if num_seg > 0 else self.time_intervals[num_seg]
        end_time = self.time_intervals[num_seg + 1]
        print(start_time, end_time, self.time_intervals)
        return start_time, end_time

    def __make_linear(self, trajectory, num_seg) -> TrajectorySegment:
        sign = np.random.choice([-1, 1], p=[self.neg_v_prob, 1 - self.neg_v_prob])
        velocity = [
            sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
            sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
            Physic.convert_velocity(np.random.choice(np.arange(0, 51, 10)))
        ]  # Скорости по x, y, z

        start_time, end_time = self.__get_time_interval(num_seg)
        print(f"Linear st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()):
            return TrajectorySegment(start_time, end_time, None, 'linear', velocity)
        else:
            initial_position = self.__get_random_position(self.__detection_radius)
            return TrajectorySegment(start_time, end_time, initial_position, 'linear', velocity)

    def __make_circular(self, trajectory, num_seg) -> TrajectorySegment:
        radius = np.random.choice(self.radius_pool)  # выбираем случайный радиус из допустимого набора
        v = Physic.convert_velocity(
            np.random.choice(self.velocity_pool))  # выбираем случайную скорость из допустимого набора
        angular_velocity = Physic.calc_w(v, radius)  # угловая скорость
        vz = np.random.choice(np.arange(-10, 10, 2))  # моделируем скорость по оси z при движении по окружности
        start_time, end_time = self.__get_time_interval(num_seg)
        print(f"Circular st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()) == 0:
            raise ValueError("Движение по окружности может быть только после прямолинейного")
        return TrajectorySegment(start_time, end_time, None, 'circular',
                                 [radius, angular_velocity, vz, np.random.choice([-1, 1])],
                                 previous_segment=trajectory.get_segments()[-1])

    def gen_traces(self) -> AirEnv:
        ae = AirEnv()
        for _ in range(self.__num_samples):
            trajectory = Trajectory()
            print(f"Id = {_}")
            for num_seg in range(self.__num_seg):
                motion_type = np.random.choice(['linear', 'circular'], [0, 1]) if num_seg >= 1 else 'linear'
                if motion_type == 'linear':
                    trajectory.add_segment(self.__make_linear(trajectory, num_seg))
                else:
                    trajectory.add_segment(self.__make_circular(trajectory, num_seg))
            new_ao = AirObject(trajectory)
            ae.attach_air_object(new_ao)
        return ae
