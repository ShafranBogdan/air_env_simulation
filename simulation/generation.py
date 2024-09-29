import numpy as np
from .air_env import AirEnv
from .air_object import AirObject
from .trajectory import Trajectory, TrajectorySegment
from .unit import Unit

class Generator(Unit):
    def __init__(self, detection_radius:float, start_time:float, end_time:float, neg_v_prob:float = 0.5, num_samples: float = 1, num_seg:float = 2, velocity_pool = np.arange(200, 401, 50), radius_pool=np.arange(2000, 3001, 200)):
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

    def calc_w(self, v: float, r: float) -> float:
        """
        Расчет угловой скорости рад / мс
        v - м / мс
        r - м
        """
        return v / r

    def convert_velocity(self, V: float) -> float:
        """
        V in meters in s
        return: V in meters in ms
        """

        return V / 1000

    def __get_random_position(self, r) -> np.array:
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec) # Случайный единичный вектор

        radius = np.random.uniform(0, 1) ** (1 / 3) # Случайный радиус внутри объема единичной сферы

        return vec * r * radius

    def __get_time_interval(self, num_seg) -> tuple:
        start_time = self.time_intervals[num_seg] + 1 if num_seg > 0 else self.time_intervals[num_seg]
        end_time = self.time_intervals[num_seg + 1]
        print(start_time, end_time, self.time_intervals)
        return start_time, end_time

    def __make_linear(self, trajectory, num_seg) -> TrajectorySegment:
        velocity = [
                    self.convert_velocity(np.random.choice(self.velocity_pool)), 
                    self.convert_velocity(np.random.choice(self.velocity_pool)), 
                    self.convert_velocity(np.random.choice(np.arange(0, 51, 10)))
                ]  # Скорости по x, y, z
        
        start_time, end_time = self.__get_time_interval(num_seg)
        print(f"Linear st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()):
            return TrajectorySegment(start_time, end_time, None, 'linear', velocity)
        else:
            initial_position = self.__get_random_position(self.__detection_radius)
            return TrajectorySegment(start_time, end_time, initial_position, 'linear', velocity)

    def __make_circular(self, trajectory, num_seg) -> TrajectorySegment:
        radius = 100
        angular_velocity = self.calc_w(self.convert_velocity(np.random.choice(self.velocity_pool)), radius) # угловая скорость
        vz = np.random.choice(np.arange(-10, 10, 2)) # моделируем скорость по оси z при движении по окружности
        start_time, end_time = self.__get_time_interval(num_seg)
        print(f"Circular st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()) == 0:
            raise ValueError("Движение по окружности может быть только после прямолинейного")
        return TrajectorySegment(start_time, end_time, None, 'circular', [radius, angular_velocity, vz, -1], previous_segment=trajectory.get_segments()[-1])

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

