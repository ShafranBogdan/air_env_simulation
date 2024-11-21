import numpy as np
from .air_env import AirEnv
from .air_object import AirObject
from .trajectory import Trajectory, TrajectorySegment
from .unit import Unit
from .logger import Logger

class Generator(Unit):
    def __init__(
            self,
            detection_radius: float,
            start_time: float, 
            end_time: float, 
            neg_v_prob: float = 0.5, 
            num_samples: int = 1, 
            num_seg: int = 2, 
            velocity_pool=np.arange(100, 201, 25), 
            radius_pool=np.arange(5000, 10001, 500),
            logger = Logger(name='generation', log_file='log_file.txt'),
        ):
        super().__init__()
        self.__detection_radius = detection_radius
        self.__num_samples = num_samples
        self.__num_seg = num_seg
        self.velocity_pool = velocity_pool
        self.radius_pool = radius_pool
        self.neg_v_prob = neg_v_prob
        self.start_time = start_time
        self.end_time = end_time
        self.__logger = logger

    def trigger(self, **kwargs) -> None:
        pass
    
    def __generate_random_intervals(self, start_time, end_time, num_seg):
        random_ratios = np.random.dirichlet(np.ones(num_seg))
        interval_durations = random_ratios * (end_time - start_time)
        
        time_intervals = [start_time]
        for duration in interval_durations:
            time_intervals.append(time_intervals[-1] + duration)
        return np.array(time_intervals)

    def calc_w(self, v: float, r: float) -> float:
        return v / r

    def convert_velocity(self, V: float) -> float:
        return V / 1000

    def __get_random_position(self, r, z_min=10**3, z_max=1.2*10**4) -> np.array:
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)
        radius = np.random.uniform(0, 1) ** (1 / 3)
        vec = vec * r * radius
        vec[2] = np.clip(vec[2], z_min, z_max)
        return vec

    def __get_time_interval(self, time_intervals, num_seg) -> tuple:
        start_time = time_intervals[num_seg] + 1 if num_seg > 0 else time_intervals[num_seg]
        end_time = time_intervals[num_seg + 1]
        return start_time, end_time

    def __make_linear(self, trajectory, time_intervals, num_seg) -> TrajectorySegment:
        if num_seg == 0:
            sign = np.random.choice([-1, 1], p=[self.neg_v_prob, 1 - self.neg_v_prob])
            velocity = [
                sign * self.convert_velocity(np.random.choice(self.velocity_pool)), 
                sign * self.convert_velocity(np.random.choice(self.velocity_pool)),
                0
            ]
        else:
            velocity = None
        start_time, end_time = self.__get_time_interval(time_intervals, num_seg)
        # self.__logger.debug(f"Linear st_t = {start_time}, end_t = {end_time}")
        # self.__logger.debug(f'V info num_seg = {num_seg}, v = {velocity}')
        if len(trajectory.get_segments()) != 0:
            return TrajectorySegment(start_time, end_time, None, 'linear', velocity, previous_segment=trajectory.get_segments()[-1])
        else:
            initial_position = self.__get_random_position(self.__detection_radius)
            # initial_position = np.array([0, 0, 5000])
            return TrajectorySegment(start_time, end_time, initial_position, 'linear', velocity)

    def __make_circular(self, trajectory, time_intervals, num_seg) -> TrajectorySegment:
        radius = np.random.choice(self.radius_pool)
        v = self.convert_velocity(np.random.choice(self.velocity_pool))
        angular_velocity = self.calc_w(v, radius)
        vz = 0
        start_time, end_time = self.__get_time_interval(time_intervals, num_seg)
        # self.__logger.debug(f"Circular st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()) == 0:
            raise ValueError("Движение по окружности может быть только после прямолинейного")
        return TrajectorySegment(start_time, end_time, None, 'circular', [radius, angular_velocity, vz, np.random.choice([-1, 1])], previous_segment=trajectory.get_segments()[-1])

    def gen_traces(self) -> AirEnv:
        ae = AirEnv()
        for _ in range(self.__num_samples):
            trajectory = Trajectory()
            # self.__logger.debug(f"Id = {_}")
            time_intervals = self.__generate_random_intervals(self.start_time, self.end_time, self.__num_seg)
            for num_seg in range(self.__num_seg):
                motion_type = ['linear', 'circular'][num_seg % 2]
                if motion_type == 'linear':
                    trajectory.add_segment(self.__make_linear(trajectory, time_intervals, num_seg))
                else:
                    trajectory.add_segment(self.__make_circular(trajectory, time_intervals, num_seg))
            new_ao = AirObject(trajectory)
            ae.attach_air_object(new_ao)
        return ae
