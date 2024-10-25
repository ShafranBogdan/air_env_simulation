import numpy as np


class TrajectorySegment:
    def __init__(self, start_time, end_time, initial_position, motion_type, params, previous_segment=None):
        """
        Параметры:
        - start_time: время начала сегмента
        - end_time: время окончания сегмента
        - initial_position: начальная позиция [x, y, z] или None, если нужно вычислить по предыдущему сегменту
        - motion_type: тип движения ('linear' или 'circular')
        - params: параметры для движения:
            Для 'linear' - (vx, vy, vz) - скорости по осям x, y, z
            Для 'circular' - (radius, angular_velocity, vz, direction) - параметры окружности и движение по оси z
        - previous_segment: предыдущий сегмент для вычисления начальной точки
        """
        self.start_time = start_time
        self.end_time = end_time
        self.motion_type = motion_type
        self.params = params

        # Если начальная позиция не задана, вычисляем её по предыдущему сегменту
        if initial_position is None and previous_segment:
            self.initial_position = previous_segment.get_position_in_segment(previous_segment.end_time)
        else:
            self.initial_position = np.array(initial_position)

        # Рассчитываем центр окружности для кругового движения, если нужно
        if self.motion_type == 'circular' and previous_segment:
            radius, angular_velocity, _, direction = self.params
            vx, vy, vz = previous_segment.params  # Скорости предыдущего линейного сегмента
            self.center = self.calculate_center_from_last_point(radius, vx, vy, direction)
            self.initial_angle = self.calculate_initial_angle(vx, vy, direction)

    def calculate_center_from_last_point(self, radius, vx, vy, direction):
        """
        Рассчитывает центр окружности для плавного входа в круговое движение.
        Центр окружности находится на расстоянии radius от последней точки предыдущей траектории
        в направлении, перпендикулярном направлению движения.

        Параметры:
        - radius: радиус окружности
        - vx, vy: компоненты скорости движения по x и y в предыдущем сегменте (линейное движение)
        - direction: направление поворота, +1 для направо, -1 для налево
        """
        incline_angle = np.arctan2(vy, vx)

        # Начальная точка траектории — это конец предыдущей прямолинейной траектории
        x0, y0, z0 = self.initial_position

        # Центр окружности смещен от конечной точки на радиус вдоль нормали
        return np.array([x0 + radius * np.cos(incline_angle - direction * np.pi / 2),
                         y0 + radius * np.sin(incline_angle - direction * np.pi / 2), z0])

    def calculate_initial_angle(self, vx, vy, direction):
        """
        Рассчитывает начальный угол для круговой траектории, чтобы начать движение по касательной.
        Начальный угол должен быть смещён на pi/2 в зависимости от направления поворота.

        Параметры:
        - vx, vy: скорости по x и y предыдущего линейного движения
        - direction: направление поворота, +1 для направо, -1 для налево
        """
        # Угол движения относительно оси X (угол наклона траектории в точке перехода)
        tangent_angle = np.arctan2(vy, vx)

        # Для плавного входа в круговую траекторию добавляем фазовый сдвиг ±pi/2
        initial_angle = tangent_angle + direction * np.pi / 2
        return initial_angle

    def get_position_in_segment(self, t):
        if t < self.start_time or t > self.end_time:
            return None  # Не в пределах этого сегмента

        if self.motion_type == 'linear':
            vx, vy, vz = self.params
            delta_t = t - self.start_time
            return self.initial_position + np.array([vx * delta_t, vy * delta_t, vz * delta_t])

        elif self.motion_type == 'circular':
            radius, angular_velocity, vz, direction = self.params
            delta_t = t - self.start_time
            # Рассчитываем текущее смещение по углу с учетом начального угла
            angle = self.initial_angle - direction * angular_velocity * delta_t
            z_movement = vz * delta_t  # Движение по оси z
            return np.array([
                self.center[0] + radius * np.cos(angle),
                self.center[1] + radius * np.sin(angle),
                self.center[2] + z_movement
            ])


class Trajectory:
    def __init__(self):
        self.__segments = []

    def add_segment(self, segment):
        if self.__segments:
            # Автоматически связываем новый сегмент с последним
            segment.initial_position = self.__segments[-1].get_position_in_segment(self.__segments[-1].end_time)
            segment.previous_segment = self.__segments[-1]
        self.__segments.append(segment)

    def get_segments(self):
        return self.__segments

    def get_position(self, t):
        for segment in self.__segments:
            position = segment.get_position_in_segment(t)
            # print(segment.start_time, segment.end_time)
            if position is not None:
                return position
        raise ValueError(f"Время t = {t} вне всех отрезков траектории")
