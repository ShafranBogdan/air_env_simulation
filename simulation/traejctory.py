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
            Для 'circular' - (radius, angular_velocity, vz) - параметры окружности и движение по оси z
        - previous_segment: предыдущий сегмент для вычисления начальной точки
        """
        self.start_time = start_time
        self.end_time = end_time
        self.motion_type = motion_type
        self.params = params

        # Если начальная позиция не задана, вычисляем её по предыдущему сегменту
        if initial_position is None and previous_segment:
            self.initial_position = previous_segment.get_position(previous_segment.end_time)
        else:
            self.initial_position = np.array(initial_position)

        # Рассчитываем центр окружности для кругового движения, если нужно
        if self.motion_type == 'circular' and previous_segment:
            radius, angular_velocity, _ = self.params
            self.center = self.calculate_center_from_last_point(radius)

    def calculate_center_from_last_point(self, radius):
        """
        Рассчитывает центр окружности, чтобы траектория начиналась в последней точке предыдущего сегмента.
        """
        # Начальная точка окружности совпадает с последней точкой предыдущей траектории.
        x0, y0, _ = self.initial_position

        # Центр окружности находится на радиусе от начальной точки
        return np.array([x0 - radius, y0])

    def get_position(self, t):
        if t < self.start_time or t > self.end_time:
            return None  # Не в пределах этого сегмента

        if self.motion_type == 'linear':
            vx, vy, vz = self.params
            delta_t = t - self.start_time
            return self.initial_position + np.array([vx * delta_t, vy * delta_t, vz * delta_t])

        elif self.motion_type == 'circular':
            center_x, center_y, radius, angular_velocity, vz = self.center[0], self.center[1], self.params[0], self.params[1], self.params[2]
            delta_t = t - self.start_time
            angle = angular_velocity * delta_t
            z_movement = vz * delta_t  # Движение по оси z
            return np.array([
                center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle),
                self.initial_position[2] + z_movement
            ])

class Trajectory:
    def __init__(self):
        self.segments = []

    def add_segment(self, segment):
        if self.segments:
            # Автоматически связываем новый сегмент с последним
            segment.initial_position = self.segments[-1].get_position(self.segments[-1].end_time)
            segment.previous_segment = self.segments[-1]
        self.segments.append(segment)

    def get_position(self, t):
        for segment in self.segments:
            position = segment.get_position(t)
            if position is not None:
                return position
        return None  # Время вне всех сегментов
