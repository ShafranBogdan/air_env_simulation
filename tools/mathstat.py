import numpy as np


class MathStat:
    @staticmethod
    def s2(x):
        """
        Расчет выборочной дисперсии
        несмещ, сост и r-эфф(т.е. эфф в классе регулярных оценок)
        1/(n-1)*sum(xi-<x>)^2
        """
        n = len(x)
        x_mean = np.mean(x)
        return np.sum([(x_i - x_mean) ** 2 for x_i in x]) / (n - 1)

    @staticmethod
    def weighted_estimator(x: np.array, sigma):
        """
        Расчет оценки (несмещенной и с минимальной дисперсией) координаты цели по данным от нескольких рлс
        x = ∑ᵢ₌₁ⁿ wᵢ xᵢ / ∑ᵢ₌ⁿ wᵢ ; i=1..m, m - кол-во рлс
        """
        v = [1 / s ** 2 for s in sigma]
        ss = np.sum(v)
        weights = np.array([v_i / ss for v_i in v])
        return np.sum(x * weights)

    @staticmethod
    def find_res_sigma(sigma):
        """
        Расчет результирующего среднего отклонения (error) после весовой оценки weighted_estimator()
        sigma_res = m / ∑ᵢ₌ⁿ vᵢ ; i=1..m, m - кол-во рлс
        """
        v = [1 / s ** 2 for s in sigma]
        return (len(sigma) / np.sum(v)) ** 0.5


class DynamicAlignment:
    """
    Класс для реализации динамической юстировки с помощью экспоненциального сглаживания.
    Коэффициенты при сглаживании (ksi) рассчитываются из соображений минимизации диспресии случайной ошибки усреднения.
    """

    def __init__(self, num_radars, max_ksi=0.99):
        self.m = num_radars  # количество радаров
        self.max_ksi = max_ksi  # пороговое значение коэф, выше которого нет смысла брать
        self.ksi_values = np.ones(num_radars) * max_ksi  # сами коэффициенты
        self.prev_alignments = np.zeros(num_radars)  # предыдущие сглаженные координаты

    def compute_alignments(self, coordinates):
        mean_coordinate = np.mean(coordinates)
        deviations = mean_coordinate - coordinates

        # экспоненциальное сглаживание
        alignments = (1 - self.ksi_values) * deviations + self.ksi_values * self.prev_alignments
        aligned_coordinates = coordinates + alignments

        self.prev_alignments = alignments
        return aligned_coordinates

    def update_ksi(self, sigmas):
        """
        Динамически обновляем коэффициенты ksi, т.к. в ходе моделирования ошибки (sigmas) радаров могут меняться
        """
        sum_of_squared_sigmas = np.sum([s ** 2 for s in sigmas])
        for j in range(self.m):
            self.ksi_values[j] = min(self.max_ksi, np.sqrt(
                (self.m * (self.m - 2) + sum_of_squared_sigmas / sigmas[j] ** 2) / (self.m * (self.m - 1))) - 1)
