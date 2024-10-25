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
        Расчет оценки (несмещенной и с мин дисперсией) координты цели по данным от нескольких рлс
        x = ∑ᵢ₌₁ⁿ wᵢ xᵢ / ∑ᵢ₌ⁿ wᵢ ; i=1..m, m - кол-во рлс
        """
        v = [1 / s**2 for s in sigma]
        ss = np.sum(v)
        weights = np.array([v_i / ss for v_i in v])
        return np.sum(x*weights)

