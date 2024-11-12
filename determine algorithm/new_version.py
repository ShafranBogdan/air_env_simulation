import numpy as np
from scipy.stats import chi2
from numpy.linalg import eig, inv
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def spherical_to_cartesian(r, theta, phi, Vr=0, Vtheta=0, Vphi=0):
    # Преобразование координат
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Скорости
    Vx = Vr * np.sin(phi) * np.cos(theta) + r * Vphi * np.cos(phi) * np.cos(theta) - r * Vtheta * np.sin(phi) * np.sin(theta)
    Vy = Vr * np.sin(phi) * np.sin(theta) + r * Vphi * np.cos(phi) * np.sin(theta) + r * Vtheta * np.sin(phi) * np.cos(theta)
    Vz = Vr * np.cos(phi) - r * Vphi * np.sin(phi)
    
    return np.array([x, y, z]), np.array([Vx, Vy, Vz])


def jacobian_transformation_with_speed(spherical_cov, r, theta, phi, V_r, V_theta, V_phi):
    """
    Вычисляет матрицу Якоби для перехода из сферической системы координат (r, theta, phi, Vr, Vtheta, Vphi)
    в декартову систему (x, y, z, Vx, Vy, Vz).
    
    Параметры:
    - r: радиус-вектор
    - theta: азимутальный угол
    - phi: угол места
    - V_r: радиальная скорость
    - V_theta: скорость по азимутальному углу
    - V_phi: скорость по углу места
    
    Возвращает:
    - Преобразованная матрица ковариаций 6x6
    """
    # Частные производные для координат x, y, z
    J = np.zeros((6, 6))
    
    # Для удобства, вычисляем синусы и косинусы заранее
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Координаты
    J[0, 0] = sin_phi * cos_theta                   # dx/dr
    J[0, 1] = -r * sin_phi * sin_theta              # dx/dtheta
    J[0, 2] = r * cos_phi * cos_theta               # dx/dphi
    
    J[1, 0] = sin_phi * sin_theta                   # dy/dr
    J[1, 1] = r * sin_phi * cos_theta               # dy/dtheta
    J[1, 2] = r * cos_phi * sin_theta               # dy/dphi
    
    J[2, 0] = cos_phi                               # dz/dr
    J[2, 2] = -r * sin_phi                          # dz/dphi
    
    # Скорости
    J[3, 0] = V_phi * cos_phi * cos_theta - V_theta * sin_phi * sin_theta  # dVx/dr
    J[3, 1] = -r * V_theta * sin_phi * cos_theta - r * V_phi * cos_phi * sin_theta  # dVx/dtheta
    J[3, 2] = V_r * cos_phi * cos_theta - r * V_phi * sin_phi * cos_theta - r * V_theta * cos_phi * sin_theta  # dVx/dphi
    J[3, 3] = sin_phi * cos_theta  # dVx/dVr
    J[3, 4] = -r * sin_phi * sin_theta  # dVx/dVtheta
    J[3, 5] = r * cos_phi * cos_theta  # dVx/dVphi
    
    J[4, 0] = V_phi * cos_phi * sin_theta + V_theta * sin_phi * cos_theta  # dVy/dr
    J[4, 1] = r * V_theta * sin_phi * sin_theta - r * V_phi * cos_phi * cos_theta  # dVy/dtheta
    J[4, 2] = V_r * cos_phi * sin_theta - r * V_phi * sin_phi * sin_theta + r * V_theta * cos_phi * cos_theta  # dVy/dphi
    J[4, 3] = sin_phi * sin_theta  # dVy/dVr
    J[4, 4] = r * sin_phi * cos_theta  # dVy/dVtheta
    J[4, 5] = r * cos_phi * sin_theta  # dVy/dVphi
    
    J[5, 0] = -V_phi * sin_phi  # dVz/dr
    J[5, 2] = V_r * -sin_phi - r * V_phi * cos_phi  # dVz/dphi
    J[5, 3] = cos_phi  # dVz/dVr
    J[5, 5] = -r * sin_phi  # dVz/dVphi

    cov_cartesian = J @ spherical_cov @ J.T 
    
    return cov_cartesian



def calculate_correlation_ellipsoid_with_rotation(current_coords, current_velocity, delta_t, spherical_cov, a_max=98.1, confidence=0.95):
    """
    Предсказывает диапазон возможных положений с учетом ковариационной матрицы ошибок и ускорения

    Параметры:
    - current_position: numpy array (3,), текущее положение объекта [r, fi, psi]
    - current_velocity: numpy array (3,), текущая скорость объекта [vr, vfi, vpsi]
    - delta_t: float, интервал времени для предсказания
    - spherical_cov: numpy array (6, 6), ковариационная матрица ошибок для [r, fi, psi, vr, vfi, vpsi]
    - a_max: float, максимальное ускорение (10g)
    - confidence_level: float, уровень доверия для эллипсоида

    Возвращает:
    - mean_predicted_position: numpy array (3,), предсказанная координата (центр эллипсоида)
    - covariance_ellipse: numpy array (3, 3), ковариационная матрица для доверительного эллипсоида
    - chi2_val: пороговое значение для расстояния Махаланобиса на основе confidence_level
    """
    # Перевод квариационной матрицы в ПДСК
    ##r, fi, psi = to_sphere_coord(current_coords[0], current_coords[1], current_coords[2])
    ##cov_cartesian = jacobian_transformation_with_speed(spherical_cov, r, fi, psi)
    cov_cartesian = jacobian_transformation_with_speed(spherical_cov, current_coords[0], current_coords[1], current_coords[2],\
                                                       current_velocity[0], current_velocity[1], current_velocity[2])

    # Аппроксимированное среднее положение следующей позиции
    print(current_velocity)
    current_coords_cart, current_velocity_cart = spherical_to_cartesian(current_coords[0], current_coords[1], current_coords[2],\
                                                       current_velocity[0], current_velocity[1], current_velocity[2])
    print(current_velocity_cart)
    
    mean_predicted_position = current_coords_cart + current_velocity_cart * delta_t

    # Максимальное смещение из-за ускорения
    max_accel_displacement = 0.5 * a_max * delta_t**2

    # Матрица возможных изменений координат из-за ускорения
    accel_covariance = max_accel_displacement * np.eye(3)

    # Матрица преобразования - перемещение на следующую координату
    F = np.eye(6)
    F[0:3, 3:6] = delta_t * np.eye(3)  # эта часть отвечает за v * t

    # Преобразуем ковариационную матрицу с помощью модели движения
    predicted_covariance = F @ cov_cartesian @ F.T

    # Ковариационная матрица координат с ошибки поправки скорости
    covariance_ellipse = predicted_covariance[0:3, 0:3]

    # Добавляем ковариацию из-за максимального ускорения
    diagonal_cov = np.zeros_like(covariance_ellipse)
    np.fill_diagonal(diagonal_cov, np.diag(covariance_ellipse))
    non_diagonal_cov = covariance_ellipse - diagonal_cov
    covariance_ellipse = (diagonal_cov**0.5 + accel_covariance)**2 + non_diagonal_cov # Поправил

    # Порог для доверительного эллипсоида по распределению Хи-квадрат
    steps_of_freedom = 3
    quant = chi2.ppf(confidence, steps_of_freedom)  # 95% доверительная вероятность - квантиль хи-2 порядка 0.95

    return mean_predicted_position, covariance_ellipse, quant

def is_within_confidence_ellipse(measured_position, mean_predicted_position, covariance_ellipse, chi2_val):
    """
    Проверяет, находится ли измеренная позиция в пределах доверительного эллипсоида.

    Параметры:
    - measured_position: numpy array (3,), измеренная позиция [r, theta, psi].
    - mean_predicted_position: numpy array (3,), предсказанная позиция.
    - covariance_ellipse: numpy array (3, 3), ковариационная матрица для доверительного эллипсоида.
    - chi2_val: float, пороговое значение для расстояния Махаланобиса.
    """
    measured_position, _ = spherical_to_cartesian(measured_position[0], measured_position[1], measured_position[2])
    diff = measured_position - mean_predicted_position
    mahalanobis_distance = np.sqrt(diff.T @ np.linalg.inv(covariance_ellipse) @ diff)
    return mahalanobis_distance <= chi2_val**0.5



def plot_covariance_ellipse_2d_projection(cov_matrix, mean, measured_position, true_coords=None, idx=-1, confidence_level=0.95):
    """
    Построение двумерного эллипса неопределенности на основе ковариационной матрицы 3x3,
    используя проекцию на координатную плоскость x-y.

    Параметры:
    - cov_matrix: Ковариационная матрица (3x3 numpy array).
    - mean: Среднее значение вектора [x, y, z] - Центр эллипса.
    - confidence_level: Уровень доверия (по умолчанию 0.95 для 95%).
    - measured_position: массив исторических измеренных координат
    - idx: индекс измерения радиолокатора
    """
    # Берем подматрицу ковариации для координат x и y
    cov_matrix_xy = cov_matrix[:2, :2]
    mean_xy = mean[:2]  # Среднее значение для проекции на плоскость x-y

    # Масштабирование эллипса для заданного уровня доверия
    chi2_val = np.sqrt(chi2.ppf(confidence_level, df=2))

    # Собственные значения и собственные векторы для подматрицы x-y
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_xy)

    # Масштабирование осей эллипса по доверительному уровню
    axis_lengths = chi2_val * np.sqrt(eigenvalues)

    # Угол наклона эллипса
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Построение эллипса
    ellipse = Ellipse(xy=mean_xy, width=2*axis_lengths[0], height=2*axis_lengths[1],
                      angle=angle, edgecolor='blue', fc='none', lw=2, label=f'{int(confidence_level*100)}% confidence')

    # Построение графика
    fig, ax = plt.subplots()
    ax.add_patch(ellipse)
    ax.plot(mean[0], mean[1], 'ro', label="Extapolated Position")
    ax.plot(measured_position[-1,0], measured_position[-1,1], 'og', label="Final Measure Position")
    ax.plot(measured_position[:-1,0], measured_position[:-1,1], 'o', label="Measure Positions")
    if true_coords is not None:
        ax.plot(true_coords[0], true_coords[1], 'ob', label="True Position")

    # Настройки графика
    plt.title(f"Траектория для измерения номер {idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.legend()
    ax.grid(True)
    plt.axis('equal')
    plt.show()

def start_simulation(idx_measuring):
    path = r'C:\Users\mi\Documents\НИР\plane_logs.csv'
    data = pd.read_csv(path)

    start_idx = 50
    stop_idx = 1000
    current_coords = data[['r_true', 'fi_true', 'psi_true']].loc[start_idx:stop_idx].values
    current_coords_cartesian = data[['x_true', 'y_true', 'z_true']].loc[start_idx:stop_idx].values
    velocity = data[['v_r_measure', 'v_fi_measure', 'v_psi_measure']].loc[start_idx:stop_idx].values * 1000
    measured_position = data[['r_measure', 'fi_measure', 'psi_measure']].loc[start_idx:stop_idx].values  # измеренная позиция объекта
    measured_position_cartesian = data[['x_measure', 'y_measure', 'z_measure']].loc[start_idx:stop_idx].values
    delta_t = 0.1
    varinces = np.array([data['r_err'][0], data['fi_err'][0], data['psi_err'][0],\
                          np.sqrt(4)*data['r_err'][0]/delta_t, np.sqrt(4)*data['fi_err'][0]/delta_t, np.sqrt(4)*data['psi_err'][0]/delta_t]) ** 2
    cov_matr = np.diag(varinces)


    for idx in idx_measuring:
        # Получаем диапазон возможных значений
        mean_predicted_position, covariance_ellipse, chi2_val = calculate_correlation_ellipsoid_with_rotation(measured_position[idx-1], velocity[idx-1], delta_t, cov_matr)


        # Проверяем, находится ли измеренная позиция в пределах допустимого диапазона
        if not is_within_confidence_ellipse(measured_position[idx], mean_predicted_position, covariance_ellipse, chi2_val):
            print("Point does not belongs to the ellipsoid.")
        else:
            print("Point belongs to the ellipsoid.")

        # Построение проекции эллипсоида на плоскость x-y
        plot_covariance_ellipse_2d_projection(covariance_ellipse, mean_predicted_position,\
                                            measured_position_cartesian[idx-3: idx+1], current_coords_cartesian[idx], idx)
  
start_simulation([60, 510, 810])