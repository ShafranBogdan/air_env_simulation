import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

class CoordinateType(Enum):
    RADIUS = "radius"
    FI = "fi"
    THETA = "theta"

def get_coord_name(coord_type: CoordinateType) -> str:
    if coord_type == CoordinateType.RADIUS:
        return "r"
    elif coord_type == CoordinateType.FI:
        return "fi"
    elif coord_type == CoordinateType.THETA:
        return "theta"

def calculate_errors_by_object(data):
    """Вычисляет ошибки по каждому объекту и сохраняет результаты в словаре."""
    errors = {}
    for obj_id, group in data.groupby('id'):
        errors[obj_id] = {
            'Smooth r std': np.sqrt(((group['r_true'] - group['r_measure_smooth']) ** 2).mean()),
            'r measure std': np.sqrt(((group['r_true'] - group['r_measure']) ** 2).mean()),
            'Smooth fi std': np.sqrt(((group['fi_true'] - group['fi_measure_smooth']) ** 2).mean()),
            'fi measure std': np.sqrt(((group['fi_true'] - group['fi_measure']) ** 2).mean()),
            'Smooth theta std': np.sqrt(((group['theta_true'] - group['theta_measure_smooth']) ** 2).mean()),
            'theta measure std': np.sqrt(((group['theta_true'] - group['theta_measure']) ** 2).mean()),
            'Smooth v_r std': np.sqrt(((group['v_r_true'] - group['v_r_measure_smooth']) ** 2).mean()),
            'v_r measure std': np.sqrt(((group['v_r_true'] - group['v_r_measure']) ** 2).mean()),
            'Smooth v_fi std': np.sqrt(((group['v_fi_true'] - group['v_fi_measure_smooth']) ** 2).mean()),
            'v_fi measure std': np.sqrt(((group['v_fi_true'] - group['v_fi_measure']) ** 2).mean()),
            'Smooth v_theta std': np.sqrt(((group['v_theta_true'] - group['v_theta_measure_smooth']) ** 2).mean()),
            'v_theta measure std': np.sqrt(((group['v_theta_true'] - group['v_theta_measure']) ** 2).mean())
        }
    return errors

def calculate_errors_by_radius(data):
    """Вычисляет ошибки в зависимости от радиуса для каждого объекта и сохраняет результаты в словаре."""
    errors_by_radius = {}
    for obj_id, group in data.groupby('id'):
        group['radius_bin'] = pd.cut(group['r_true'], bins=np.arange(0, group['r_true'].max() + 1000, 1000))
        errors_by_radius[obj_id] = {}
        
        for radius_bin, radius_group in group.groupby('radius_bin', observed=False):
            errors_by_radius[obj_id][radius_bin] = {
                'Smooth r std': np.sqrt(((radius_group['r_true'] - radius_group['r_measure_smooth']) ** 2).mean()),
                'r measure std': np.sqrt(((radius_group['r_true'] - radius_group['r_measure']) ** 2).mean()),
                'Smooth fi std': np.sqrt(((radius_group['fi_true'] - radius_group['fi_measure_smooth']) ** 2).mean()),
                'fi measure std': np.sqrt(((radius_group['fi_true'] - radius_group['fi_measure']) ** 2).mean()),
                'Smooth theta std': np.sqrt(((radius_group['theta_true'] - radius_group['theta_measure_smooth']) ** 2).mean()),
                'theta measure std': np.sqrt(((radius_group['theta_true'] - radius_group['theta_measure']) ** 2).mean())
            }
    return errors_by_radius

def plot_errors_by_radius(errors_by_radius):
    """Создает графики ошибок в зависимости от радиуса для каждого объекта."""
    for obj_id, error_data in errors_by_radius.items():
        radius_centers = [bin_interval.mid for bin_interval in error_data.keys()]
        plot_data = {key: [err[key] for err in error_data.values()] for key in error_data[next(iter(error_data))]}
        font_size = 15  # Размер шрифта
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Ошибки в зависимости от дальности до объекта {obj_id}', fontsize=font_size + 2)
        

        # График ошибок по r
        axs[0].scatter(radius_centers, plot_data['Smooth r std'], color='blue', marker='o', label='СКО сглаженной координаты')
        axs[0].scatter(radius_centers, plot_data['r measure std'], color='orange', marker='o', label='СКО измеренной координаты')
        axs[0].set_xlabel('Дальность до объекта, м', fontsize=font_size)
        axs[0].set_ylabel('Ошибка, м', fontsize=font_size)
        axs[0].set_title('Ошибки по координате R', fontsize=font_size + 2)
        axs[0].legend(fontsize=font_size - 2)
        axs[0].grid(True)

        # График ошибок по fi
        axs[1].scatter(radius_centers, np.array(plot_data['Smooth fi std']) * 180 / np.pi, color='green', marker='o', label='СКО сглаженной координаты')
        axs[1].scatter(radius_centers, np.array(plot_data['fi measure std']) * 180 / np.pi, color='red', marker='o', label='СКО измеренной координаты')
        axs[1].set_xlabel('Дальность до объекта, м', fontsize=font_size)
        axs[1].set_ylabel('Ошибка, °', fontsize=font_size)
        axs[1].set_title('Ошибки по координате φ', fontsize=font_size + 2)
        axs[1].legend(fontsize=font_size)
        axs[1].grid(True)

        # График ошибок по theta
        axs[2].scatter(radius_centers, np.array(plot_data['Smooth theta std']) * 180 / np.pi, color='purple', marker='o', label='СКО сглаженной координаты')
        axs[2].scatter(radius_centers, np.array(plot_data['theta measure std']) * 180 / np.pi, color='brown', marker='o', label='СКО измеренной координаты')
        axs[2].set_xlabel('Дальность до объекта, м', fontsize=font_size)
        axs[2].set_ylabel('Ошибка, °', fontsize=font_size)
        axs[2].set_title('Ошибки по координате θ', fontsize=font_size + 2)
        axs[2].legend(fontsize=font_size)
        axs[2].grid(True)

        # Увеличение шрифта для общего оформления
        plt.xticks(fontsize=font_size - 2)
        plt.yticks(fontsize=font_size - 2)

        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_alpha_beta(data, coord_type):
    """
    Строит графики коэффициентов α и β для каждой траектории отдельно.
    
    Параметры:
    - data: DataFrame с данными.
    - coord_type: Тип координаты (RADIUS, FI, THETA).
    """
    # Определение столбцов для α и β в зависимости от типа координаты
    if coord_type == CoordinateType.RADIUS:
        alpha_col = 'alpha_r'
        beta_col = 'beta_r'
    elif coord_type == CoordinateType.FI:
        alpha_col = 'alpha_fi'
        beta_col = 'beta_fi'
    elif coord_type == CoordinateType.THETA:
        alpha_col = 'alpha_theta'
        beta_col = 'beta_theta'
    else:
        raise ValueError(f"Unknown coordinate type: {coord_type}")
    
    # Проверка наличия необходимых столбцов
    if alpha_col not in data.columns or beta_col not in data.columns:
        raise ValueError(f"Missing required columns: {alpha_col}, {beta_col} in data.")
    
    # Группировка данных по идентификаторам объектов
    for obj_id, obj_data in data.groupby('id'):
        # Создание нового окна для каждого объекта
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # График alpha
        axes[0].plot(obj_data['r_true'], obj_data[alpha_col], label=f'{alpha_col}', color='blue', alpha=0.7)
        axes[0].set_title(f'Коэффициент α для координаты {coord_type} (Объект {obj_id})')
        axes[0].set_xlabel('Дальность до объекта, м')
        axes[0].set_ylabel('Коэффициент α')
        axes[0].grid(True)
        axes[0].legend()
        
        # График beta
        axes[1].plot(obj_data['r_true'], obj_data[beta_col], label=f'{beta_col}', color='red', alpha=0.7)
        axes[1].set_title(f'Коэффициент β для координаты {coord_type} (Объект {obj_id})')
        axes[1].set_xlabel('Дальность до объекта, м')
        axes[1].set_ylabel('Коэффициент β')
        axes[1].grid(True)
        axes[1].legend()
        
        # Настройка общего заголовка и отображение
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_noise_signal_ratio(data):
    """Создает график зависимости отношения шума к сигналу от истинного радиуса для каждого объекта."""
    font_size = 15
    for obj_id, group in data.groupby('id'):
        plt.figure(figsize=(10, 6))
        plt.plot(group['r_true'], group['noise/signal ratio'], linestyle='-')
        plt.title(f'Зависимость отношения сигнал/шум от дальности до объекта {obj_id}', fontsize=font_size)
        plt.xlabel('Дальность до объекта, м', fontsize=font_size)
        plt.ylabel('Отношение сигнал/шум, дБ', fontsize=font_size)
        plt.grid(True)
        plt.show()

def plot_trajectory_in_xy_plane(data, detection_radius):
    """Создает график траектории объектов на плоскости XY с учетом радара."""
    fig, ax = plt.subplots()
    ax.set_xlim(-detection_radius - 10, detection_radius + 10)
    ax.set_ylim(-detection_radius - 10, detection_radius + 10)
    ax.scatter(0, 0, color='red', label='Радар', s=10, zorder=5)
    radar_circle = plt.Circle((0, 0), detection_radius, color='blue', fill=False, linestyle='--', label='Область обнаружения радара')
    ax.add_patch(radar_circle)


    for obj_id, ao_data in data.groupby('id'):

        # Истинные координаты
        ax.scatter(ao_data['x_true'], ao_data['y_true'], color='blue', s=10, label=f"Истинные координаты объекта {obj_id}")
        ax.plot(ao_data['x_true'], ao_data['y_true'], color='blue', linestyle='-', alpha=0.5, label=f"Истинный путь объекта {obj_id}")
        # for i, (x, y) in enumerate(zip(ao_data['x_true'], ao_data['y_true'])):
        #     ax.text(x, y, str(i), color='blue', fontsize=7)
        
        # Измеренные координаты
        ax.scatter(ao_data['x_measure'], ao_data['y_measure'], color='red', s=10, label=f"Измеренные координаты объекта {obj_id}")
        # for i, (x, y) in enumerate(zip(ao_data['x_measure'], ao_data['y_measure'])):
        #     ax.text(x, y, str(i), color='red', fontsize=7)

        # Сглаженные измеренные координаты
        ax.scatter(ao_data['x_measure_smooth'], ao_data['y_measure_smooth'], color='green', s=10, label=f"Сглаженные координаты объекта {obj_id}")
        # for i, (x, y) in enumerate(zip(ao_data['x_measure_smooth'], ao_data['y_measure_smooth'])):
        #     ax.text(x, y, str(i), color='green', fontsize=7)

    ax.set_xlabel('Координата x, м')
    ax.set_ylabel('Координата y, м')
    ax.set_title('Траектории в XY плоскости')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.show()
