import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

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
        group['radius_bin'] = pd.cut(group['r_true'], bins=np.arange(0, group['r_true'].max() + 100, 100))
        errors_by_radius[obj_id] = {}
        
        for radius_bin, radius_group in group.groupby('radius_bin'):
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
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Ошибки в зависимости от радиуса для объекта {obj_id}')
        
        # График ошибок по r
        axs[0].plot(radius_centers, plot_data['Smooth r std'], color='blue', marker='o', label='Smooth r std')
        axs[0].plot(radius_centers, plot_data['r measure std'], color='orange', marker='o', label='r measure std')
        axs[0].set_xlabel('Радиус (r_true)')
        axs[0].set_ylabel('Ошибка')
        axs[0].set_title('Ошибки по координате r')
        axs[0].legend()
        axs[0].grid(True)
        
        # График ошибок по fi
        axs[1].plot(radius_centers, np.array(plot_data['Smooth fi std']) * 180 / np.pi, color='green', marker='o', label='Smooth fi std')
        axs[1].plot(radius_centers, np.array(plot_data['fi measure std']) * 180 / np.pi, color='red', marker='o', label='fi measure std')
        axs[1].set_xlabel('Радиус (r_true)')
        axs[1].set_ylabel('Ошибка')
        axs[1].set_title('Ошибки по координате fi')
        axs[1].legend()
        axs[1].grid(True)
        
        # График ошибок по theta
        axs[2].plot(radius_centers, np.array(plot_data['Smooth theta std']) * 180 / np.pi, color='purple', marker='o', label='Smooth theta std')
        axs[2].plot(radius_centers, np.array(plot_data['theta measure std']) * 180 / np.pi, color='brown', marker='o', label='theta measure std')
        axs[2].set_xlabel('Радиус (r_true)')
        axs[2].set_ylabel('Ошибка')
        axs[2].set_title('Ошибки по координате theta')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_noise_signal_ratio(data):
    """Создает график зависимости отношения шума к сигналу от истинного радиуса для каждого объекта."""
    for obj_id, group in data.groupby('id'):
        plt.figure(figsize=(10, 6))
        plt.plot(group['r_true'], group['noise/signal ratio'], linestyle='-')
        plt.title(f'Зависимость noise/signal ratio от r_true для объекта {obj_id}')
        plt.xlabel('r_true')
        plt.ylabel('noise/signal ratio')
        plt.grid(True)
        plt.show()

def plot_trajectory_in_xy_plane(data, detection_radius):
    """Создает график траектории объектов на плоскости XY с учетом радара."""
    fig, ax = plt.subplots()
    ax.set_xlim(-detection_radius - 10, detection_radius + 10)
    ax.set_ylim(-detection_radius - 10, detection_radius + 10)
    ax.scatter(0, 0, color='red', label='Radar', s=10, zorder=5)
    radar_circle = plt.Circle((0, 0), detection_radius, color='blue', fill=False, linestyle='--', label='Radar Range')
    ax.add_patch(radar_circle)
    
    true_colors = itertools.cycle(['blue', 'purple', 'cyan'])  # Цвета для истинных координат
    measure_colors = itertools.cycle(['orange', 'red', 'yellow'])

    for obj_id, ao_data in data.groupby('id'):
        # Получение цвета для данного объекта
        true_color = next(true_colors)
        measure_color = next(measure_colors)

        # Истинные координаты
        ax.scatter(ao_data['x_true'], ao_data['y_true'], color=true_color, s=10, label=f"Object {obj_id} true coords")
        ax.plot(ao_data['x_true'], ao_data['y_true'], color=true_color, linestyle='-', alpha=0.5, label=f"Object {obj_id} true path")
        # for i, (x, y) in enumerate(zip(ao_data['x_true'], ao_data['y_true'])):
        #     ax.text(x, y, str(i), color='blue', fontsize=5)
        
        # Измеренные координаты
        ax.scatter(ao_data['x_measure'], ao_data['y_measure'], color='orange', s=10, label=f"Object {obj_id} measure coords")
        # for i, (x, y) in enumerate(zip(ao_data['x_measure'], ao_data['y_measure'])):
        #     ax.text(x, y, str(i), color='orange', fontsize=5)

        # Сглаженные измеренные координаты
        ax.scatter(ao_data['x_measure_smooth'], ao_data['y_measure_smooth'], color='green', s=10, label=f"Object {obj_id} measure smooth coords")
        # for i, (x, y) in enumerate(zip(ao_data['x_measure_smooth'], ao_data['y_measure_smooth'])):
        #     ax.text(x, y, str(i), color='green', fontsize=5)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('AirObject Trajectory in XY Plane')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
