import numpy as np
from tqdm import tqdm

from simulation import Time
from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem
from simulation import Trajectory, TrajectorySegment
from simulation import Generator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t = Time()

detection_radius = 40000
t1 = 0
t2 = 10**5
num_samples = 5
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=num_samples, num_seg=2)
air_env = gen.gen_traces()
radar = RadarSystem(detection_radius=detection_radius, 
                    air_env=air_env, 
                    detection_period=100, 
                    error=np.array([1., 0.001, 0.001]), 
                    sharp_fluctuation_prob=0.2,
                    P_ray=150000,
                    G_recv=50,
                    G_trans=50,
                    lamda=0.0035,
                    sigma=30,
                    tau=0.0001,
                    miss1=1.5,
                    miss2=1.5,
                    miss3=1.,
                    N=5,
                    k1=3, # градус
                    k2=2*10**6,
                    )
for ms in range(t1, t2):
    radar.trigger()
    t.step()

logs = radar.get_data()
errors = {}

# Группируем данные по id летательных объектов
for obj_id, group in logs.groupby('id'):
    diff_r_1 = np.sqrt(((group['r_true'] - group['r_measure_smooth']) ** 2).mean())
    diff_r_2 = np.sqrt(((group['r_true'] - group['r_measure']) ** 2).mean())
    diff_fi_1 = np.sqrt(((group['fi_true'] - group['fi_measure_smooth']) ** 2).mean())
    diff_fi_2 = np.sqrt(((group['fi_true'] - group['fi_measure']) ** 2).mean())
    diff_psi_1 = np.sqrt(((group['theta_true'] - group['theta_measure_smooth']) ** 2).mean())
    diff_psi_2 = np.sqrt(((group['theta_true'] - group['theta_measure']) ** 2).mean())
    diff_vr_1 = np.sqrt(((group['v_r_true'] - group['v_r_measure_smooth']) ** 2).mean())
    diff_vr_2 = np.sqrt(((group['v_r_true'] - group['v_r_measure']) ** 2).mean())
    diff_vfi_1 = np.sqrt(((group['v_fi_true'] - group['v_fi_measure_smooth']) ** 2).mean())
    diff_vfi_2 = np.sqrt(((group['v_fi_true'] - group['v_fi_measure']) ** 2).mean())
    diff_vtheta_1 = np.sqrt(((group['v_theta_true'] - group['v_theta_measure_smooth']) ** 2).mean())
    diff_vtheta_2 = np.sqrt(((group['v_theta_true'] - group['v_theta_measure']) ** 2).mean())
    
    # Сохраняем ошибки для текущего объекта
    errors[obj_id] = {
        'Smooth r std': diff_r_1,
        'r measure std': diff_r_2,
        'Smooth fi std': diff_fi_1,
        'fi measure std': diff_fi_2,
        'Smooth theta std': diff_psi_1,
        'theta measure std': diff_psi_2,
        'Smooth v_r std': diff_vr_1,
        'v_r measure std': diff_vr_2,
        'Smooth v_fi std': diff_vfi_1,
        'v_fi measure std': diff_vfi_2,
        'Smooth v_theta std': diff_vtheta_1,
        'v_theta measure std': diff_vtheta_2
    }

# Печать результатов для каждого объекта
for obj_id, error_values in errors.items():
    print(f'Object ID: {obj_id}')
    for error_name, error_value in error_values.items():
        print(f'{error_name} = {error_value}')
    print('---')
logs.to_csv("data.csv", index=False)

# Создаем графики для каждого объекта
for obj_id, group in logs.groupby('id'):
    plt.figure(figsize=(10, 6))
    plt.plot(group['r_true'], group['noise/signal ratio'], linestyle='-')
    plt.title(f'Зависимость noise/signal ratio от r_true для объекта {obj_id}')
    plt.xlabel('r_true')
    plt.ylabel('noise/signal ratio')
    plt.grid(True)
    plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-detection_radius - 10, detection_radius + 10)
ax.set_ylim(-detection_radius - 10, detection_radius + 10)
#Radar
ax.scatter(0, 0, color='red', label='Radar', s=10, zorder=5)
radar_circle = plt.Circle((0, 0), detection_radius, color='blue', fill=False, linestyle='--', label='Radar Range')
ax.add_patch(radar_circle)

for obj_id, ao_data in logs.groupby('id'):
    
    ax.scatter(ao_data['x_true'], ao_data['y_true'], color='blue', label=f"Air object {obj_id} true coords (points)", s=10)
    
    ax.plot(ao_data['x_measure'], ao_data['y_measure'], color='orange', linestyle='-', label=f"Air object {obj_id} measure coords (line)")
    
    ax.plot(ao_data['x_measure_smooth'], ao_data['y_measure_smooth'], color='green', linestyle='--', label=f"Air object {obj_id} measure smooth coords (line)")

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('AirObject Trajectory in XY Plane')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
