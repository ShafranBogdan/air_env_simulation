import numpy as np

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
radar = RadarSystem(detection_radius=detection_radius, air_env=air_env, detection_period=100, error=50)
for ms in range(t1, t2):
    radar.trigger()
    t.step()

logs = radar.get_data()
diff_coord_1 = np.sqrt(((logs['r_true'] - logs['r_measure_smooth']) ** 2).mean())
diff_coord_2 = np.sqrt(((logs['r_true'] - logs['r_measure']) ** 2).mean())
diff_v_1 = np.sqrt(((logs['v_r_true'] - logs['v_r_measure_smooth']) ** 2).mean())
diff_v_2 = np.sqrt(((logs['v_r_true'] - logs['v_r_measure']) ** 2).mean())
print(f'Smooth r std = {diff_coord_1}, r measure std = {diff_coord_2}')
print(f'Smooth v_r std = {diff_v_1}, v_r measure std = {diff_v_2}')
logs.to_csv("logs.csv", index=False)

fig, ax = plt.subplots()
ax.set_xlim(-detection_radius - 10, detection_radius + 10)
ax.set_ylim(-detection_radius - 10, detection_radius + 10)
#Radar
ax.scatter(0, 0, color='red', label='Radar', s=10, zorder=5)
radar_circle = plt.Circle((0, 0), detection_radius, color='blue', fill=False, linestyle='--', label='Radar Range')
ax.add_patch(radar_circle)

for id, ao_data in logs.groupby('id'):
    ax.plot(ao_data['x_true'], ao_data['y_true'], label=f"Air object {id} true coords")
    if ao_data['x_measure_extr'] is not None:
        ax.plot(ao_data['x_measure_extr'], ao_data['y_measure_extr'], label=f"Air object {id} measure_extr coords")

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('AirObject Trajectory in XY Plane')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
