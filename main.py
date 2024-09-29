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

detection_radius = 4000
t1 = 0
t2 = 10**6
num_samples = 3
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=num_samples, num_seg=2)
air_env = gen.gen_traces()
radar = RadarSystem(detection_radius=detection_radius, air_env=air_env, detection_period=100)
for ms in range(t1, t2):
    radar.trigger()
    t.step()

logs = radar.get_data()
logs.to_csv("logs.csv", index=False)
# print(logs)

fig, ax = plt.subplots()
ax.set_xlim(-detection_radius - 10, detection_radius + 10)
ax.set_ylim(-detection_radius - 10, detection_radius + 10)
#Radar
ax.scatter(0, 0, color='red', label='Radar', s=10, zorder=5)
radar_circle = plt.Circle((0, 0), detection_radius, color='blue', fill=False, linestyle='--', label='Radar Range')
ax.add_patch(radar_circle)

for id, ao_data in logs.groupby('id'):
    ax.plot(ao_data['x_true'], ao_data['y_true'], label=f"Air object {id} true coords")
    # ax.plot(ao_data['x_measure'], ao_data['y_measure'], label=f"Air object {id} measure coords")

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('AirObject Trajectory in XY Plane')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
