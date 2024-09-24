import numpy as np

from simulation import Time
from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def convert_velocity(V: float):
    """
    V in meters in s
    return: V in meters in ms
    """

    return V / 1000

def make_track(t, params):
    for i in range(len(params)):
        t_track = params[i][0]
        if t > t_track:
            continue
        return params[i][1]


t = Time()
# tracks = [[100, lambda t: np.array([convert_velocity(200) * t + 10, convert_velocity(300) * t + 10, 1])]
#           [1000, lambda t: np.array([10 + 200 * np.cos(0.01 * (t - 101)), 110 + 200 * np.sin(0.01 * (t - 101)), 1])]
#           ]
track1 = lambda t: np.array([convert_velocity(200) * t + 10, convert_velocity(300) * t + 10, 1])

track2 = lambda t: (
    np.array([convert_velocity(200) * t + 10, convert_velocity(100) * t + 10, 1]) 
    if t <= 100
    else np.array([10 + 20 * np.cos(0.01 * (t - 101)), 
                   20 + 20 * np.sin(0.01 * (t - 101)), 1])
)

ao = AirObject(track2)
air_env = AirEnv([ao])
detection_radius = 200
radar = RadarSystem(detection_radius=detection_radius, air_env=air_env)
t1 = 0
t2 = 500
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

xdata, ydata = [], []

line, = ax.plot(xdata, ydata, lw=2)

for i in range(len(logs)):
    xdata.append(logs['x_true'].iloc[i])
    ydata.append(logs['y_true'].iloc[i])


    line.set_data(xdata, ydata)
    
    # Перерисовываем график
    plt.draw()
    plt.pause(0.01)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('AirObject Trajectory in XY Plane')
plt.show()
