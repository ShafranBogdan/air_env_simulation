import numpy as np

from simulation import Time
from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem
from simulation import Trajectory, TrajectorySegment

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calc_w(v: float, r: float):
    """
    Расчет угловой скорости рад / мс
    v - м / мс
    r - м
    """
    return v / r

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

# Создание объекта траектории
trajectory = Trajectory()

# Первый сегмент: прямолинейное движение с момента t=0 до t=100 по осям x, y, z
initial_position = [10, 10, 5]  # Начальная точка (x, y, z)

velocity = [convert_velocity(200), 
            convert_velocity(300), 
            convert_velocity(0)
            ]  # Скорости по x, y, z

trajectory.add_segment(TrajectorySegment(0, 100, initial_position, 'linear', velocity))

# Второй сегмент: движение по окружности
radius = 100
angular_velocity = calc_w(convert_velocity(300), radius)
vz = 0
trajectory.add_segment(TrajectorySegment(101, 300, None, 'circular', [radius, angular_velocity, vz], previous_segment=trajectory.segments[-1]))

ao = AirObject(trajectory)
air_env = AirEnv([ao])
detection_radius = 200
radar = RadarSystem(detection_radius=detection_radius, air_env=air_env)
t1 = 0
t2 = 300
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
