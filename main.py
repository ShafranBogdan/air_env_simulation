import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from simulation import RadarSystem
from simulation import Generator
from simulation import PBU, SimulationManager
from tools import MathStat
from logger import Logger

# пбу в 0, у каждого своя с-ма координаты относ и потом пересчитывать {{{ ЮСТИРОВКА(учет ошибок, чтобы дальше было лучше,
# исключ. систем. ошибки( нр неправ север и надо повернуть с-му) можно промоделировать это и тд)
#
# TODO а что если есть смещенность у какого-то из рлс, оценить её и учесть это (как-то по первым измерениям)
# 
# TODO насколько итоговая ошибка std (после оценки) лучше чем были -> look photo (notes)
# TODO разные errors x2, x10 -> заметно хороша при большом отношении ошибок

#  снова задуматься над физичностью полета в конкретных координатах
#  ещё раз подумать о характерных величинах в реал лайф

# TODO пока что закостылил init_position в generation

def vizual(data, detection_radius):
    # Визуализация
    fig, ax = plt.subplots()
    ax.set_xlim(-detection_radius * 1.5 - 10, detection_radius * 1.5 + 10)
    ax.set_ylim(-detection_radius * 1.5 - 10, detection_radius * 1.5 + 10)

    ax.add_patch(plt.Circle((10000, 10000), detection_radius, fill=False, linestyle='--', label='Radar Range'))
    ax.add_patch(plt.Circle((-10000, 10000), detection_radius, fill=False, linestyle='--', label='Radar Range'))
    ax.add_patch(plt.Circle((-10000, -10000), detection_radius, fill=False, linestyle='--', label='Radar Range'))
    ax.add_patch(plt.Circle((10000, -10000), detection_radius, fill=False, linestyle='--', label='Radar Range'))

    plt.plot(data['x_true'], data['y_true'], label=f"Air object  true coords")

    # plt.draw()
    plt.xlabel('X Coordinate meters')
    plt.ylabel('Y Coordinate meters')
    plt.title('AirObject Trajectory in XY Plane')
    plt.tight_layout()
    plt.show()



detection_radius = 40000
t1 = 0
t2 = 100
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=1, num_seg=2)
air_env = gen.gen_traces()

e1 = 2
e2 = 7
e3 = 5
e4 = 3
radar1 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=detection_radius, air_env=air_env, mean = 15, error=e1)
radar2 = RadarSystem(position=np.array([-10000, 10000, 0]), detection_radius=detection_radius, air_env=air_env, error=e2)
radar3 = RadarSystem(position=np.array([-10000, -10000, 0]), detection_radius=detection_radius, air_env=air_env, error=e3)
radar4 = RadarSystem(position=np.array([10000, -10000, 0]), detection_radius=detection_radius, air_env=air_env, error=e4)

sm = SimulationManager(air_env, PBU([radar1, radar2, radar3, radar4]))  # передавать {ao} временное решение
sm.run(t1, t2)

logger = Logger()
# сохраняем данные в папку /logs
dataframes = sm.get_data()
for i in range(1, len(dataframes) + 1):
    logger.log_dataFrame(dataframes[i - 1], f'logs{i}')

df1 = pd.read_csv("logs/logs1.csv")
df2 = pd.read_csv("logs/logs2.csv")
df3 = pd.read_csv("logs/logs3.csv")
df4 = pd.read_csv("logs/logs4.csv")
list_of_df = [df1, df2, df3, df4]


vizual(df1, detection_radius)

# ----------------------------------------------------- MEAN -----------------------------------------------
# e = np.zeros(t2)
# for i in range(1, t2):
#     x_true = list_of_df[0]["x_true"][i]
#     x_avg = np.mean([df["x_measure"][i] for df in list_of_df])  # среднее координаты по всем радарам
#     e[i] = round(abs(x_true - x_avg), 5)


# ----------------------------------------------------- coords_vizual -----------------------------------------------
X = np.zeros((4, t2))
x_true = np.zeros(t2)
x_estimated = np.zeros(t2)
sigmas = sm.get_radar_errors()
e_w = np.zeros(t2)

popravka = np.zeros(4)
for i in range(1, t2):
    if (i+1)%30 == 0:
        delta = np.mean(X[0, (i//2 - 1):i]) - np.mean( [np.mean(X[1, (i//2 - 1):i]), np.mean(X[2, (i//2 - 1):i]), np.mean(X[3, (i//2 - 1):i])] )
        popravka[0] = delta
        print(popravka)

    x_true[i] = list_of_df[0]["x_true"][i]
    X[:, i] = [df["x_measure"][i] for df in list_of_df]
    x_estimated[i] = MathStat.weighted_estimator([df["x_measure"][i] for df in list_of_df] - popravka, sigmas)
    e_w[i] = round(abs(x_true[i] - x_estimated[i]), 5)

x1 = X[0, :]
x2 = X[1, :]
x3 = X[2, :]
x4 = X[3, :]

print(np.mean(x1), np.mean( [np.mean(x2), np.mean(x3), np.mean(x4)] ))



# for i in range(4):
#     plt.plot(np.arange(t2), X[i, :], label='meas')

# plt.plot(np.arange(t2), x_true, label='true')
plt.plot(np.arange(t2), e_w, label='estimated')

plt.legend()
plt.grid()
plt.show()



# ----------------------------------------------------- WEIGHTS -----------------------------------------------
# e_w = np.zeros(t2)
# sigmas = sm.get_radar_errors()
# for i in range(1, t2):
#     x_true = list_of_df[0]["x_true"][i]
#     x_estimated = MathStat.weighted_estimator([df["x_measure"][i] for df in list_of_df], sigmas)
#     e_w[i] = round(abs(x_true - x_estimated), 5)

# ----------------------------------------------------- VIZUAL -----------------------------------------------



# plt.plot(np.arange(5, t2), e[5:], color="r", label='mean')
# plt.plot(np.arange(1, t2), e_w[1:], color="b", alpha=0.5, label='weights')
# # plt.plot(np.arange(1, t2), e_w2[1:], color="g", alpha=0.5, label='weights2')
# plt.legend()
# plt.grid()
# plt.show()
