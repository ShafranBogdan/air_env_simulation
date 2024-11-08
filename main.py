import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import RadarSystem
from simulation import Generator
from simulation import PBU, SimulationManager
from tools import MathStat, DynamicAlignment
from logger import Logger

#  снова задуматься над физичностью полета в конкретных координатах
#  ещё раз подумать о характерных величинах в реал лайф

# TODO !!! пока что закостылил init_position в generation

# ----------------------------------------------------- Initialization -----------------------------------------------
t1 = 0
t2 = 100
detection_radius = 40000
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=1, num_seg=2)
air_env = gen.gen_traces()

e1 = 2
e2 = 7
e3 = 5
e4 = 13
loc1 = 12
loc2 = 7
loc3 = -5
loc4 = 2

n_radars = 4

radar1 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=detection_radius, air_env=air_env,
                     mean=loc1, error=e1)
radar2 = RadarSystem(position=np.array([-10000, 10000, 0]), detection_radius=detection_radius, air_env=air_env,
                     mean=loc2, error=e2)
radar3 = RadarSystem(position=np.array([-10000, -10000, 0]), detection_radius=detection_radius, air_env=air_env,
                     mean=loc3, error=e3)
radar4 = RadarSystem(position=np.array([10000, -10000, 0]), detection_radius=detection_radius, air_env=air_env,
                     mean=loc4, error=e4)

sm = SimulationManager(air_env, PBU([radar1, radar2, radar3, radar4]))  # передавать {ao} временное решение
sm.run(t1, t2)

# визуализируем радары
sm.visualize()

logger = Logger()
# сохраняем данные в папку /logs
dataframes = sm.get_data()
for i in range(len(dataframes)):
    logger.log_dataFrame(dataframes[i], f'logs{i + 1}')

# Считываем данные
list_of_df = [pd.read_csv(f"logs/logs{i + 1}.csv") for i in range(n_radars)]

# ----------------------------------------------------- DynamicAlignment -----------------------------------------------


x_true = np.zeros(t2)
x_estimated_w_align = np.zeros(t2)
x_estimated = np.zeros(t2)
x_mean = np.zeros(t2)

sigmas = sm.get_radars_errors()
print("sigmas =      ", sigmas)
print("result_sigma after weighted_estimator =", MathStat.find_res_sigma(sigmas))
smoother = DynamicAlignment(n_radars)
smoother.update_ksi(sigmas)

e_estimated_w_align = np.zeros(t2)
e_estimated = np.zeros(t2)
e_mean = np.zeros(t2)

for i in range(1, t2):
    # берем истинные координаты
    x_true[i] = list_of_df[0]["x_true"][i]

    # измеренные координаты от всех радаров
    coords = [df["x_measure"][i] for df in list_of_df]
    # сгладим их
    aligned_coords = smoother.compute_alignments(coords)

    # 1) просто среднее
    x_mean[i] = np.mean(coords)
    e_mean[i] = round(abs(x_true[i] - x_mean[i]), 5)

    # 2) взвешенная оценка без сглаживания
    x_estimated[i] = MathStat.weighted_estimator(coords, sigmas)
    e_estimated[i] = round(abs(x_true[i] - x_estimated[i]), 5)

    # 3) взвешенная оценка + сглаживание
    x_estimated_w_align[i] = MathStat.weighted_estimator(aligned_coords, sigmas)
    e_estimated_w_align[i] = round(abs(x_true[i] - x_estimated_w_align[i]), 5)

step = 5
plt.plot(np.arange(0, t2, step), x_true[::step], label='true')
plt.plot(np.arange(0, t2, step), x_mean[::step], label='mean')
plt.plot(np.arange(0, t2, step), x_estimated[::step], label='estimated')
plt.plot(np.arange(0, t2, step), x_estimated_w_align[::step], label='estimated_w_align')
plt.legend()
plt.grid()
plt.show()

plt.plot(np.arange(0, t2, step), e_mean[::step], label='mean', alpha=0.8)
plt.plot(np.arange(0, t2, step), e_estimated[::step], label='estimated')
plt.plot(np.arange(0, t2, step), e_estimated_w_align[::step], color='r', label='estimated_w_align')
plt.legend()
plt.title("Модуль ошибки относительно истинной координаты")
plt.grid()
plt.show()
