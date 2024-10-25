# --- экспоненциальное сглаживание
# eps = np.array([0.7, 0.6])
# m = 2
# dx_volna = np.ndarray((tm, m))
# dx_volna[0] = [0, 0]
#
# e_eps = np.zeros(tm)
# for i in range(1, tm):
#     x_true = list_of_df[0]["x_true"][i]
#     x = [df["x_measure"][i] for df in list_of_df]
#     x_avg = np.mean(x)  # среднее координаты по всем радарам
#     dx = np.array([(x_avg - df["x_measure"][i]) for df in list_of_df])
#     tmp = []
#     for q in range(m):
#         tmp.append((1 - eps[q]) * dx[q] + eps[q] * dx_volna[i][q])
#     dx_volna[i] = tmp
#
#     x_volna = np.mean(x + dx_volna)
#     e_eps[i] = round(abs(x_true - x_volna), 8)
# print(dx_volna)

# ----------- Создание объекта траектории
# trajectory = Trajectory()
#
# # Первый сегмент: прямолинейное движение с момента t=0 до t=100 по осям x, y, z
# initial_position = [0, 0, 5]  # Начальная точка (x, y, z)
#
# velocity = [Physic.convert_velocity(220),
#             Physic.convert_velocity(220),
#             Physic.convert_velocity(0)]  # Скорости по x, y, z
# trajectory.add_segment(TrajectorySegment(0, 300, initial_position, 'linear', velocity))
#
# # Второй сегмент: движение по окружности
# radius, vz = 100, 0
# angular_velocity = Physic.calc_w(Physic.convert_velocity(300), radius)
# trajectory.add_segment(TrajectorySegment(301, 1000, None, 'circular', [radius, angular_velocity, vz],
#                                          previous_segment=trajectory.segments[-1]))




import numpy as np


sigma = [4, 4]

v = [1 / s**2 for s in sigma]

ss = np.sum(v)
s_res = ss**(-0.5)

print( (ss/v[0])**0.5 )




