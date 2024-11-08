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


# ----------------------------------------------------- coords_vizual -----------------------------------------------
# X = np.zeros((n_radars, t2))
# x_true = np.zeros(t2)
# x_estimated = np.zeros(t2)
# sigmas = sm.get_radar_errors()
# print("sigmas =      ", sigmas)
# print("result_sigma =", MathStat.find_res_sigma(sigmas))
#
# e_w = np.zeros(t2)
#
# popravka = np.zeros(n_radars)
# print("popravka each 20 steps")
# for i in range(1, t2):
#     if (i+1)%20 == 0:
#         delta = np.mean(X[0, (i//2 - 1):i]) - np.mean( [np.mean(X[1, (i//2 - 1):i]), np.mean(X[2, (i//2 - 1):i]), np.mean(X[3, (i//2 - 1):i])] )
#         # delta = np.mean(X[0, (i//2 - 1):i]) - np.mean( [np.mean(X[1, (i//2 - 1):i])])#, np.mean(X[2, (i//2 - 1):i]), np.mean(X[3, (i//2 - 1):i])] )
#         popravka[0] = delta
#         print(popravka)
#
#     x_true[i] = list_of_df[0]["x_true"][i]
#     X[:, i] = [df["x_measure"][i] for df in list_of_df]
#     x_estimated[i] = MathStat.weighted_estimator([df["x_measure"][i] for df in list_of_df] - popravka, sigmas)
#     e_w[i] = round(abs(x_true[i] - x_estimated[i]), 5)
#
# x1 = X[0, :]
# x2 = X[1, :]
# x3 = X[2, :]
# x4 = X[3, :]
#
# print(np.mean(x1), np.mean( [np.mean(x2), np.mean(x3), np.mean(x4)] ))
#
#
#
# for i in range(n_radars):
#     plt.plot(np.arange(t2), X[i, :], label='meas')




