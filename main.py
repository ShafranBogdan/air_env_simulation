import numpy as np
from tqdm import tqdm

from simulation import Time
from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem, CoordinateType
from simulation import Trajectory, TrajectorySegment
from simulation import Generator
from simulation import (
    calculate_errors_by_object,
    calculate_errors_by_radius,
    plot_errors_by_radius,
    plot_noise_signal_ratio,
    plot_trajectory_in_xy_plane,
    plot_alpha_beta,
)

import matplotlib.pyplot as plt
import pandas as pd

detection_radius = 100000
t1 = 0
t2 = 5 * 10**4
detection_period = 1

t = Time()
t.set_dt(detection_period)

num_samples = 3
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=num_samples, num_seg=1)
air_env = gen.gen_traces(origin_flag=0)
radar = RadarSystem(detection_radius=detection_radius, 
                    air_env=air_env, 
                    detection_period=detection_period,
                    error=np.array([1., 0.1, 0.1]),
                    sharp_fluctuation_prob=0.,
                    P_ray=150_000,
                    G_recv=40,
                    G_trans=40,
                    lamda=0.0035,
                    rcs_mean=30,
                    tau=0.0001,
                    miss1=1.5,
                    miss2=1.5,
                    miss3=1.,
                    N=5,
                    k1=5, # градус
                    k2=10**6,
                    k=10,
                    )

for ms in tqdm(range(t1, t2, detection_period)):
    radar.trigger()
    t.step()

data = radar.get_data()
data.to_csv("data.csv", index=False)
errors = calculate_errors_by_object(data)
errors_by_radius = calculate_errors_by_radius(data)
# Печать результатов для каждого объекта
for obj_id, error_values in errors.items():
    print(f'Object ID: {obj_id}')
    for error_name, error_value in error_values.items():
        #if error_value > 1000:
        print(f'{error_name} = {error_value}')
    print('---')
plot_alpha_beta(data, CoordinateType.FI)
plot_errors_by_radius(errors_by_radius)
# plot_noise_signal_ratio(data)
plot_trajectory_in_xy_plane(data, detection_radius=detection_radius)
