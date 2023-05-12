# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launchsite
from trajectory_generator import Real_Data, Wind, drag
from trajectory_generator.constants import TIME_STEP
from trajectory_generator.atmosphere import Atmosphere
import os

real_data = Real_Data(
    name = "KarmanMini2 Relaunch Data",
    desired_sample_time = TIME_STEP,
    real_data_path= "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/TR2_EasyMega_Flight_Data.csv",
    #real_data_path = "example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv",
    sim_filepath ="trajectory_generator/lower_stage_altitude_vs_time.csv",
    OR_data_filepath = "example/OR_karmanmini2.csv"
)

kmini2_L = Stage(
    name="lower_stage",
    dry_mass= 0.577,            # kg
    fuel_mass=0.650-0.577,      # kg
    thrust= 64,                 # N
    burn_time=1.01,             # s
    diameter=0.0411,            # m
    length = 0.848,             # m
    separation_time= 1,         # s
    kml_colour="ffffff00"
)

kmini2_U = Stage(
    name="upper_stage",
    dry_mass= 0.577,            # kg
    fuel_mass=0.0,              # kg
    thrust= 0,                  # N
    burn_time=0.0,              # s
    diameter=0.0411,            # m
    kml_colour="ffffff00"
)

launch_site = Launchsite(
    "Midlands Rocketry Club, United Kingdom",
    latitude = 52.668250, longitude= -1.524632, altitude = 10,
    azimuth = 0,      # pointing to true north
    elevation = 85    # pointing nearly to zenith
)

# rocket = Rocket("KMini2", "Sunride", launch_site, use_cd_file = False, real_data = real_data)

rocket = Rocket("KMini2", "Sunride", launch_site, use_cd_file = False)

rocket.stages = [kmini2_L]
rocket.run_simulation()


# Save the simulated altitude and compate with real data 
# rocket.generate_csv_altitude_vs_time()
# rocket.plot_altitude_range()

# dens = Atmosphere(364)
# print("this is hte density!!", dens.density)

# Wind.simulate_turbulence(20)
# real_data.plot_mach_cd()
# real_data.get_time_accel_z()

# real_data.plot_data()
# real_data.environment_analysis()

# """ TR2 Recorded Flight 1 (single stage )"""
# file = real_data.real_data_path
# # file = "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/TR2_EasyMega_Flight_Data.csv"
# time, _ = real_data.read_csv_col(file, 3)
# time = real_data.resample_array(time,TIME_STEP) # refitted to match the simulation TIME_STEP

# accel_x,_ = real_data.read_csv_col(file,15)
# accel_x = real_data.resample_array(accel_x,TIME_STEP) 

# accel_y,_ = real_data.read_csv_col(file,16)
# accel_y = real_data.resample_array(accel_y,TIME_STEP)

# accel_z,_ = real_data.read_csv_col(file,17)
# accel_z = real_data.resample_array(accel_z,TIME_STEP)

# alt_recorded,_ = real_data.read_csv_col(file,8)
# alt_recorded = real_data.resample_array(alt_recorded,TIME_STEP) 

# V_recorded,_ = real_data.read_csv_col(file,10)
# V_recorded = real_data.resample_array(V_recorded,TIME_STEP) 





# DEBUG
# print(time,accel_x,accel_y,accel_z, alt_recorded, V_recorded)
# print(len(time), len(accel_x), len(accel_y), len(accel_y),  len(alt_recorded), len(V_recorded) )


# new_time = len(new_time)
# print(new_time)

#rocket.plot_all()
#rocket.plot_accel()
#rocket.generate_csv_altitude_vs_time()
# real_data.compare("C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv")

for stage in rocket.stages:
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stage.export_KML(
        f"output/{rocket.name}_{stage.name}.kml",
        downsample_factor=10,
    )