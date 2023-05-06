# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launcher
from trajectory_generator import Flight_Data, Wind, drag
from trajectory_generator.constants import TIME_STEP

import os
kmini2_L = Stage(
    name="lower_stage",
    dry_mass= 0.577,            # kg
    fuel_mass=0.650-0.577,      # kg
    thrust= 64,                 # N
    burn_time=1.01,             # s
    diameter=0.0411,            # m
    length = 0.848,             # m
    separation_time= 1, # s
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

launch_site = Launcher(
    "Midlands Rocketry Club, United Kingdom",
    latitude= 52.668250, longitude= -1.524632, altitude = 10,
    azimuth=0, # pointing to true north
    elevation=85 # pointing nearly to zenith
)

flight_data = Flight_Data(
    "KarmanMini2 Relaunch",
    TIME_STEP,
    "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv",
    "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/trajectory_generator/lower_stage_altitude_vs_time.csv"
)

# Wind.simulate_turbulence(20)
cd_filepath = "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/OR_karmanmini2.csv"
drag.get_cd_mach(cd_filepath)


missile = Rocket("KMini2", "Sunride", launch_site)
missile.stages = [kmini2_L]

# missile.run_simulation()

# Save the simulated altitude and compate with real data 
#missile.generate_csv_altitude_vs_time()

# flight_data.plot_data()
# flight_data.environment_analysis()


# missile.plot_altitude_range()
#missile.plot_all()
#missile.plot_accel()
#missile.generate_csv_altitude_vs_time()
# flight_data.compare("C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv",
#     "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/lower_stage_altitude_vs_time.csv"  
# )




for stage in missile.stages:
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stage.export_KML(
        f"output/{missile.name}_{stage.name}.kml",
        downsample_factor=10,
    )