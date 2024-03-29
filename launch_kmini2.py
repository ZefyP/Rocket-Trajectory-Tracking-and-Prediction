# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launchsite
from trajectory_generator import Flight_Data, Wind, drag
from trajectory_generator.constants import TIME_STEP
from trajectory_generator.atmosphere import Atmosphere
import os

# temp for wind test
import matplotlib.pyplot as plt

flight_data = Flight_Data(
    name = "KarmanMini2 Relaunch Data",
    desired_sample_time = TIME_STEP,
    real_data_path = "example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv",
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
    latitude = 52.669628, longitude= -1.521624, altitude = 10,
    azimuth = 0,      # pointing to true north
    elevation = 85    # pointing nearly to zenith
)

# rocket = Rocket("KMini2", "Sunride", launch_site, use_cd_file = False, flight_data = flight_data)

rocket = Rocket("KMini2", "Sunride", launch_site, use_cd_file = False)

rocket.stages = [kmini2_L]
rocket.run_simulation()


""" Test the rocket landing spot method """
wind = Wind(
            altitude = 255,
            wind_speed = 5,
            wind_direction = 'north',
            frequency = 20 # hz for noise sample. 20hz is every 50th sample
            #effective_area = Cd * A

            ) 

""" Generate desired wind profile or load an existing """
#wind.plot_wind(altitude, 'north') 
#wind.generate_wind_profile(0,500) # use only once and then comment out
wp_filepath = "C:\ergasia\projects\Rocket-Trajectory-Tracking-and-Prediction\wind_profile_0-50000_5_north_20.csv"
# wind.plot_wind_profile(wp_filepath)


""" Test Landing Area prediction """
#rocket = Rocket(apogee=1000, apogee_direction=90)  # apogee of 10000 feet, direction straight up
#landing_spot = wind.land_spot()
#print(landing_spot)  # prints the x and y coordinates of the landing spot
# wind.landing_area()

"""Save the simulated altitude and compare with real data """
# rocket.generate_csv_altitude_vs_time()
#rocket.plot_altitude_range()
rocket.plot_all()
# dens = Atmosphere(364)
# print("this is hte density!!", dens.density)

# Wind.generate_gusts(20)
# flight_data.plot_mach_cd()
# flight_data.get_time_accel_z()

# flight_data.plot_data()
# flight_data.environment_analysis()



#rocket.plot_all()
#rocket.plot_accel()
#rocket.generate_csv_altitude_vs_time()
# flight_data.compare("C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv")

for stage in rocket.stages:
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stage.export_KML(
        f"output/{rocket.name}_{stage.name}.kml",
        downsample_factor=10,
    )