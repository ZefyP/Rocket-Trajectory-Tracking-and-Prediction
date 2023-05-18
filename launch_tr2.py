# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launchsite, Flight_Data
from trajectory_generator.constants import TIME_STEP


flight_data = Flight_Data(
    name = "KarmanMini2 Relaunch Data",
    desired_sample_time = TIME_STEP,
    real_data_path = "example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv",
    sim_filepath ="trajectory_generator/lower_stage_altitude_vs_time.csv",
    OR_data_filepath = "example/tr2_cd.csv"
)

lower_stage = Stage(
    name="Lower stage",
    dry_mass=2.628,
    fuel_mass=(3.141-2.628),
    thrust=808,
    burn_time=1.26+1,
    separation_time=1,
    diameter=0.102,
    kml_colour="ffffff00"
)

upper_stage = Stage(
    name="Upper stage",
    dry_mass=4.798,
    fuel_mass= (5.239-4.798),
    thrust=473,
    burn_time=1.47+0.75,
    diameter=0.102,
    #separation_time=1, #
    kml_colour="ff3c14dc"
)

MRC = Launchsite(
    "Midlands Rocketry Club, United Kingdom",
    latitude = 52.669628, longitude= -1.521624, altitude = 10,
    azimuth = 0,      # pointing to true north
    elevation = 85    # pointing nearly to zenith
)

rocket = Rocket("Test Rocket 2", "Sunride", MRC, use_cd_file=True)
rocket.stages = [lower_stage, upper_stage,]
rocket.run_simulation()
#rocket.plot_all()
#rocket.plot_accel()
rocket.plot_altitude_range()

for stage in rocket.stages:
    stage.export_KML(
        f"output/example_rocket/{rocket.name}_{stage.name}.kml",
        downsample_factor=10,
    )
