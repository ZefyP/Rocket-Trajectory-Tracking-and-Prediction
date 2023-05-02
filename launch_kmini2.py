# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launcher
import os
kmini2_L = Stage(
    name="lower_stage",
    dry_mass= 0.045,            # kg
    fuel_mass=0.073-0.045,      # kg
    thrust= 64,                 # N
    burn_time=1.01,             # s
    diameter=0.0411,            # m
    length = 0.848,             # m
    separation_time= 1, # s
    kml_colour="ffffff00"
)

 
kmini2_U = Stage(
    name="upper_stage",
    dry_mass= 0.028,            # kg
    fuel_mass=0.0,              # kg
    thrust= 0,                 # N
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

missile = Rocket("KMini2", "Sunride", launch_site)
missile.stages = [kmini2_L]
missile.run_simulation()
#missile.plot_altitude_range()
#missile.plot_all()
missile.plot_accel()

for stage in missile.stages:
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stage.export_KML(
        f"output/{missile.name}_{stage.name}.kml",
        downsample_factor=10,
    )