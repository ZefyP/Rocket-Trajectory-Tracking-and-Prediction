# -*- coding: utf-8 -*-

from trajectory_generator import Missile, Stage, Launcher
import os
kmini2 = Stage(
    name="Karman Mini 2",
    dry_mass= 45,           # g
    fuel_mass=73-45,        # g
    thrust=68000,           # mN
    burn_time=1.01,         # s
    diameter=0.411,         # metres?
    kml_colour="ffffff00"
)

launch_site = Launcher(
    "Midlands Rocketry Club, United Kingdom",
    latitude=52.668250, longitude=-1.524632, altitude=10,
    azimuth=0, # pointing to true north
    elevation=85 # pointing nearly to zenith
)

missile = Missile("KMini2", "Sunride", launch_site)
missile.stages = [kmini2]
missile.run_simulation()
missile.plot_altitude_range()
missile.plot_all()

for stage in missile.stages:
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stage.export_KML(
        f"output/{missile.name}_{stage.name}.kml",
        downsample_factor=10,
    )