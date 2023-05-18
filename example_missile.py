# -*- coding: utf-8 -*-

from trajectory_generator import Rocket, Stage, Launchsite

lower_stage = Stage(
    name="Lower stage",
    dry_mass=1000,
    fuel_mass=5000,
    thrust=300e3,
    burn_time=10,
    separation_time=1,
    diameter=0.5,
    kml_colour="ffffff00"
)

upper_stage = Stage(
    name="Upper stage",
    dry_mass=500,
    fuel_mass=2500,
    thrust=50e3,
    burn_time=10,
    diameter=0.35,
    separation_time=50,
    kml_colour="ff3c14dc"
)

another_stage = Stage(
    name="3rd stage",
    dry_mass=20,
    fuel_mass=0,
    thrust=0,
    burn_time=10,
    diameter=0.2,
    kml_colour="ff91ff00"
)

hebrides = Launchsite(
    "Hebrides",
    latitude=57.360934, longitude=-7.402981, altitude=30,
    azimuth=-90, elevation=85
)

missile = Rocket("Example Rocket", "BAE Systems", hebrides)
missile.stages = [lower_stage, upper_stage, another_stage]
missile.run_simulation()
#missile.plot_all()
missile.plot_accel()

for stage in missile.stages:
    stage.export_KML(
        f"output/example_rocket/{missile.name}_{stage.name}.kml",
        downsample_factor=10,
    )
