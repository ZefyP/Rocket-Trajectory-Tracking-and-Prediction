# -*- coding: utf-8 -*
from simulator import Launcher, Stage, Rocket

 
lower_stage = Stage(
    name="Lower stage",
    dry_mass=1000,          # kg
    fuel_mass=5000,         # kg
    thrust=300e3,           # Newtons
    burn_time=10,           # s 
    separation_time=1,      # s
    diameter=0.5,           # m
    kml_colour="ffffff00"
)

upper_stage = Stage(
    name="Upper stage",
    dry_mass=500,
    fuel_mass=2500,
    thrust=50e3,
    burn_time=10,
    diameter=0.35,
    kml_colour="ff3c14dc"
)

FAR = Launcher(
    "FAR",
    latitude=35.422510, longitude=-117.732800, altitude=30
    , azimuth=0, elevation=85
    # some uk site 
    # latitude=57.360934, longitude=-7.402981, altitude=30,
    # azimuth=-90, elevation=85
)



missile = Rocket("Example Dart", "Sunride", FAR)
missile.stages = [lower_stage, upper_stage]
missile.run_simulation()
missile.plot_all()

for stage in missile.stages:
    stage.export_KML(
        f"output/{missile.name}_{stage.name}.kml",
        downsample_factor=10,
    )


