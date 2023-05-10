# -*- coding: utf-8 -*
#from simulator import Launchsite, Stage, Rocket
from trajectory_generator import Rocket, Stage, Launchsite

 
booster = Stage(
    name="DART Booster",
    dry_mass=14.85,
    fuel_mass=24-14.85,
    thrust=7500,
    burn_time=2.5,
    separation_time=0,
    diameter=120e-3,
    kml_colour="ffffff00"
)

dart = Stage(
    name="SpaceDart",
    dry_mass=5,
    fuel_mass=0,
    thrust=0,
    burn_time=0,
    diameter=40e-3,
    kml_colour="ff3c14dc"
)

launch_site = Launchsite(
    "Romania Launch Site",
    latitude=45.519722, longitude=27.910278, altitude=120,
    azimuth=0, elevation=85
)

missile = Rocket("SpaceDart", "Spacefleet", launch_site)
missile.stages = [booster, dart]
missile.run_simulation()
missile.plot_altitude_range()
missile.plot_all()