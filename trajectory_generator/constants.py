# -*- coding: utf-8 -*-

TRAJECTORY_TIMEOUT = 1000  # seconds before trajectory simulation ends
TIME_STEP = 0.01 # amount of seconds between each calculation
N_TIME_INTERVALS = int(TRAJECTORY_TIMEOUT / TIME_STEP)
EARTH_RADIUS = 6378137  # metres
GRAVITY = 9.80665
mpl_colours = ["royalblue", "darkviolet", "orangered", "gold"]
