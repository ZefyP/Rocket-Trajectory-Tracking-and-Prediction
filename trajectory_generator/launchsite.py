# -*- coding: utf-8 -*-

"""
This class represents the rocket launcher on the launch pad. 
It utilises the launcher's name, location, orientation, and initial velocity.
It includes methods to convert the launcher's latitude, longitude, and altitude to ECEF coordinates, 
as well as calculate the unit vectors for the rocket's initial position and orientation on the launcher pad.
"""

import numpy as np
from .utils import lla2ecef, aer2ecef_unit_vector


class Launcher:
    name: str

    latitude: float     # latitude of launch site
    longitude: float    # longitude of launch site
    altitude: float     # altitude of launch site

    azimuth: float      # azimuth of launch rail
    elevation: float    # elevation of launch rail

    # Unit vectors which give initial position and orientation of the rocket on the pad
    # ECEF coordinates, see Stage.py for more details
    orientation_ECEF: np.ndarray
    position_ECEF: np.ndarray

    initial_velocity: np.ndarray

    def __init__(self, name, latitude, longitude, altitude, azimuth,
                 elevation, initial_velocity=np.array([0, 0, 0])):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.azimuth = azimuth
        self.elevation = elevation

        self.position_ECEF = lla2ecef(
            self.latitude, self.longitude, self.altitude
        )

        self.orientation_ECEF = aer2ecef_unit_vector(
            self.azimuth, self.elevation, self.latitude, self.longitude,
            self.altitude
        )

        self.initial_velocity = initial_velocity
