# -*- coding: utf-8 -*-

import numpy as np
from .constants import TRAJECTORY_TIMEOUT, TIME_STEP, N_TIME_INTERVALS, EARTH_RADIUS, GRAVITY
from .ambiance import Atmosphere
from .utils import ecef2lla # earth centered coordinates to latitude-longitude-altitude
import simplekml


class Stage:
    """
    Represents one stage of a rocket or rocket
    
    All units are SI unless otherwise specified
    """
    name: str = None

    # mass of the stage without any fuel
    dry_mass: float

    # mass of the fuel in the tanks before launch
    fuel_mass: float

    total_mass: float

    # frontal area of the rocket
    # metres squared
    cross_sectional_area: float

    # max thrust produced by the stage in Newtons
    thrust: float

    # time spent firing the motor 
    burn_time: float

    # amount of seconds after motor burnout before stage separation occurs
    # can be None if this is the uppermost stage
    separation_time: float = None

    # used to determine the amount of fuel used
    # if not given, all fuel is assumed to be burnt linearly with burn_time
    specific_impulse: float

    # time vector
    time: np.ndarray

    # position, velocity and acceleration matrices are defined as follows
    # 3 (x, y, z) rows and n columns
    # N_TIME_INTERVALS = number of time intervals
    # N_TIME_INTERVALS = TRAJECTORY_TIMEOUT/TIME_STEP
    # these are ECEF (Earth-centered, earth-fixed) coordinates
    # 
    # +-----------+---+-----+------------------+
    # | Dimension | 0 | ... | N_TIME_INTERVALS |
    # +-----------+---+-----+------------------+
    # | x         |   |     |                  |
    # | y         |   |     |                  |
    # | z         |   |     |                  |
    # +-----------+---+-----+------------------+    
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    gravity_vectors: np.ndarray
    thrust_vectors: np.ndarray
    drag_vectors: np.ndarray
    fuel_mass_vector: np.ndarray
    lla_vector: np.ndarray
    surface_position: np.ndarray
    _range: np.ndarray

    has_separated: bool = False

    burn_start_time = None

    has_impacted_ground: bool = False

    kml_colour: str  # hex code for KML styling

    def __init__(self, name, dry_mass, fuel_mass, thrust, burn_time,
                 specific_impulse=None, separation_time=0,
                 diameter=None, cross_sectional_area=None,
                 kml_colour="ffffffff"):
        self.name = name
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.thrust = thrust
        self.burn_time = burn_time
        self.separation_time = separation_time
        self.kml_colour = kml_colour

        self.total_mass = dry_mass + fuel_mass

        # define cross sectional area (m^2) for drag calculations
        if not diameter and not cross_sectional_area:
            print(f"WARNING: diameter or cross sectional area not " +
                  "specified for {self.name}, drag shall equal zero")
            cross_sectional_area = 0
        elif diameter:
            self.cross_sectional_area = (np.pi * diameter ** 2) / 4
        elif cross_sectional_area:
            self.cross_sectional_area = cross_sectional_area

        # if specific impulse is not given, calculate optimal
        if specific_impulse is None and fuel_mass > 0:
            self.specific_impulse = self.thrust / ((self.fuel_mass / self.burn_time) * GRAVITY)
            # print(f"[{self.name}] calculated ISP {self.specific_impulse}")
        elif fuel_mass == 0:
            self.specific_impulse = 1
        else:
            self.specific_impulse = specific_impulse

        # np arrays must be created within constructor to prevent memory
        # sharing across all instances meaning stages have identical trajectories
        self.time = np.arange(0, TRAJECTORY_TIMEOUT, TIME_STEP)
        self.position = np.zeros((3, N_TIME_INTERVALS))
        self.velocity = np.zeros((3, N_TIME_INTERVALS))
        self.acceleration = np.zeros((3, N_TIME_INTERVALS))
        self.gravity_vectors = np.zeros((3, N_TIME_INTERVALS))
        self.thrust_vectors = np.zeros((3, N_TIME_INTERVALS))
        self.drag_vectors = np.zeros((3, N_TIME_INTERVALS))
        self.fuel_mass_vector = np.zeros((N_TIME_INTERVALS,))
        self.lla_vector = np.zeros((3, N_TIME_INTERVALS))
        self.surface_position = np.zeros((3, N_TIME_INTERVALS))
        self._range = np.zeros((N_TIME_INTERVALS,))
        self.fuel_mass_vector[0] = self.fuel_mass

    def get_lla(self, position_index):
        position = self.position[:, position_index]
        x, y, z = position[0], position[1], position[2]

        lat, long, alt = ecef2lla(x, y, z)
        lat, long, alt = float(lat), float(long), float(alt)
        return lat, long, alt

    def get_mach_number(self, velocity_index, atmosphere: Atmosphere):
        if atmosphere is None:
            return None
        current_velocity = self.velocity[:, velocity_index]

        velocity_magnitude = np.sqrt(
            current_velocity.dot(current_velocity)
        )
        # mach number is the ratio of velocity magn to the speed of sound in the atmosphere
        mach_number = velocity_magnitude / atmosphere.speed_of_sound
        return mach_number

    def get_gravity_vector(self, position_index, mass):
        # Calculate the gravitational force vector acting on the rocket at a specific position index and mass
        current_position = self.position[:, position_index]
        position_magnitude = np.sqrt(
            current_position.dot(current_position)
        )
        # Calculate the unit vector in the direction of gravity
        gravity_direction = - current_position / position_magnitude
        # Get its magnitude using F = G * m1 * m2 / r^2
        gravity_magnitude = ((6.67e-11 * 5.972e24 * mass) / position_magnitude ** 2)
        return gravity_magnitude * gravity_direction

    def get_lla_position_vector(self):
        lat, long, alt = ecef2lla(self.position[0, :], self.position[1, :], self.position[2, :])

        output = np.zeros(self.position.shape)
        output[0, :] = long.reshape(1, lat.shape[0])
        output[1, :] = lat.reshape(1, lat.shape[0])
        output[2, :] = alt.reshape(1, lat.shape[0])

        return output

    def export_KML(self, filename: str, downsample_factor=1, extrude=True, description=True):
        print(f"[{self.name}] Exporting KML to '{filename}'")
        kml = simplekml.Kml()
        ls = kml.newlinestring(name=self.name)

        pos = self.get_lla_position_vector()

        coords = [tuple(pos[:, x]) for x in np.arange(0, pos.shape[1], downsample_factor)]
        ls.coords = coords
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        ls.style.linestyle.color = self.kml_colour
        ls.style.linestyle.width = 3
        ls.extrude = 1 if extrude else 0
        ls.polystyle.color = "64" + self.kml_colour[2::]
        ls.polystyle.outline = 0
        if description:
            ls.description = str(self)

        kml.save(filename)
        print(f"[{self.name}] KML export finished")

    def get_impact_location_ECEF(self):
        if not self.has_impacted_ground:
            return None

        return self.position[:, -1]

    def export_CSV(self, filename):
        output_matrix = np.zeros((self.time.shape[0], 7))
        output_matrix[:, 0] = self.time
        output_matrix[:, 1:4] = self.position.T
        output_matrix[:, 4:7] = self.velocity.T

        np.savetxt(
            filename,
            output_matrix,
            delimiter=",",
            fmt="%f",
            header="TIME,X,Y,Z,vX,vY,vZ",
            comments=""
        )

    def __str__(self):
        precision = 6  # float precision
        column_width = 10
        alignment = ">"
        
        extra_info = ""

        format_string = f"{alignment}{column_width}.{precision}"

        if self.has_impacted_ground:
            altitude = self.lla_vector[2, :]
            apogee = np.amax(altitude)

            extra_info = \
                f"Flight Time:          {float(self.time[-1]):{format_string}} s\n" + \
                f"Apogee:               {float(apogee / 1000):{format_string}} km\n" + \
                f"Range:                {float(self._range[-1] / 1000):{format_string}} km\n"
            
        return f"{self.name:^36}\n{'=' * len(self.name):^36}\n" + \
               f"Dry Mass:             {float(self.dry_mass):{format_string}} kg\n" + \
               f"Fuel Mass:            {float(self.fuel_mass):{format_string}} kg\n" + \
               f"Total Mass:           {float(self.total_mass):{format_string}} kg\n" + \
               f"Cross-Sectional Area: {float(self.cross_sectional_area):{format_string}} m^2\n" + \
               f"Thrust:               {float(self.thrust):{format_string}} Newtons\n" + \
               f"Burn Time:            {float(self.burn_time):{format_string}} s\n" + \
               f"Separation Time:      {float(self.separation_time):{format_string}} s\n" + \
               extra_info
