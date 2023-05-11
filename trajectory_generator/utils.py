# -*- coding: utf-8 -*-
"""
A collection of helper functions that can be used across multiple modules in this project. 
It includes common mathematical operations and utilities for data processing. The functions have been sourced from Mathworks and designed 
to be modular and reusable, with clear input and output specs. 
The script is intended to improve code reusability, reduce code duplication and improve code readability.
"""
import numpy as np
from .constants import EARTH_RADIUS


def ecef2aer(a, b):
    # https://uk.mathworks.com/help/map/ref/ecef2aer.html
    return enu2aer(ecef2enu(a, b))


def ecef2enu(a, b):
    # https://uk.mathworks.com/help/map/ref/ecef2enu.html
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
    lat, long, alt = ecef2lla(a[0], a[1], a[2])
    lat, long, alt = float(lat), float(long), float(alt)

    slat = np.sin(np.radians(lat))
    slon = np.sin(np.radians(long))
    clat = np.cos(np.radians(lat))
    clon = np.cos(np.radians(long))

    enu = np.matmul(np.array([
        [- slat, clon, 0],
        [- slat * clon, - slat * clon, clat],
        [clat * clon, clat * slon, slat]
    ]), b - a)
    return enu


def enu2aer(enu):
    """
    https://uk.mathworks.com/help/map/ref/enu2aer.html
    """
    _range = np.linalg.norm(enu)
    azimuth = np.arctan2(enu[0], enu[1])
    elevation = np.arcsin(enu[2] / _range)

    return np.degrees(azimuth), np.degrees(elevation), _range


def aer2enu(azimuth, elevation, _range):
    """
    Convert azimuth, elevation and range to local East, North Up coordinates
    https://uk.mathworks.com/help/map/ref/aer2enu.html
    """
    cos_azimuth = np.cos(np.radians(azimuth))
    cos_elevation = np.cos(np.radians(elevation))
    sin_azimuth = np.sin(np.radians(azimuth))
    sin_elevation = np.sin(np.radians(elevation))

    enu = np.array([
        _range * cos_elevation * sin_azimuth,
        _range * cos_elevation * cos_azimuth,
        _range * sin_elevation
    ])

    return enu


def enu2ecef(enu, obs_lat, obs_long, obs_alt):
    """
    https://uk.mathworks.com/help/map/ref/enu2ecef.html

    Convert local East, North, Up coordinates to ECEF coordinates given 
    a latitude, longitude and altitude
    
    Note: Must be geodetic coordinates and not geocentric, else the 
    elliptical nature of the earth is not accounted for
    """
    slat = np.sin(np.radians(obs_lat))
    slon = np.sin(np.radians(obs_long))
    clat = np.cos(np.radians(obs_lat))
    clon = np.cos(np.radians(obs_long))

    obs_ecef = lla2ecef(obs_lat, obs_long, obs_alt)

    ecef = np.matmul(np.array([
        [- slon, -slat * clon, clat * clon],
        [clon, -slat * slon, clat * slon],
        [0, clat, slat]
    ]), enu) + obs_ecef

    return ecef


def aer2ecef(azimuth, elevation, _range, obs_lat, obs_long, obs_alt):
    # https://uk.mathworks.com/help/map/ref/aer2ecef.html
    enu = aer2enu(azimuth, elevation, _range)
    ecef = enu2ecef(enu, obs_lat, obs_long, obs_alt)
    return ecef


def aer2ecef_unit_vector(azimuth, elevation, obs_lat, obs_long, obs_alt):
    """
    https://uk.mathworks.com/help/map/ref/aer2ecef.html

    Given a current azimuth, elevation and range, as well as a latitude, 
    longitude and altitude, calculate an ECEF unit vector pointing in the 
    direction of a target.
    
    Used to calculate launch directions 
    """
    # Convert azimuth, elevation, and range to ENU coordinates
    enu = aer2enu(azimuth, elevation, 1)
    # Convert observer's latitude, longitude, and altitude to ECEF coordinates
    ecef_obs = lla2ecef(obs_lat, obs_long, obs_alt)
    # Convert ENU to ECEF coordinates
    ecef = enu2ecef(enu, obs_lat, obs_long, obs_alt)
    # Convert ENU to ECEF coordinates by subtracting the observer's ECEF pos from target's EXEF pos
    return ecef - ecef_obs


# source: https://github.com/kvenkman/ecef2lla
# other source (more credible) https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/7942/versions/1/previews/lla2ecef.m/index.html
def lla2ecef(lat, lon, alt):
    """
    Convert latitude, longitude and altitude to Earth-Centered Earth-Fixed (ECEF) coordinates.

    Uses the WGS84 reference ellipsoid model.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        alt (float): Altitude in meters.

    Returns:
        np.ndarray: 3D ECEF coordinates in meters.
    """
    # Set WGS84 constants
    a = 6378137                 # Semi-major axis
    a_sq = a ** 2              
    e = 8.181919084261345e-2    # Eccentricity
    e_sq = e ** 2               
    b_sq = a_sq * (1 - e_sq)    # Semi-minor axis squared
    
    # Convert lat, lon to radians
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    alt = alt

    # Calculate N (the radius of curvature in the prime vertical)
    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    # Calculate x, y, z ECEF coordinates
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((b_sq / a_sq) * N + alt) * np.sin(lat)
    # Combine into a single 3D array
    result = np.array([x, y, z])

    return result

# Convert ECEF coordinates to LLA coordinates.
# Note that this conversion is utilised much more than the inverse.
def ecef2lla(x, y, z):
    # Convert input scalars or vectors to column vectors. [meters]
    x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
    y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
    z = np.array([z]).reshape(np.array([z]).shape[-1], 1)
    
    # Set WGS84 constants.
    a = 6378137             # Semi-major axis
    e_sq = 6.69437999014e-3 # Eccentricity squared
    f = 1 / 298.257223563   # Flattening factor
    b = a * (1 - f)         # Semi-minor axis

    # Calculations:
    r = np.sqrt(x ** 2 + y ** 2)        # Distance from z-axis.
    ep_sq = (a ** 2 - b ** 2) / b ** 2  # Eccentricity prime squared.
    ee = (a ** 2 - b ** 2)              # Second eccentricity squared.
    f = (54 * b ** 2) * (z ** 2)
    g = r ** 2 + (1 - e_sq) * (z ** 2) - e_sq * ee * 2
    c = (e_sq ** 2) * f * r ** 2 / (g ** 3)
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3.)
    p = f / (3. * (g ** 2) * (s + (1. / s) + 1) ** 2)
    q = np.sqrt(1 + 2 * p * e_sq ** 2)
    
    # Handle invalid (negative?) input to sqrt input over poles.
    sqrt_input = 0.5 * (a ** 2) * (1 + (1. / q)) - p * (z ** 2) * (1 - e_sq) / (q * (1 + q)) - 0.5 * p * (r ** 2)
    if np.any(sqrt_input < 0):
        # Print warning message for negative sqrt input over poles.
        print(sqrt_input)
        print("Negative square root input over poles detected.")
    sqrt_input[sqrt_input < 0] = 0

    # Compute additional intermediate variables.
    r_0 = -(p * e_sq * r) / (1 + q) + np.sqrt(sqrt_input)
    u = np.sqrt((r - e_sq * r_0) ** 2 + z ** 2)
    v = np.sqrt((r - e_sq * r_0) ** 2 + (1 - e_sq) * z ** 2)
    z_0 = (b ** 2) * z / (a * v)
    h = u * (1 - b ** 2 / (a * v))
    phi = np.arctan((z + ep_sq * z_0) / r) # Phi is the geodetic latitude.
    lambd = np.arctan2(y, x)               # Lambda is the # Geodetic latitude.
    
    # Convert from radians to degrees and return.
    return phi * 180 / np.pi, lambd * 180 / np.pi, h


def fast_has_impacted_earth(ecef_coords: np.ndarray) -> bool:
    # calculate the distance from the ECEF coordinates to the center of the Earth
    distance_from_centre = np.sqrt(ecef_coords.dot(ecef_coords))

    # check if the distance is less than the Earth's radius, indicating impact with the Earth
    if distance_from_centre < EARTH_RADIUS:
        return True
    else:
        return False

# Calculate the norm of the input vector using np.linalg.norm().
# This means scaling the magnitutde to length of 1 while preserving its direction.
# It also simplifies calculations such as computing the dot product and determining the angle between two vectors.
def normalise_vector(input_vector: np.ndarray):
    return input_vector / np.linalg.norm(input_vector)




# import math

# def calc_terminal_velocity(mass, area, rho, cd):
#     g = 9.81
#     V = math.sqrt((2 * mass * g) / (rho * area * cd))
#     return V

# def calc_lateral_drift(mass, area, rho, cd, V):
#     g = 9.81
#     L = (2 * mass * g) / (rho * area * cd * V**2)
#     return L

# # constants
# rho = 1.225  # air density at sea level, kg/m^3
# cd = 1.75    # assumed constant drag coefficient
# diameter = 1.2  # m
# area = math.pi * (diameter/2)**2  # m^2

# # calculate terminal velocity and lateral drift for two masses
# mass1 = 50  # kg
# mass2 = 60  # kg

# V1 = calc_terminal_velocity(mass1, area, rho, cd)
# V2 = calc_terminal_velocity(mass2, area, rho, cd)

# delta_V = abs(V1 - V2) / ((V1 + V2) / 2) * 100  # percentage difference in terminal velocity
# print(f"Percentage difference in terminal velocity: {delta_V:.2f}%")

# L1 = calc_lateral_drift(mass1, area, rho, cd, V1)
# L2 = calc_lateral_drift(mass2, area, rho, cd, V2)

# delta_L = abs(L1 - L2) / ((L1 + L2) / 2) * 100  # percentage difference in lateral drift
# print(f"Percentage difference in lateral drift: {delta_L:.2f}%")

