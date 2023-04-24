# -*- coding: utf-8 -*-
import numpy as np
from .constants import EARTH_RADIUS


def enu2aer(enu):
    _range = np.linalg.norm(enu)
    azimuth = np.arctan2(enu[0], enu[1])
    elevation = np.arcsin(enu[2] / _range)

    return np.degrees(azimuth), np.degrees(elevation), _range


def aer2enu(azimuth, elevation, _range):
    """
    Convert azimuth, elevation and range to local East, North Up coordinates
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
    enu = aer2enu(azimuth, elevation, _range)
    ecef = enu2ecef(enu, obs_lat, obs_long, obs_alt)
    return ecef


def aer2ecef_unit_vector(azimuth, elevation, obs_lat, obs_long, obs_alt):
    """
    Given a current azimuth, elevation and range, as well as a latitude, 
    longitude and altitude, calculate an ECEF unit vector pointing in the 
    direction of a target
    
    Used to calculate launch directions 
    """

    enu = aer2enu(azimuth, elevation, 1)
    ecef_obs = lla2ecef(obs_lat, obs_long, obs_alt)
    ecef = enu2ecef(enu, obs_lat, obs_long, obs_alt)

    return ecef - ecef_obs


# source: https://github.com/kvenkman/ecef2lla
# other source (more credible) https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/7942/versions/1/previews/lla2ecef.m/index.html
def lla2ecef(lat, lon, alt):
    a = 6378137
    a_sq = a ** 2
    e = 8.181919084261345e-2
    e_sq = e ** 2
    b_sq = a_sq * (1 - e_sq)

    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    alt = alt

    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((b_sq / a_sq) * N + alt) * np.sin(lat)

    result = np.array([x, y, z])

    return result


# it should be noted that converting ECEF to LLA is way more involved than
#    the other way around
def ecef2lla(x, y, z):
    # x, y and z are scalars or vectors in meters
    x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
    y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
    z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

    a = 6378137
    e_sq = 6.69437999014e-3

    f = 1 / 298.257223563
    b = a * (1 - f)

    # calculations:
    r = np.sqrt(x ** 2 + y ** 2)
    ep_sq = (a ** 2 - b ** 2) / b ** 2
    ee = (a ** 2 - b ** 2)
    f = (54 * b ** 2) * (z ** 2)
    g = r ** 2 + (1 - e_sq) * (z ** 2) - e_sq * ee * 2
    c = (e_sq ** 2) * f * r ** 2 / (g ** 3)
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3.)
    p = f / (3. * (g ** 2) * (s + (1. / s) + 1) ** 2)
    q = np.sqrt(1 + 2 * p * e_sq ** 2)
    # RuntimeWarning: invalid value encountered in sqrt
    # handle invalid (negative?) input to sqrt input over poles
    sqrt_input = 0.5 * (a ** 2) * (1 + (1. / q)) - p * (z ** 2) * (1 - e_sq) / (q * (1 + q)) - 0.5 * p * (r ** 2)
    if np.any(sqrt_input < 0):
        print(sqrt_input)
    sqrt_input[sqrt_input < 0] = 0

    r_0 = -(p * e_sq * r) / (1 + q) + np.sqrt(sqrt_input)
    u = np.sqrt((r - e_sq * r_0) ** 2 + z ** 2)
    v = np.sqrt((r - e_sq * r_0) ** 2 + (1 - e_sq) * z ** 2)
    z_0 = (b ** 2) * z / (a * v)
    h = u * (1 - b ** 2 / (a * v))
    phi = np.arctan((z + ep_sq * z_0) / r)
    lambd = np.arctan2(y, x)

    return phi * 180 / np.pi, lambd * 180 / np.pi, h


def fast_has_impacted_earth(ecef_coords: np.ndarray) -> bool:
    distance_from_centre = np.sqrt(ecef_coords.dot(ecef_coords))

    if distance_from_centre < EARTH_RADIUS:
        return True
    else:
        return False


def normalise_vector(input_vector: np.ndarray):
    return input_vector / np.linalg.norm(input_vector)
