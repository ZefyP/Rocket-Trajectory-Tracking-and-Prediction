"""
This class  calculates the altitude from a given pressure value, using the ICAO 1993 standard. 
Altitude Calculation is different for when the temperature gradient, beta is zero versus when not. 

The following formulae are used:

When beta != 0:
    H = H_b + T_b / beta * ((P_b / P)^((beta * R) / g_0) - 1)
When beta = 0:
    H = H_b + R * T / g_0 * ln(P_b / P)


Each row is a vector of (H_b, T_b, \beta, P_b)
where H_b is base geopotential height (height from sealevel), T_b is base temperature, beta is temperature gradient and P_b is base pressure 
within that layer.
"""

# include modules

import math
import numpy as np
import matplotlib.pyplot as plt
import csv


# Constants
GAS_CONST = 8.31446261815324        # Universal Gas Constant [J/(K*mol)] 
SPECIFIC_GAS_CONST = 287.05287      # Gas Constant for air
GRAVITY_ACCEL = 9.80665             # Gravitational acceleration [m/s^2]
SEALEVEL_PRESSURE = 101325          # P0, sea level pressure [Pa]
SEALEVEL_TEMP = 288.15              # sea level temperature [K]
AVAGADRO_CONST = 6.02214076e23      # Avogadro's Constant
SEALEVEL_MOLARMASS =  28.964420e-03 # (NOT PRIMARY, DERIVED) Molar mass of air at sea level [kg/mol]
SPECIFIC_GAS_CONST = 287.05287      # specific gas constant (Assumed constant)
RADIUS_OF_EARTH = 6356766           # constant

class Atmos:
    def __init__(self, pressure):
        self.geometric_alt = None
        self.geopotential_alt = None
        self.layer_const = None
        self.layers = np.array([
            # Matrix of layer constants
            # Each Layer is a vector = [Geopotential [m], Temperature [K], Temp Gradient[K/m]]
        
            [0, 288.15, -0.0065, 101325],       # Troposphere
            [11000, 216.65, 0, 22632.06],       # Tropopause
            [20000, 216.65, 0.001, 5474.889],   # Stratosphere
            [32000, 228.65, 0.0028, 868.0187],  # Stratosphere
            [47000, 270.65, 0, 110.9063],       # Stratopause
            [51000, 270.65, -0.0028, 66.93887], # Mesosphere
            [71000, 214.65, -0.002, 3.956420],  # Mesosphere
            [84852, 186.946, 0, 0.3733824],     # Mesopause
            [84852, 186.946, 0, 0]])            # Thermosphere

        self.pressure = pressure

    def get_altitude(self):
        """
        Get the altitude at this point in the Atmosphere
        :return: The altitude [m] above Mean Sea Level
        """
        if self.geometric_alt is None:
            self.geometric_alt = self.altitude(self.pressure)
        return self.geometric_alt

    def get_layer_constants(self, p):
        """
        Get the layer constants for the current pressure
        :param p: The current pressure [Pa]
        :return: A vector of (H_b, T_b, beta, P_b) which represents the layer constants
        """
        i = np.searchsorted(self.layers[:, 3], p, side='right') - 1
        return self.layers[i]

    def to_geopotential(self, alt):
        return alt * RADIUS_OF_EARTH / (RADIUS_OF_EARTH + alt)

    def altitude(self, pressure):
        if pressure <= 0:
            raise ValueError("Input pressure must be positive.")
        
        layer = self.get_layer_constants(pressure)
        Hb, Tb, beta, Pb = layer
        if beta != 0:
            return Hb + Tb / beta * ((Pb / pressure) ** (beta * GAS_CONST / GRAVITY_ACCEL) - 1)
        else:
            log = math.log(Pb / pressure)
            return Hb + SPECIFIC_GAS_CONST * Tb / GRAVITY_ACCEL * log


# geometric_alt = None
# geopotential_alt = None
# layer_const = None
# layers = np.array([
#     # Matrix of layer constants
#     # Each Layer is a vector = [Geopotential [m], Temperature [K], Temp Gradient[K/m]]

#     [0, 288.15, -0.0065, 101325],       # Troposphere
#     [11000, 216.65, 0, 22632.06],       # Tropopause
#     [20000, 216.65, 0.001, 5474.889],   # Stratosphere
#     [32000, 228.65, 0.0028, 868.0187],  # Stratosphere
#     [47000, 270.65, 0, 110.9063],       # Stratopause
#     [51000, 270.65, -0.0028, 66.93887], # Mesosphere
#     [71000, 214.65, -0.002, 3.956420],  # Mesosphere
#     [84852, 186.946, 0, 0.3733824],     # Mesopause
#     [84852, 186.946, 0, 0]])            # Thermosphere


# def get_altitude():
#     """
#     Get the altitude at this point in the Atmosphere
#     :return: The altitude [m] above Mean Sea Level
#     """
#     if geometric_alt is None:
#         geometric_alt = altitude(pressure)
#     return geometric_alt

# def get_layer_constants(p):
#     """
#     Get the layer constants for the current pressure
#     :param p: The current pressure [Pa]
#     :return: A vector of (H_b, T_b, beta, P_b) which represents the layer constants
#     """
#     i = np.searchsorted(layers[:, 3], p, side='right') - 1
#     return layers[i]

# def to_geopotential(self, alt):
#     return alt * RADIUS_OF_EARTH / (RADIUS_OF_EARTH + alt)

# def altitude(pressure):
#     layer = get_layer_constants(pressure)
#     Hb, Tb, beta, Pb = layer
#     if beta != 0:
#         return Hb + Tb / beta * ((Pb / pressure) ** (beta * GAS_CONST / GRAVITY_ACCEL) - 1)
#     else:
#         return Hb + SPECIFIC_GAS_CONST * Tb / GRAVITY_ACCEL * math.log(Pb / pressure)


