# -*- coding: utf-8 -*-

import numpy as np
import math
from trajectory_generator.constants import GRAVITY as g
#from trajectory_generator.constants import R, GAMMA
R = 287.058 # specific gas constant under the ideal gas low: J/(kg K)
GAMMA = 1.4
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from trajectory_generator.atmosphere import Atmosphere



class Parachute:
    """
    This class stores the information of a parachute.

    Attributes:
    """
    name : str          # name of the parachute
    
    diameter : float    # m 

    shape : str         # select parachute shape: square,hexagon,octagon,circle
    
    trigger : bool      # defines if the parachute is to be triggered

    lag : float         # time between the parachute ejection system and full deployment/parachute fully open

    #pressureSignal:list # (t, pressure) list that is passed to the trigger function

    def __init__(
            
            self,
            name,
            diameter,
            shape ='square',
            trigger = True,
            lag = 0.1
    ):
        
        self.name = name
        self.diameter = diameter
        self.shape = shape
        self.trigger = trigger
        self.lag = lag
        # self.altitudeSignal = altitudeSignal = []


def get_chute_surface_area(diameter,shape = 'circle'):
    # source: https://www.apogeerockets.com/education/downloads/Newsletter449.pdf

    if shape == 'square':
        Surface = diameter ** 2
    elif shape == 'hexagon':
        Surface = 0.866 * diameter ** 2
    elif shape == 'octagon':
        Surface = 0.828 * diameter ** 2
    elif shape == 'circle':
        Surface = 0.2 * math.pi * diameter ** 2
    else:
        raise ValueError("Unsupported parachute shape. Choose between square, hexagon, octagon and circle. The input is case sensitive.")

    return Surface

def basic_parachute_cd(): # using this makes the calculation no better than open rocket. TODO: development
    return 0.8


def calc_density(altitude):
 # Calculate temperature and density based on altitude
    if altitude < 11000:
        # Below 11 km
        T = 288.15 - 0.0065 * altitude
        p = 101325 * (T / 288.15)**(-g / (R * 0.0065))
    elif altitude < 20000:
        # Between 11 km and 20 km
        T = 216.65
        p = 22632.06 * math.exp(-g * (altitude - 11000) / (R * T))
    elif altitude < 32000:
        # Between 20 km and 32 km
        T = 216.65 + 0.001 * (altitude - 20000)
        p = 5474.89 * (T / 216.65)**(-g / (R * 0.001))
    elif altitude < 47000:
        # Between 32 km and 47 km
        T = 228.65 + 0.0028 * (altitude - 32000)
        p = 868.02 * (T / 228.65)**(-g / (R * 0.0028))
    elif altitude < 51000:
        # Between 47 km and 51 km
        T = 270.65
        p = 110.91 * math.exp(-g * (altitude - 47000) / (R * T))
    else:
        # Above 51 km
        T = 270.65 + 0.0028 * (altitude - 51000)
        p = 66.94 * (T / 270.65)**(-g / (R * 0.0028))

    rho = p / (R * T)
    return   rho , T

def calculate_reynolds_number(diameter, density, mach):
    """
    Calculates the Reynolds number for a given diameter, altitude, and Mach number.

    Args:
        diameter (float): The diameter of the object in meters.
        altitude (float): The altitude in meters above sea level.
        mach (float): The Mach number of the object.

    Returns:
        float: The calculated Reynolds number.

    Constants:
        GAMMA (float): The ratio of specific heats for air at a constant pressure and volume.
        R (float): The specific gas constant for air.
        C (float): The Sutherland's law constant.
        S (float): The Sutherland's law constant.
        T_s (float): The reference temperature for Sutherland's law.
        mu_0 (float): The dynamic viscosity of air at the reference temperature.

    Variables:
        rho (float): The air density based on altitude.
        a (float): The speed of sound at the given altitude.
        v (float): The velocity based on Mach number and speed of sound.
        T (float): The temperature based on altitude.
        p (float): The pressure based on altitude.
        mu (float): The dynamic viscosity of air based on Sutherland's law.
        Re (float): The Reynolds number calculated based on the input values.

    Formula:
        Re = (rho * v * diameter) / mu

    """

    # calculate based on the Atmospheric Model
    rho, T = calc_density(altitude)
    # Calculate speed of sound and velocity based on Mach number
    a = math.sqrt(GAMMA * R * T)
    v = mach * a
    # Calculate dynamic viscosity based on Sutherland's law
    C = 120         # constant for air # TODO : source
    S = 110.4       # Sutherland's constant for air, which is 110.4 K
    T_s = 110.4     # reference temperature at which the dynamic viscosity is known, which is also 110.4 K for air
    mu_0 = 1.716e-5 # dynamic viscosity of air at T_s
    mu = mu_0 * ((T_s + C) / (T + C)) * ((T / T_s)**1.5)
    
    # Calculate Reynolds number
    Re = rho * v * diameter / mu
    #print(altitude, "..." , Re, "................................", rho  ,  mu , "...................................", T , p, "..................................")
    
    return Re


def parachute_descent_speed(area, mass, rho_air, cd): # TODO: must add account for different rho at different altitude
    """
    Calculates the steady descent speed of a parachute given its area, the mass it is supporting,
    the density of the air it is descending through, and its drag coefficient.

    Args:
    area (float): the area of the parachute (m^2)
    mass (float): the mass being supported by the parachute (kg)
    rho_air (float): the density of the air the parachute is descending through (kg/m^3)
    cd (float): the drag coefficient of the parachute

    Returns:
    float: the steady descent speed of the parachute (m/s)
    """
    # terminal_velocity = -self.drag_coeff / math.sqrt(self.get_density(self.initial_alt)) # not used as we have more factors affecting this
    steady_V_z =  math.sqrt(2 * mass * g / (rho_air * area * cd))

    return steady_V_z

##################################
# test script
##################################
# diameter = 0.30 # 0.3 meters ~ 12 inch
# area = get_chute_surface_area(diameter,'circle')

# altitudes = [100,200,500, 1000,2000,5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000] # meters
# mach_numbers = [0.1, 0.2, 0.3, 0.4, 0.5]
# chute_cd = 1.55 # from datasheet of the chute

# # Plot Reynolds number for different altitude and Mach number combinations
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = ['r', 'b', 'g', 'c', 'm']
# markers = ['o', 's', 'D', '^', 'v']
# for j in range(len(mach_numbers)):
#     Re = []
#     V_desc = []
#     density = [] 
#     mass = 0.577  # TODO : dry_mass should be pulled from Stage class object
      
#     for altitude in altitudes:
        
#         Re.append(calculate_reynolds_number(diameter, altitude, mach_numbers[j]))
#         density,_ = calc_density(altitude) # it is a tuple
#         #V_desc = [parachute_descent_speed(area, mass, density, chute_cd) for altitude in altitudes]

#     f = interpolate.interp1d(altitudes, Re, kind='linear')
#     f_Vdesc = interpolate.interp1d(altitudes, V_desc, kind='linear')
#     altitudes_new = np.linspace(min(altitudes), max(altitudes), num=100)
#     ax.plot(altitudes_new, f(altitudes_new), label=f'Mach {mach_numbers[j]}', color=colors[j], marker=markers[j],markersize=6, markevery=2)
#     #plt.plot(altitudes_new, f_Vdesc(altitudes_new), label=f'Mach {mach_numbers[j]} - V_desc', color=colors[j], linestyle='--', marker=markers[j],markersize=6, markevery=2)

#     print(f(altitudes_new))
    
# ax.set_xlabel('Altitude (m)', fontsize=12)
# ax.set_ylabel('Reynolds number', fontsize=12)
# ax.set_title('Reynolds number vs altitude and Mach number', fontsize=14)
# ax.legend(fontsize=10)
# plt.show()




def parachute_cd(re, diameter):
    """
    Args:
    re (float): the Reynolds number
    diameter (float): the diameter of the parachute (m)
    """
    if re <= 100000:
        cd = 0.8
    elif re > 100000 and re <= 200000:
        cd = 0.75
    else:
        cd = 0.7
    return cd

def alt_parachute_cd(re, diameter):
    """
    Alternative way that calculates the drag coefficient of a parachute for a given Reynolds number and diameter.

    Args:
        re (float): Reynolds number
        diameter (float): diameter of the parachute in meters

    Returns:
        float: drag coefficient
    """
    # Define the power law function
    def power_law(re, a, b):
        return a * re ** b

    # Define the Reynolds number range for fitting the power law function
    re_range = np.array([1000, 10000, 100000, 200000, 300000, 400000, 500000])

    # Define the corresponding drag coefficient values
    cd_range = np.array([0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7])

    # Fit the power law function to the data
    popt, _ = curve_fit(power_law, re_range, cd_range)

    # Calculate the drag coefficient using the fitted power law function
    cd = power_law(re, *popt)

    return cd
    # return cd * (diameter / 0.2) ** 2

def plot_re_cd():
    diameter = 0.12
    area = diameter**2
    re = calculate_reynolds_number(diameter, altitude, mach)
    # cd = alt_parachute_cd(re, diameter)
    # cd = parachute_cd(re, diameter)
    # V = parachute_descent_speed(area, 0.577, 0.97, cd) # TODO: input dry mass from Stage class TODO: import rho from ambience
    

    # create a range of Reynolds numbers to plot
    re_range = np.linspace(0, 500000, 1000)

    # calculate corresponding drag coefficients using parachute_cd function
    cd_range = [alt_parachute_cd(re, diameter) for re in re_range]

    # create a plot
    plt.plot(re_range, cd_range)
    plt.xlabel('Reynolds Number')
    plt.ylabel('Drag Coefficient')
    plt.show()

plot_re_cd()







# for altitude in altitudes:
#     for mach in mach_numbers:
#         Re = calculate_reynolds_number(diameter, altitude, mach)
#         print(f"At {altitude} meters altitude and Mach {mach}, Reynolds number is {Re:.2f}")
