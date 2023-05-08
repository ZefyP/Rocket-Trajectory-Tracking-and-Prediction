# -*- coding: utf-8 -*-
"""
 This drag function returns the drag coefficient for the given Mach number based on a set of conditional statements. 
 The data is derived from a reference source, Sutton's "Rocket Propulsion Elements", 7th edition, page 108. 
"""
from trajectory_generator.stage import Stage
from trajectory_generator.flight_data import Flight_Data
from trajectory_generator.parachute import Parachute
from trajectory_generator.constants import GRAVITY as G, TIME_STEP
from typing import List, Callable

import csv, math
import numpy as np
import matplotlib.pyplot as plt


def V2_rocket_drag_function(stages: List[Stage], mach: float, flight_data: Flight_Data) -> float:
    # Drag function for V2
    drag_coefficient: float = 0

    if mach > 5:
        drag_coefficient = 0.15
    elif mach > 1.8 and mach <= 5:
        drag_coefficient = -0.03125 * mach + 0.30625
    elif mach > 1.2 and mach <= 1.8:
        drag_coefficient = -0.25 * mach + 0.7
    elif mach > 0.8 and mach <= 1.2:
        drag_coefficient = 0.625 * mach - 0.35
    elif mach <= 0.8:
        drag_coefficient = 0.15
        #drag_coefficient = 0.52 # karman mini 2

    # print(mach, drag_coefficient) # DEBUG
    return drag_coefficient

def fetch_cd_function(stages: List[Stage], mach: float, flight_data: Flight_Data) -> float:
    """
    Returns a callable function that takes a Mach number and returns the drag coefficient (CD) for that Mach number.
    The CD data is fetched from the Flight_Data instance provided.
    """

    # print("Trying to fetch cd function from the Drag module") # DEBUG
    cd_data = flight_data.fetch_cd()
    mach_data = flight_data.fetch_mach()
    # print(mach_data,cd_data) # DEBUG

    def get_cd(mach: float) -> float:
        """
        Returns the CD value for a given Mach number by linearly interpolating between the CD data points.
        """
    if mach < mach_data[0]:
        return cd_data[0]
    if mach > mach_data[-1]:
        return cd_data[-1]
    for i in range(len(mach_data) - 1):
        if mach_data[i] <= mach <= mach_data[i + 1]:
            m1 = mach_data[i]
            m2 = mach_data[i + 1]
            cd1 = cd_data[i]
            cd2 = cd_data[i + 1]
            slope = (cd2 - cd1) / (m2 - m1)

            cd = cd1 + slope * (mach - m1)

            # print(mach, cd)  # DEBUG
            return cd
    
    return get_cd



def addParachute(
            name, CdS, trigger, samplingRate=100, lag=0, noise=(0, 0, 0)
        ):
            """
            Generates a new parachute and records its attributes, 
            such as its opening delay, drag coefficient, and trigger function.

            Args
            ----
            name : string
                Name of the parachute, such as drogue and main. The name has no impact on the simulation and is only used to present data in a more organized manner.
            CdS : float
                The drag coefficient multiplied by the reference area for the parachute. It is used to compute the drag force on the parachute by using the equation F = ((1/2)*rho*V^2)*CdS. The drag force is the dynamic pressure on the parachute multiplied by its CdS coefficient. It is expressed in square meters.
            trigger : function
                A function that determines if the parachute ejection system is to be triggered. It must take the freestream pressure in pascal and the simulation state vector as inputs, which is defined by [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]. It will be called according to the given sampling rate. If the parachute ejection system should be triggered, it should return True, otherwise False.
            samplingRate : float, optional
                The sampling rate at which the trigger function works. It is used to simulate the refresh rate of onboard sensors, such as barometers. The default value is 100, and the value should be given in hertz.
            lag : float, optional
                The time between the parachute ejection system being triggered and the parachute being fully opened. During this time, the simulation assumes the rocket is flying without a parachute. The default value is 0, and it should be given in seconds.
            noise : tuple, list, optional
                A list in the format (mean, standard deviation, time-correlation) is used to add noise to the pressure signal passed to the trigger function. The default value is (0, 0, 0), and the units are in pascal.

            Returns
            -------
            parachute : Parachute
                A Parachute that contains the trigger, samplingRate, lag, CdS, noise, and name. 
                It also stores cleanPressureSignal, noiseSignal, and noisyPressureSignal, which are filled in during the flight simulation.
            """
            # Create a parachute
            parachute = Parachute(name, CdS, trigger, samplingRate, lag, noise)

            # Add parachute to list of parachutes
            parachutes.append(parachute)

            # Return self
            return parachutes[-1]



class Altitude:
    def __init__(self, descent_mode, apogee_alt, ascent_rate, cd):

        self.apogee_altitude = apogee_alt
        self.ascent_rate = ascent_rate

        self.drag_coeff = cd
        self.descent_mode = descent_mode
        
        self.initial_alt = None
        self.full_deployment_time = None
        self.previous_V_z = 0               # vertical speed
        
    def get_density(altitude):
        # source : https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html
        if altitude > 25000: # meters
            """ Upper Stratosphere Model"""
            temp = -131.21 + 0.00299 * altitude # in celcius
            pressure = 2.488 * pow((temp+273.1)/216.6,-11.388) # pressure decreases exponentially

        elif altitude <= 25000 and altitude > 11000:
            """ Lower Stratosphere Model"""
            temp = -56.46
            pressure = 22.65 * math.exp(1.73 - 0.000157 * altitude)

        else:
            """ Troposphere Model"""
            temp = 15.04 - 0.00649 * altitude
            pressure = 101.29 * pow((temp + 273.1)/288.08,5.256)

        return pressure / (0.2869 * (temp + 273.1))


    def predict_descend_altitude(self, time_into_flight, alt):

        if time_into_flight == 0:
            self.initial_alt = alt
            self.full_deployment_time = (self.apogee_altitude - self.initial_alt) / self.ascent_rate

        if self.descent_mode == 0:
            if time_into_flight <= self.full_deployment_time:
                alt = self.initial_alt + time_into_flight * self.ascent_rate
                return True, alt

        V_z = -self.drag_coeff / math.sqrt(self.get_density(alt))
        
        if self.is_at_terminal_velocity(V_z):
            V_z = 0
        else:
            V_z = self.calculate_V_z(V_z)

        alt = alt + TIME_STEP * V_z

        if alt <= 0:
            return False, alt

        return True, alt
    
    def is_at_terminal_velocity(self, V_z):
        """
        If the parachute deploys successfully, we expect terminal velocity within the first 3-5 seconds. 
        """
        return math.isclose(V_z, self.calculate_terminal_velocity(), rel_tol=0.01)

    def calculate_V_z(self, V_z):
        if V_z < self.previous_V_z - G*TIME_STEP:
            V_z = self.previous_V_z - G*TIME_STEP
        elif V_z > self.previous_V_z + G*TIME_STEP:
            V_z = self.previous_V_z + G*TIME_STEP

        self.previous_V_z = V_z
        return V_z

    def calculate_terminal_velocity(self):
        terminal_velocity = -self.drag_coeff / math.sqrt(self.get_density(self.initial_alt))
        return terminal_velocity
    