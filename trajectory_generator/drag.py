# -*- coding: utf-8 -*-
"""
 This drag function returns the drag coefficient for the given Mach number based on a set of conditional statements. 
 The data is derived from a reference source, Sutton's "Rocket Propulsion Elements", 7th edition, page 108. 
"""
from trajectory_generator.stage import Stage
from trajectory_generator.flight_data import Flight_Data
from typing import List, Callable

import csv
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