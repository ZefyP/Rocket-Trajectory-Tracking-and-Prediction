"""
this script will output the cd for a given Mach number 
"""
#from .stage import Stage
from typing import List
import numpy as np

import matplotlib.pyplot as plt #temp for dev   
plt.style.use("ggplot") # include the style sheet to make graphs pretty


# Define points to use for interpolation
points = [(0.2, 0.15), (0.8, 0.17), (1.2, 0.42), (1.8, 0.25), (5.0, 0.15)] # Drag coefficient approximation derived from Sutton, "Rocket Propulsion Elements", 7th ed, p108

  
# Function to perform linear interpolation of the drag coefficient based on Mach number (x)
def interpolate_cd(points, x):
    for i in range(len(points)-1):
        if points[i][0] <= x <= points[i+1][0]:
            # Calculate the slope and y-intercept for the two closest points
            slope = (points[i+1][1] - points[i][1]) / (points[i+1][0] - points[i][0])
            # Use the slope and y-intercept to calculate the interpolated drag coefficient
            y_intercept = points[i][1] - slope * points[i][0]
            cd = slope * x + y_intercept
            return cd
    

def plot_mach_vs_cd():

    # Generate points for plotting
    x_range = [i/10 for i in range(0, 51)] # using 50+1 to represent the granularity of mach numbers from 0.0 to 5.0 inclusive.  (5 - 0) / 0.1 + 1 = 51
    y_values = [interpolate_cd(points, x) for x in x_range]

    # Plot the graph
    # fig,ax1 = plt.subplots()
    plt.plot(x_range, y_values)
    
    # try to smooth out the graph
    # poly = np.polyfit(x_range,y_values,5)
    # poly_y = np.poly1d(poly)(x_range)
    # plt.plot(x_range,poly_y)

    # set labels and display the plot
    plt.xlabel('Mach Number')
    plt.ylabel('Drag Coefficient')
    plt.title('Mach Number vs Drag Coefficient')
    plt.show()

# call the program
plot_mach_vs_cd()
    

#----------------------dev--------------------------------------
# def V2_rocket_drag_function(stages: List[Stage], mach: float) -> float:
#     # Drag function for V2
#     fig = plt.subplots()

#        # Use interpolation to claculate the drag coeffiecient for the given mach number
#     drag_coefficient: float = interpolate_cd(points, mach)

#     for stage in stages:
#         altitude = stage.get_lla_position_vector()[2, :]
#         # get the rocket's speed vector and calculate the magnitude
#         speed = np.linalg.norm(stage.velocity, axis=0) # using the velocity index we calculate the mach number at get_mach_number()
#         plt.plot(speed, drag_coefficient, "g")
#         # ax2.plot(stage.time, altitude, "b")

#     plt.xlabel("Speed (m/s)")
#     plt.ylabel("Drag Coefficient", color="g")
#     plt.grid(False)

#     plt.show()

# def interpolate_cd(points, x):
#     for i in range(len(points)-1):
#         if points[i][0] <= x <= points[i+1][0]:
#             slope = (points[i+1][1] - points[i][1]) / (points[i+1][0] - points[i][0])
#             y_intercept = points[i][1] - slope * points[i][0]
#             return slope * x + y_intercept
#     return None

# V2_rocket_drag_function()

# points = [(0.2, 0.15), (0.8, 0.17), (1.2, 0.42), (1.8, 0.25), (5.0, 0.15)]

#------------------------------------------------------------------------------------------
# def V2_rocket_drag_function(stages: List[Stage], mach: float) -> float:
#     # Drag function for V2
#     fig, ax1 = plt.subplots()
    
#     drag_coefficient: float = 0

#     if mach > 5:
#         drag_coefficient = 0.15
#     elif mach > 1.8 and mach <= 5:
#         drag_coefficient = -0.03125 * mach + 0.30625
#     elif mach > 1.2 and mach <= 1.8:
#         drag_coefficient = -0.25 * mach + 0.7
#     elif mach > 0.8 and mach <= 1.2:
#         drag_coefficient = 0.625 * mach - 0.35
#     elif mach <= 0.8:
#         drag_coefficient = 0.15

#     for stage in stages:
#         #altitude = stage.get_lla_position_vector()[2, :]
#         #speed = np.linalg.norm(stage.velocity, axis=0)
#         ax1.plot(stage.time, speed, "g")
#         # ax2.plot(stage.time, altitude, "b")

#     ax1.set_xlabel("MACH")
#     ax1.set_ylabel("CD)", color="g")
#     plt.grid(False)

#     plt.show()
