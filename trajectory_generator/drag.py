# -*- coding: utf-8 -*-
"""
 This drag function returns the drag coefficient for the given Mach number based on a set of conditional statements. 
 The data is derived from a reference source, Sutton's "Rocket Propulsion Elements", 7th edition, page 108. 
"""
from trajectory_generator.stage import Stage
from typing import List

import csv
import numpy as np
import matplotlib.pyplot as plt


def V2_rocket_drag_function(stages: List[Stage], mach: float) -> float:
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
        # drag_coefficient = 0.52 # karman mini 2

    return drag_coefficient


# Define a function to resample the data based on a desired sample time
# This is tailored to OPEN ROCKET FILES 
def read_OR_csv_col(filepath,column):
    #column = int[column]# convert column to int

    
    # Read the CSV file
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        # Skip 2 rows
        for _ in range(2):
            next(reader)

        # Save the column header
        col_header = next(reader)[column]
        
        data_column = []

 # Save the data and skip empty and non-numerical cells
        for row in reader:
            # print(([column])) # DEBUG
            try:
                cell = float(row[column])
            except ValueError:
                    continue
            data_column.append(cell) # OpenRocket standard export

    return ( np.array(data_column), col_header )


def plot_csv_cols(filepath, col_x, col_y):
    """
    Takes desired columns of file and plots them.
    """
    x = []
    y = []
    x , x_header = read_OR_csv_col(filepath, col_x)
    y , y_header = read_OR_csv_col(filepath, col_y)
    print(x_header , y_header)


    # Plot samples at the given frequency
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    ax.plot(x,y)
    ax.set_xlabel('{}'.format(x_header), fontsize = 10)
    ax.set_ylabel('{}'.format(y_header), fontsize = 10)
    ax.set_title('{} vs {}'.format(x_header,y_header), fontsize = 12)
    plt.show()

def get_mach_cd(filepath):
    #mach = read_OR_csv_col(filepath,25)
    #cd = read_OR_csv_col(filepath,29)
    plot_csv_cols(filepath,25,29) # 25 and 29 is the index of the column

def get_time_accel_z(filepath):
    #mach = read_OR_csv_col(filepath,25)
    #cd = read_OR_csv_col(filepath,29)
    plot_csv_cols(filepath,0,3) # 25 and 29 is the index of the column