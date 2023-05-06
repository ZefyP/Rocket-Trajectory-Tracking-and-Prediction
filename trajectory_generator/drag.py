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

    return drag_coefficient


# Define a function to resample the data based on a desired sample time
def read_csv_col(filepath,column):
    column = column.astype(np.float)# convert column to int

    
    # Read the CSV file
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        # Skip the first 6 rows
        for _ in range(6):
            next(reader)

        data_column = []
        for row in reader:
            print(([column]))
            if row[column]:
                data_column.append(float(row[column])) # OpenRocket standard export

    
    return np.array(data_column)


def plot_csv_cols(filepath, col_x, col_y):
    """
    Takes desired columns of file and plots them.
    """
    x = []
    y = []
    x = read_csv_col(filepath, col_x)
    y = read_csv_col(filepath, col_y)

    # Plot samples at the given frequency
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    ax.plot(x,y)
    #ax.set_xlabel('{}'.format(x), fontsize = 8)
    #ax.set_ylabel('{}'.format(y), fontsize = 8)
    #ax.set_title('{}} vs {}'.format(x,y))

    plt.show()


def get_cd_mach(filepath):

    mach = read_csv_col(filepath,25)
    cd = read_csv_col(filepath,29)

    plot_csv_cols(filepath,mach,cd)
    