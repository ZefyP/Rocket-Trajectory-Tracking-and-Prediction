# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import interpolate
from constants import TIME_STEP
import os

class Flight_Data:
    
    name: str = None

    # Define the desired sample time in seconds
    desired_sample_time: float = TIME_STEP

    # Define the file path to the CSV file
    file_path: str = "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv"
   
    # hex code for KML file styling
    kml_colour: str  

    def __init__(self, name, desired_sample_time, file_path,
                 kml_colour="ffffffff"):
        self.name = name
        self.file_path = file_path
        self.desired_sample_time = desired_sample_time
        self.kml_colour = kml_colour


        # define cross sectional area (m^2) for drag calculations
        if not file_path:
            print(f"WARNING: the filepath is not defined " +
                  "for {self.name}.")
            

    # Define a function to resample the data based on a desired sample time
    def resample_data(csv_file, desired_sample_time):
        # Define the output file path for the resampled data
        base_file_name, extension = os.path.splitext(csv_file)
        resampled_file = base_file_name + f"_resampled_{desired_sample_time:.1f}s" + extension

        # Read the CSV file
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            time = []
            altitude = []
            for row in reader:
                time.append(float(row[0]))
                altitude.append(float(row[2]))

        # Create an interpolation function
        f = interpolate.interp1d(time, altitude)

        # Resample the data using the interpolation function
        resampled_time = np.arange(time[0], time[-1], desired_sample_time)
        resampled_altitude = f(resampled_time)

        # Write the resampled data to a new CSV file
        with open(resampled_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Altitude"])
            for i in range(len(resampled_time)):
                writer.writerow([resampled_time[i], resampled_altitude[i]])

        # Return the path of the resampled file
        return resampled_file


    # Define a function to plot the resampled data
    def plot_resampled_data(resampled_file, desired_sample_time):
        # Read the resampled CSV file
        with open(resampled_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            resampled_time = []
            resampled_altitude = []
            for row in reader:
                resampled_time.append(float(row[0]))
                resampled_altitude.append(float(row[1]))

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot the resampled data
        ax.plot(resampled_time, resampled_altitude)

        # Set the axis labels and title
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title(f"Resampled Altitude vs. Time (Sample Time = {desired_sample_time} s)")

        # Show the plot
        plt.show()

    # Resample the data and get the path of the resampled file
    resampled_file = resample_data(file_path, desired_sample_time)

    # Plot the resampled data
    plot_resampled_data(resampled_file, desired_sample_time)
