# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import interpolate
from trajectory_generator.constants import *
import os

class Flight_Data:
    
    name: str = None

    # Define the desired sample time in seconds
    desired_sample_time: float = TIME_STEP

    # Define the file path to the CSV file
    file_path: str = "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv"
    sim_filepath: str =  "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/lower_stage_altitude_vs_time.csv"
   
    # hex code for KML file styling
    kml_colour: str  

    def __init__(self, name, desired_sample_time, file_path, sim_filepath,
                 kml_colour="ffffffff"):
        self.name = name
        self.file_path = file_path
        self.sim_filepath = sim_filepath
        self.desired_sample_time = desired_sample_time
        self.kml_colour = kml_colour
        


        # define cross sectional area (m^2) for drag calculations
        if not file_path:
            print(f"WARNING: the filepath is not defined for {self.name}.")
            

    # Define a function to resample the data based on a desired sample time
    def resample_data(self):
        # Define the output file path for the resampled data
        base_file_name, extension = os.path.splitext(self.file_path)
        resampled_file = base_file_name + f"_resampled_{self.desired_sample_time:.1f}s" + extension

        # Read the CSV file
        with open(self.file_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            time = []
            altitude = []
            for row in reader:
                time.append(float(row[0]))
                altitude.append(float(row[1]))

       # Create an interpolation function
        f = interpolate.interp1d(
            time, 
            altitude, 
            bounds_error=False, 
            fill_value=(altitude[0], altitude[-1]),
         )


        # Resample the data using the interpolation function
        resampled_time = np.arange(time[0], time[-1], self.desired_sample_time)
        resampled_altitude = f(resampled_time)

        # Write the resampled data to a new CSV file
        with open(resampled_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Altitude"])
            for i in range(len(resampled_time)):
                writer.writerow([resampled_time[i], resampled_altitude[i]])

        # Return the path of the resampled file
        return resampled_file


    # Define a function to plot the resampled data and simulation output of altitude plot
    def plot_data(self):

        # Resample the real data
        resampled_file = self.resample_data()
        sim_file = self.resample_data()
        
        # Store data from CSV files
        with open(resampled_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            resampled_time = []
            resampled_altitude = []
            for row in reader:
                resampled_time.append(float(row[0]))
                resampled_altitude.append(float(row[1]))

        with open(sim_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            sim_time = []
            sim_altitude = []
            for row in reader:
                sim_time.append(float(row[0]))
                sim_altitude.append(float(row[1]))


        # Check that resampled_time is within the range of sim_time
        if resampled_time[0] < sim_time[0] or resampled_time[-1] > sim_time[-1]:
            raise ValueError("Resampled real data time is outside the range of simulated time")

        
        # Interpolate the simulated data onto the same time points as the real data
        f_sim = interpolate.interp1d(sim_time, sim_altitude)
        #sim_altitude_resampled = f_sim(resampled_time)

        # Calculate the error between the resampled data and the simulated data
        error = np.array(resampled_altitude) - np.array(sim_altitude)
            
        # Calculate the mean error and max error
        mean_error = np.mean(error)
        max_error = np.max(error)

         # Create a figure and axis object
        fig, ax = plt.subplots(2,2)

        # Plot the resampled data and simulation output on the first subplot
        ax[0,0].plot(resampled_time, sim_altitude, label='Resampled Data')
        ax[0,0].plot(sim_time, sim_altitude, label='Simulation Output')
        ax[0,0].set_xlabel('Time (s)')
        ax[0,0].set_ylabel('Altitude (m)')
        ax[0,0].set_title('Altitude vs Time')
        ax[0,0].legend()


        # Plot the error between the resampled data and simulated data on the second subplot
        ax[0,1].plot(resampled_time, error, label='Error vs Time')
        ax[0,1].set_xlabel('Time (s)')
        ax[0,1].set_ylabel('Error (m)')
        ax[0,1].set_title('Error vs Time')
        ax[0,1].legend()

        # Print mean error and max error
        print(f"Mean Error: {mean_error:.2f} m")
        print(f"Max Error: {max_error:.2f} m")


        # # Create a figure and axis object
        # fig, ax2 = plt.subplots()

        # Plot data
        # ax1.plot(resampled_time, resampled_altitude, label='Real Launch Data')
        # ax1.plot(sim_time, sim_altitude_resampled, label='Simulation')

        # # Set the axis labels and title
        # ax1.set_xlabel("Time (s)")
        # ax1.set_ylabel("Altitude (m)")
        # ax1.set_title(f"Resampled Altitude vs. Time (Sample Time = {TIME_STEP} s)")
        # ax1.legend()

        # # Plot error data
        # ax2.plot(resampled_time, error, label='Error')
        # ax2.set_xlabel("Time (s)")
        # ax2.set_ylabel("Altitude Error (m)")
        # ax2.legend()


        # # Show the plot
        plt.show()

    
