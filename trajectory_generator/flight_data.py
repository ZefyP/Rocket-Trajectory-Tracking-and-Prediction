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
    def read_data_csv(self,filepath):
        
        # Read the CSV file
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            time = []
            altitude = []
            for row in reader:
                time.append(float(row[0]))
                altitude.append(float(row[1]))

        return np.array(time), np.array(altitude)
    
    def resample_data(self,filepath):

        # Define the output file path for the resampled data
        base_file_name, extension = os.path.splitext(filepath)
        resampled_file = base_file_name + f"_resampled_{self.desired_sample_time:.1f}s" + extension

        # Read the CSV file
        time, altitude = self.read_data_csv(filepath)

        # Resample the data using the interpolation function
        resampled_time = np.arange(time[0], time[-1], self.desired_sample_time)
        resampled_time = np.round(resampled_time, 3) # 3 decimal points

        resampled_altitude = np.interp(resampled_time, time, altitude)
        resampled_altitude = np.round(resampled_altitude, 3)

        # resampled_time = np.arange(max(time[0], sim_time[0]), min(time[-1], sim_time[-1]), self.desired_sample_time)
        # resampled_altitude = f(resampled_time)

        # Write the resampled data to a new CSV file
        with open(resampled_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Altitude"])
            for i in range(len(resampled_time)):
                writer.writerow([resampled_time[i], resampled_altitude[i]])
                print([time[i],altitude[i]],[resampled_time[i], resampled_altitude[i]]) # debug: read the resampled data

        # Return the path of the resampled file
        return resampled_file


    # Define a function to plot the resampled data and simulation output of altitude plot
    def plot_data(self):

        # Resample the real data
        resampled_file = self.resample_data(self.file_path)
        sim_file = self.resample_data(self.sim_filepath)
        # read
        resampled_time, resampled_altitude = self.read_data_csv(resampled_file)
        sim_resampled_time, sim_resampled_altitude = self.read_data_csv(sim_file)


        # Resize the arrays to have the same length
        length = min(len(resampled_altitude), len(sim_resampled_altitude))
        resampled_altitude = np.resize(resampled_altitude, length)
        resampled_time = np.resize(resampled_time, length)
        
        sim_resampled_altitude = np.resize(sim_resampled_altitude, length)
        sim_resampled_time = np.resize(sim_resampled_time, length)

    
        # Calculate the error between the resampled data and the simulated data
        error = resampled_altitude - sim_resampled_altitude
        #error = np.abs((resampled_altitude - sim_resampled_altitude) / resampled_altitude)
        error[np.isnan(error)] = 0
        
        #percentage_error = abs(error) / (resampled_altitude[:len(sim_resampled_altitude)] + 10) * 100
        percentage_error = (error / (resampled_altitude)) * 100
        percentage_error[np.isnan(percentage_error)] = 0

        
        for i in range(len(error)):
            perc_error = abs(error[i]) / sim_resampled_altitude[i] * 100
            if perc_error > 100:
                print(f"Index: {i}, Altitude Error: {error[i]:.2f}, Percent Error: {perc_error:.2f}%")
            



        # compute percentage error
        # percentage_error = (error / resampled_altitude) * 100
        # clip percentage error to [-100%, 100%] range
        # percentage_error = np.clip(percentage_error, -100, 100)


        # Calculate the mean error and max error
        mean_error = np.mean(error)
        max_error = np.max(error)




        # Create a figure and axis object
        fig, ax = plt.subplots(1,3)

        # Plot the resampled data and simulation output on the first subplot
        ax[0].plot(resampled_time, resampled_altitude, label='Real Data')
        ax[0].plot(sim_resampled_time, sim_resampled_altitude, label='Simulation Output')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Altitude (m)')
        ax[0].set_title('Altitude vs Time')
        ax[0].legend()

        # Plot the error between the resampled data and simulated data on the second subplot
        ax[1].plot(resampled_time, error, label='Altitude Error vs Time')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Error (m)')
        ax[1].set_title('Error vs Time')
        ax[1].legend()

        # Plot the error between the resampled data and simulated data on the second subplot
        ax[2].plot(resampled_time, percentage_error, label='Percentage Error vs Time')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Error (%)')
        ax[2].set_title('Percentage Error')
        ax[2].legend()

        # Print mean error and max error
        print(f"Mean Error: {mean_error:.2f} m")
        print(f"Max Error: {max_error:.2f} m")

        # # Show the plot
        plt.show()







        
        # # Store data from CSV files
        # with open(resampled_file, "r") as f:
        #     reader = csv.reader(f)
        #     next(reader) # skip header
        #     resampled_time = []
        #     resampled_altitude = []
        #     for row in reader:
        #         resampled_time.append(float(row[0]))
        #         resampled_altitude.append(float(row[1]))

        # with open(sim_file, "r") as f:
        #     reader = csv.reader(f)
        #     next(reader) # skip header
        #     sim_time = []
        #     sim_altitude = []
        #     for row in reader:
        #         sim_time.append(float(row[0]))
        #         sim_altitude.append(float(row[1]))


        # Check that resampled_time is within the range of sim_time
        # if resampled_time[0] < sim_time[0] or resampled_time[-1] > sim_time[-1]:
        #     raise ValueError("Resampled real data time is outside the range of simulated time")
        
        # # Interpolate the simulated data onto the same time points as the real data
        # f_sim = interpolate.interp1d(sim_time, sim_altitude)
        # sim_altitude_resampled = f_sim(sim_time)

        # Calculate the error between the resampled data and the simulated data
        # new_resampled_time = np.linspace(resampled_time[0], resampled_time[-1], len(sim_time))
        # error = np.array(resampled_altitude) - np.array(f_sim(new_resampled_time))
        # print(resampled_altitude, "   ", sim_altitude, "            ", error)

        



        
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