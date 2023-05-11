# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
from trajectory_generator.constants import *
import os
from ambiance import Atmosphere

class Real_Data:
    
    name: str = None

    # Define the desired sample time in seconds
    desired_sample_time: float = TIME_STEP

    # Define the file path to the CSV file
    real_data_path: str = "example/Raven 4 Kmini Relaunch - Flight 1 Data  - Altitude Baro.csv"
    sim_filepath: str =  "lower_stage_altitude_vs_time.csv"

    # import open rocket data file
    OR_data_filepath: str = "example/OR_karmanmini2.csv"
       
    # hex code for KML file styling
    kml_colour: str 


    def __init__(self, name, desired_sample_time, 
                 real_data_path, sim_filepath, OR_data_filepath,
                 kml_colour="ffffffff"):
        self.name = name
        self.desired_sample_time = desired_sample_time
        self.real_data_path = real_data_path
        self.sim_filepath = sim_filepath
        self.OR_data_filepath = OR_data_filepath
        self.kml_colour = kml_colour


        # define cross sectional area (m^2) for drag calculations
        if not real_data_path:
            print(f"WARNING: the filepath of the Real launch data is not defined for {self.name}.")
            

    # Define a function to resample the data based on a desired sample time
    def read_data_csv(self,launch_data__path):
        
        # Read the CSV file
        with open(launch_data__path, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            time = []
            altitude = []
            pressure =[]
            for row in reader:
                time.append(float(row[0]))
                altitude.append(float(row[1]))
                pressure.append(float(row[2]))

        return np.array(time), np.array(altitude)
    
    def resample_data(self,real_data_path):

        # Define the output file path for the resampled data
        base_file_name, extension = os.path.splitext(real_data_path)
        resampled_file = base_file_name + f"_resampled_{self.desired_sample_time:.1f}s" + extension

        # Read the CSV file
        time, altitude = self.read_data_csv(real_data_path)

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
                # print([time[i],altitude[i]],[resampled_time[i], resampled_altitude[i]]) # debug: read the resampled data

        # Return the path of the resampled file
        return resampled_file
    
    def environment_analysis(altitude):
        """
        takes a list of altitude measured
        """

        # determine atmospheric properties
        try:
            atmosphere = Atmosphere(altitude)
        except ValueError:
            # we have left the atmosphere
            atmosphere = None


        for row in altitude:
            atmosphere = Atmosphere(altitude[row])
            print("your current layer is" + atmosphere._get_layer_nums() )
         
            pressure = atmosphere.pressure
            print("pressure at these altitudes are" + pressure[row])


    # Define a function to plot the resampled data and simulation output of altitude plot
    def plot_data(self):

        # Resample the real data
        resampled_file = self.resample_data(self.real_data_path)
        sim_file = self.resample_data(self.real_data_path)
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

    """
    Manipulating OPENROCKET simulated data to improve the simulations : 
    """
    
    def replace_nan(self,array, replace_with=0):
        """
        Replaces all NaN values in a 1D numpy array with a specified value.

        Parameters:
        array (numpy.ndarray): The 1D numpy array to modify.
        replace_with (float): The value to replace NaN values with. Defaults to 0.

        Returns:
        numpy.ndarray: The modified numpy array.
        """
        mask = np.isnan(array)
        array[mask] = replace_with
        return np.nan_to_num(array, nan = replace_with)


    # This is tailored to OPEN ROCKET FILES 
    def read_OR_csv_col(self, OR_data_filepath,column):
        # Define a function to resample the data based on a desired sample time
 
        # Read the CSV file
        with open(self.OR_data_filepath, "r") as f:
            reader = csv.reader(f)
            # Skip 2 rows
            for _ in range(2):
                next(reader)

            # Save the column header
            col_header = next(reader)[column] 
            # Define where the column data will be saved
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



    """
    Manipulating EASYMEGA recored data to improve the simulations : 

    """

     # This is tailored to pre processed files (row[0]: skipped, row[1]: header, row[2+n]:data)
    def read_csv_col(self,real_data_filepath, column):
        # Define a function to resample the data based on a desired sample time
 
        # Read the CSV file
        with open(real_data_filepath, "r") as f:
            reader = csv.reader(f)
            # Skip 1 row
            for _ in range(1):
                next(reader)

            # Save the column header
            col_header = next(reader)[column] 
            # Define where the column data will be saved
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
   
   
    def plot_csv_cols(self,filepath, col_x, col_y):
        """
        Takes desired columns of file and plots them.
        """
        x = []
        y = []
        x , x_header = self.read_OR_csv_col(filepath, col_x)
        y , y_header = self.read_OR_csv_col(filepath, col_y)
        print(x_header , y_header)


        # Plot samples at the given frequency
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

        ax.plot(x,y)
        ax.set_xlabel('{}'.format(x_header), fontsize = 10)
        ax.set_ylabel('{}'.format(y_header), fontsize = 10)
        ax.set_title('{} vs {}'.format(x_header,y_header), fontsize = 12)
        plt.show()


    def fetch_cd(self):
        cd, _ = self.read_OR_csv_col(self.OR_data_filepath,29) # ignore the column header
        cd = self.replace_nan(cd, replace_with= 0) # this will replace all nan in the array with 0
        
        # print("--------------------------------I am in fetch_cd !!!!!!!!!")
        # print(cd) # DEBUG
        return cd

    def fetch_mach(self):
        mach, _ = self.read_OR_csv_col(self.OR_data_filepath,25) # ignore the column header
        mach = self.replace_nan(mach, replace_with= 0) # this will replace all nan in the array with 0
        
        # print("--------------------------------I am in fetch_mach !!!!!!!!!")
        # print(mach) # DEBUG
        return mach

    def plot_mach_cd(self):
        #mach = read_OR_csv_col(filepath,25)
        #cd = read_OR_csv_col(filepath,29)
        self.plot_csv_cols(self.OR_data_filepath,25,29) # 25 and 29 is the index of the column

    def get_time_accel_z(self):
        #mach = read_OR_csv_col(filepath,25)
        #cd = read_OR_csv_col(filepath,29)
        self.plot_csv_cols(self.OR_data_filepath,0,3) # 25 and 29 is the index of the column