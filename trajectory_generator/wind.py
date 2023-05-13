from trajectory_generator.atmosphere import Atmosphere
from trajectory_generator.rocket import Rocket
from trajectory_generator.stage import Stage

# class modules
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import warnings
import math

import time # timeout
import random # monte carlo sim
from mpl_toolkits.basemap import Basemap # TODO: add to requirements,txt 
from mpl_toolkits.mplot3d import Axes3D



"""
TODO:
Adding a method to simulate turbulence for different altitudes using the simulate_turbulence method in the Atmosphere class.

Adding a method to generate a wind profile by combining the turbulence vectors with wind direction vectors for different altitudes.

Adding a method to save the wind profile to a file.

Adding methods to plot the wind profile in 3D and 2D.
"""

class Wind:


    def __init__(self, altitude):
        self.atmosphere = Atmosphere(altitude)
        # self.layer = Atmosphere._get_layer_nums()


    def simulate_turbulence(self,frequency, plotting=True):
        """
        Generate pink noise, filter it to simulate turbulence, and extract samples at a specified frequency.

        Params:
            frequency (float): The frequency (per second) at which to extract samples from the pink noise.

        Returns:
            None
        """
        # Generate white noise with 10000 samples
        white_noise = np.random.randn(10000)

        if frequency == 0:
            pink_noise = 0
            samples = 0
            if plotting == True:
             warnings.warn("Turbulence Frequency is 0.")
        else:
            # Filter white noise to create pink noise
            b, a = signal.butter(1, 1/(2*np.pi*frequency), 'highpass', fs=1000)
            pink_noise = signal.filtfilt(b, a, white_noise)

            # Extract samples from every 50th sample at the given frequency
            # i.e. every 50th sample to simulate turbulence at 20 Hz.
            # Extract samples i.e.at 20 Hz
            samples = pink_noise[::int(1000/frequency)]

            # Apply absolute function to ensure all values are positive
            samples = np.abs(samples)
            # Normalise the data to lie between 0 and 1.
            samples = samples / np.max(samples)
            
            # Reshape into a 2-dimensional array: [no of samples, no of channels]: [samples,1]
            samples = samples.reshape(-1,1)
            # samples_20hz = pink_noise[::50]

        if plotting == True:
            # Plot white and pink noise
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].plot(white_noise)
            ax[0].set_xlabel('Time (s)', fontsize = 8)
            ax[0].set_ylabel('Amplitude')
            ax[0].set_title('White noise')


            ax[1].plot(pink_noise)
            ax[1].set_xlabel('Time (s)', fontsize = 8)
            ax[1].set_ylabel('Amplitude')
            ax[1].set_title('Pink noise')

                    
            for ax in ax:
                ax.tick_params(labelsize=6) # adjust tick font size


            # Plot samples at the given frequency
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.plot(samples)
            ax.set_xlabel('Time (s)', fontsize = 8)
            ax.set_ylabel('Amplitude')
            ax.set_title('Samples at {} Hz'.format(frequency))

            ax.tick_params(labelsize=10) # adjust tick font size

            fig.subplots_adjust(hspace=4.5)
            plt.show()

        return samples


    def wind_direction(self,direction=None): 
        """
        Generate a wind direction vector based on input direction.

        Params:
            direction (str): The direction of the wind as a string, e.g. 'north', 'northwest', 'south', etc.

        Returns:
            wind_dir (np.ndarray): A 3D wind direction vector.
                                    [x,y,z]
        # TODO: Improvet to direction based on 360 deg to rad
        """
        # Define unit vectors in each direction
        # X, Y , Z  axis :  North/South , East/West, Up, Down
        no_wind = np.array([0, 0, 0])
        north = np.array([1, 0, 0])
        south = np.array([-1, 0, 0])
        east = np.array([0, 1, 0])
        west = np.array([0, -1, 0])
        up = np.array([0, 0, 1])
        down = np.array([0, 0, -1])
        northeast = np.array([1, 1, 0])
        northwest = np.array([1, -1, 0])
        southeast = np.array([-1, 1, 0])
        southwest = np.array([-1, -1, 0])

        # Define wind direction vector based on input direction
        if  direction == None:
            wind_dir = no_wind
        elif direction == 'north':
            wind_dir = north
        elif direction == 'south':
            wind_dir = south
        elif direction == 'east':
            wind_dir = east
        elif direction == 'west':
            wind_dir = west
        elif direction == 'up':
            wind_dir = up
        elif direction == 'down':
            wind_dir = down
        elif direction == 'northeast':
            wind_dir = northeast
        elif direction == 'northwest':
            wind_dir = northwest
        elif direction == 'southeast':
            wind_dir = southeast
        else:
            # direction == 'southwest'
            wind_dir = southwest

        return wind_dir

    def turbulence_vector(self,frequency, direction=None):
        """
        Generate a turbulence vector based on input frequency and direction.

        Params:
            frequency (float): The frequency (per second) at which to extract samples from the pink noise.
            direction (str): The direction of the wind as a string, e.g. 'north', 'northwest', 'south', etc.

        Returns:
            turbulence_vector (np.ndarray): A 3D turbulence vector.
                                                [x,y,z]
        """
        # Generate turbulence samples
        samples = self.simulate_turbulence(frequency, plotting=False)

        # Calculate turbulence magnitude
        turbulence_magnitude = np.std(samples)

        # Get wind direction vector
        wind_dir = self.wind_direction(direction)

        # Multiply wind direction vector by turbulence magnitude
        turbulence_vector = wind_dir * turbulence_magnitude

        return turbulence_vector

    def plot_wind(wind_object, freq, direction):
        # Get wind direction vector
        wind_dir = wind.wind_direction(direction)

        # Get turbulence vector
        turbulence = wind.turbulence_vector(freq, direction)

        # Create 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot wind vector
        ax.quiver(0, 0, 0, wind_dir[0], wind_dir[1], wind_dir[2], color='blue', label='Wind Vector')

        # Plot turbulence vector
        ax.quiver(0, 0, 0, turbulence[0], turbulence[1], turbulence[2], color='red', label='Turbulence Vector')

        # Set limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('North/South')
        ax.set_ylabel('East/West')
        ax.set_zlabel('Up/Down')

        # Add legend
        ax.legend()

        plt.show()

    def generate_wind_profile(self,altitudes, wind_speed, wind_dir, turbulence_freq):
        """
        Generate a wind profile by combining the turbulence vectors with wind direction vectors for different altitudes.

        Parameters:
        altitudes   (list): List of altitudes at which the wind profile needs to be generated.
        wind_speed  (float): Mean wind speed at the ground level. # TODO: Add to class objects. User should obtain from weather forecast. 
        wind_dir    (str): Wind direction at the ground level in degrees.
        turbulence_freq (int): Turbulence intensity at the ground level.

        Returns:
        wind_profile (list): A list of 3D vectors representing the wind profile at different altitudes.
        """
        # Generate wind profile at different altitudes
        wind_profile = []  # x , y , z
        for altitude in altitudes:
            turbulence_vector = self.turbulence_vector(turbulence_freq, wind_dir)
            wind_dir_vector = self.wind_direction(wind_dir)
            # Compute the altitude factor based on the input altitude. 
            # The altitude factor is used to adjust the magnitude of the wind vectors based on the altitude.
            altitude_factor = (altitude / 10) ** (1 / 7)

            # Set wind wind speed to vary with altitude according to a power law. # TODO: backup with source and specify per atmospheric layer.
            wind_speed_factor = wind_speed * altitude_factor

            # Combine the wind direction and the turbulence vector.
            # This represents the direction and magnitude of the wind at the given altitude and location.
            wind_vector = [wind_dir_vector[i] * wind_speed_factor + turbulence_vector[i] for i in range(3)]
            wind_profile.append(wind_vector)
        return wind_profile
        
        
    def plot_wind_profile_3d(wind_profile):
        """
        Plot the wind profile in 3D.

        Parameters:
        wind_profile (list): A list of 3D vectors representing the wind profile.

        Returns:
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for vector in wind_profile:
            ax.quiver(0, 0, 0, vector[0], vector[1], vector[2])
        plt.show()

    def plot_wind_profile_2d(wind_profile):
        """
        Plot the wind profile in 2D.

        Parameters:
        wind_profile (list): A list of 3D vectors representing the wind profile.

        Returns:
        None
        """
        plt.xlabel('X')
        plt.ylabel('Y')
        for vector in wind_profile:
            plt.quiver(0, 0, vector[0], vector[1], scale=100)
        plt.show()

    # stage.apogee_direction = []
    # stage.apogee_position = []
    

    wind_speed=1.0          # test
    wind_direction= 45.0       # test
    position = [53.38165675,-1.4821555953306795, 10]  # drop test from Sheffield coordinates
    apogee_direction = 45.0   # test
    
    def descent_vector(self,wind_speed,wind_direction):
        # deg to rad
        wind_direction_radians = math.radians(wind_direction)
        print("wind_direction_radians:", wind_direction_radians)
        print("wind_speed:", wind_speed)
        print("apogee_direction:", self.apogee_direction)
        # x and y components are based on the wind speed and direction, 
        # the z component is based on the rocket's apogee direction and position.
        x = -math.sin(wind_direction_radians) * wind_speed
        y = -math.cos(wind_direction_radians) * wind_speed
        z = -self.position[2] / math.tan(math.radians(self.apogee_direction))
        x = round(x, 2)
        y = round(y, 2)
        z = round(z, 2)
        print("x:", x)
        print("y:", y)
        print("z:", z)
        
        magnitude = math.sqrt(x**2 + y**2 + z**2)
        x = x / magnitude
        y = y / magnitude
        z = z / magnitude

        # update x and y components based on current position
        self.position[0] += x
        self.position[1] += y
        self.position[2] += z
        # x *= self.wind_speed
        # y *= self.wind_speed

        
        return self.position[:]
    

    def landing_area(self, timeout=20, num_trials=10):
        landing_spots = []
        altitudes = []
        start_time = time.time()

        for i in range(num_trials):
            # calculate wind speed and direction using Monte Carlo simulation
            wind_speed_mc = random.normalvariate(self.wind_speed, self.wind_speed * 0.1)
            wind_direction_mc = random.normalvariate(self.wind_direction, 2) # deviation 2

            position = self.position[:]
            altitude = position[2]

            # iteratively update the rocket's position
            while altitude >= 0:
                # check if timeout has been reached
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    print(f"Timeout reached after {timeout} seconds")
                    break

                descent = self.descent_vector(wind_speed_mc,wind_direction_mc)
                position[0] += descent[0]
                position[1] += descent[1]
                position[2] += descent[2]
                altitude -= descent[2]

            landing_spot = tuple(position[:2])
            landing_spots.append(landing_spot)
            altitudes.append(position[2])

        print(f"Rocket landed! Spot: {landing_spots} Number of trials: {num_trials}")
        print(f"Minimum altitude: {min(altitudes)}")
        print(f"Maximum altitude: {max(altitudes)}")

        # generate scatter plot of landing coordinates and altitudes
        # fig, ax = plt.subplots()
        # scatter = ax.scatter([x[0] for x in landing_spots], [x[1] for x in landing_spots], c=altitudes, cmap='cool')
        # ax.set_title("Landing Coordinates")
        # ax.set_xlabel("X Coordinate")
        # ax.set_ylabel("Y Ccordinate")
        # plt.show()
        
                
                # get the minimum and maximum latitudes and longitudes of the landing spots
        min_lat = min([spot[0] for spot in landing_spots])
        max_lat = max([spot[0] for spot in landing_spots])
        min_lon = min([spot[1] for spot in landing_spots])
        max_lon = max([spot[1] for spot in landing_spots])

        # set the map limits to the range of the landing spots
        padding = 0.1  # add a padding of 10% to the map limits
        mllat = min_lat - padding * abs(max_lat - min_lat)
        mllon = min_lon - padding * abs(max_lon - min_lon)
        mrlat = max_lat + padding * abs(max_lat - min_lat)
        mrlon = max_lon + padding * abs(max_lon - min_lon)

         # create the basemap object with the adjusted map limits
        m = Basemap(projection='mill', llcrnrlat=mllat, urcrnrlat=mrlat, \
                    llcrnrlon=mllon, urcrnrlon=mrlon, resolution='c')


        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='white',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
        x, y = m([spot[1] for spot in landing_spots], [spot[0] for spot in landing_spots])
        
        for i, spot in enumerate(landing_spots):
            plt.annotate(str(i+1), xy=(x[i], y[i]), xytext=(5, 5), textcoords='offset points')

        m.scatter(x, y, s=5, marker='o', color='r')

       

        plt.title("Landing Coordinates")
        plt.show()

        return landing_spots

""" Generate a wind profile with altitude range from 0 to 1000 meters """

# altitudes = list(range(0,1001,10)) # altitude range 0-1000 meters
# wind = Wind(altitudes)

# wind_profile = wind.generate_wind_profile(  altitudes,
#                                             wind_speed = 10, # at ground level
#                                             wind_dir = 'north',
#                                         turbulence_freq= 20 )


""" Plot the wind profile in 3D """
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# wind.plot_wind_profile_3d(ax, wind_profile)
# plt.show()

""" Plot the wind profile in 2D """
# fig, ax = plt.subplots()
# wind.plot_wind_profile_2d(ax, wind_profile)
# plt.show()




# wind = Wind(5000)
# wind.plot_wind(10, 'north')

# for altitude in range(100,1000,10):

#     wind = Wind(altitude)
#     print(f"Altitude: {altitude}, Turbulence Vector: {wind.turbulence_vector(20, 'north')}")

# # Plot the wind direction and amplitude for each altitude
# for altitude in range(100, 1010, 10):
#     wind = Wind(altitude)
#     amplitude = wind.turbulence_vector(20, 'north')
#     direction = np.deg2rad(np.linspace(0, 360, len(amplitude)))
#     plt.plot(direction, amplitude, label=f'{altitude} m')
# plt.legend()
# plt.xlabel('Direction (radians)')
# plt.ylabel('Amplitude')
# plt.title('Wind Turbulence at Different Altitudes')
# plt.show()

#Wind.simulate_turbulence(20, True)
# high_turb = Wind.simulate_turbulence(100, False)
# med_turb = Wind.simulate_turbulence(50, False)
# low_turb = Wind.simulate_turbulence(20, False)
# no_turb = Wind.simulate_turbulence(0, False)

# print(low_turb)
#print(Wind.wind_direction())





    # def get_wind_speed(self):
    #     if self.layer == 'troposphere':
    #         return self.atmosphere.wind_speed           # m/s
    #     elif self.layer == 'lower_stratosphere':
    #         return self.atmosphere.wind_speed + 10
    #     elif self.layer == 'upper_stratosphere':
    #         return self.atmosphere.wind_speed + 20
    #     elif self.layer == 'mesosphere':
    #         return self.atmosphere.wind_speed + 30
    #     else:
    #         return self.atmosphere.wind_speed + 40

    # def get_wind_direction(self):
    #     return self.atmosphere.wind_direction
    

    # def parachute(self):
    #     drift = []
    #     # Define parachute parameters
    #     parachute_area = 1.0  # m^2
    #     parachute_cd = 1.2  # drag coefficient

    #     # Define wind parameters
    #     wind_speed = 5  # m/s
    #     wind_cd = 0.5  # drag coefficient
    #     atmosphere = Atmosphere(altitude = Rocket.launchsite.altitude, model='exponential')
        
    #     # Calculate drag force
    #     altitude = Rocket.position[2]
    #     density = atmosphere.density(altitude)
    #     velocity = Rocket.velocity - wind_speed  # subtract wind speed from rocket velocity
    #     v_mag = norm(velocity)
    #     drag_force = 0.5 * density * v_mag**2 * parachute_cd * parachute_area  # drag force from parachute
    #     drag_force_side = 0.5 * density * v_mag**2 * wind_cd * Stage.diameter * Stage.length  # drag force from wind
    #     drag_force_vector = -drag_force * velocity / v_mag  # drag force vector
    #     drag_force_side_vector = -drag_force_side * velocity / v_mag  # drag force side vector

    #     # Resolve drag force into x, y, z components
    #     drag_force_x = drag_force_vector[0] + drag_force_side_vector[0]
    #     drag_force_y = drag_force_vector[1] + drag_force_side_vector[1]
    #     drag_force_z = drag_force_vector[2] + drag_force_side_vector[2]

    #     # Update rocket position and velocity using rocket equation
    #     Rocket.update(dt, drag_force_x, drag_force_y, drag_force_z)

    #     # Store simulation results
    #     t_list.append(t*dt)
    #     v_list.append(Rocket.velocity)
    #     pos_list.append(Rocket.position)
    #     return drift