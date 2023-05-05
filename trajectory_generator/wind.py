from trajectory_generator.atmosphere import Atmosphere
from trajectory_generator.rocket import Rocket
from trajectory_generator.stage import Stage

# class modules
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Wind:
    def __init__(self, altitude):
        self.atmosphere = Atmosphere(altitude)
        # self.layer = Atmosphere._get_layer_nums()

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
    #     atmosphere = Atmosphere(altitude = Rocket.launcher.altitude, model='exponential')
        
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


    def simulate_turbulence(frequency):
        """
        Generate pink noise, filter it to simulate turbulence, and extract samples at a specified frequency.

        Params:
            frequency (float): The frequency (per second) at which to extract samples from the pink noise.

        Returns:
            None
        """
         
        # Generate white noise with 10000 samples
        white_noise = np.random.randn(10000)

        # Filter white noise to create pink noise
        b, a = signal.butter(1, 1/(2*np.pi*frequency), 'highpass', fs=1000)
        pink_noise = signal.filtfilt(b, a, white_noise)

        # Extract samples from every 50th sample at the given frequency
        # i.e. every 50th sample to simulate turbulence at 20 Hz.
        samples = pink_noise[::int(1000/frequency)]

        # Extract samples at 20 Hz
        # samples_20hz = pink_noise[::50]

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