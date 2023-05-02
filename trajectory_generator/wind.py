from .atmosphere import Atmosphere
from .rocket import Rocket
from .stage import Stage

class Wind:
    def __init__(self, altitude):
        self.atmosphere = Atmosphere(altitude)
        self.layer = Atmosphere._get_layer_nums()

    def get_wind_speed(self):
        if self.layer == 'troposphere':
            return self.atmosphere.wind_speed           # m/s
        elif self.layer == 'lower_stratosphere':
            return self.atmosphere.wind_speed + 10
        elif self.layer == 'upper_stratosphere':
            return self.atmosphere.wind_speed + 20
        elif self.layer == 'mesosphere':
            return self.atmosphere.wind_speed + 30
        else:
            return self.atmosphere.wind_speed + 40

    def get_wind_direction(self):
        return self.atmosphere.wind_direction
    

    def parachute(self):
        drift = []
        # Define parachute parameters
        parachute_area = 1.0  # m^2
        parachute_cd = 1.2  # drag coefficient

        # Define wind parameters
        wind_speed = 5  # m/s
        wind_cd = 0.5  # drag coefficient
        atmosphere = Atmosphere(altitude = Rocket.launcher.altitude, model='exponential')
        
        # Calculate drag force
        altitude = Rocket.position[2]
        density = atmosphere.density(altitude)
        velocity = Rocket.velocity - wind_speed  # subtract wind speed from rocket velocity
        v_mag = norm(velocity)
        drag_force = 0.5 * density * v_mag**2 * parachute_cd * parachute_area  # drag force from parachute
        drag_force_side = 0.5 * density * v_mag**2 * wind_cd * Stage.diameter * Stage.length  # drag force from wind
        drag_force_vector = -drag_force * velocity / v_mag  # drag force vector
        drag_force_side_vector = -drag_force_side * velocity / v_mag  # drag force side vector

        # Resolve drag force into x, y, z components
        drag_force_x = drag_force_vector[0] + drag_force_side_vector[0]
        drag_force_y = drag_force_vector[1] + drag_force_side_vector[1]
        drag_force_z = drag_force_vector[2] + drag_force_side_vector[2]

        # Update rocket position and velocity using rocket equation
        missile.update(dt, drag_force_x, drag_force_y, drag_force_z)

        # Store simulation results
        t_list.append(t*dt)
        v_list.append(missile.velocity)
        pos_list.append(missile.position)
        return drift

