""""

TODO:
- create a program that 
- reconstract the gps 
- import real accell data to calculate the real vector forces and get the last GPS estimation.
- you don't need to do initial conditions just input it in the rocket.
- every timestep calculate the real data and ignore the simulated.
- once the accelleration data is finished ( due to loss in telemetry ), iterate as per a normal Rocket.simulation() with initial conditions.

"""


# -*- coding: utf-8 -*-
import numpy as np
from time import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from typing import List, Callable

from drag import *
from .utils import normalise_vector, lla2ecef
from .constants import *

from .stage import Stage
from .launchsite import Launchsite
from .rocket import Rocket
from .atmosphere import Atmosphere
from .real_data import Real_Data

from math import sin, cos



class Investigator:
    # The name of the rocket.
    name: str
    
    timestamp: int
    
    accel_z: float  # vertical
    accel_x: float  # horizontal

    latest_alt: float # meters

    rocket_params: Rocket

    # current_Stage: Stage

    """
    A list of stages of the rocket.
    The index 0 represents the first (lower) stage, while the last item represents the upper stage. """
    stages: List[Stage] = []
   
    """
    A user-defined function to calculate the drag characteristics of the rocket.
    The function takes the following parameters:
    0: List[Stage] - A list of connected stages to calculate the drag for.
    1: float - The current Mach number of the body.
    The function must return a float that represents the drag coefficient. """
    drag_function: Callable[[List[Stage], float], float]

    """
    User-defined function to calculate the thrust characteristics of the rocket.
    The function returns a thrust vector in Newtons.
    The function takes the following parameters:
    0: List[Stage] - A list of connected stages to calculate the thrust for.
    1: int - The current 'position index' of the body.
    The function must return an ECEF thrust vector. """
    thrust_function: Callable[[List[Stage], int], np.ndarray]
   
    launchsite: Launchsite

    def __init__(self, name: str,
                 launchsite: Launchsite,
                 # timestamp: int, # edit
                 accel_z: float,
                 accel_x: float,
                 latest_alt: float,
                 rocket_params: Rocket = None,
                 stages: List[Stage] = [],
                 use_cd_file: bool = False,
                 real_data: Real_Data = None, # allow an instance of this class as an argument
                 drag_function: Callable[[List[Stage], float], float] = V2_rocket_drag_function,
                 thrust_function: Callable[[List[Stage], int], np.ndarray] = None
                 #chute_cd: Stage.parachutes(get_chute_cd())
                 ):

        self.name = name
        self.launchsite = launchsite

        # FLIGHT LOG at loss of contact
        # self.timestamp = timestamp
        self.accel_z = accel_z
        self.accel_x = accel_x
        self.latest_alt = latest_alt
        # self.timestamp = timestamp
        
        # self.real_data = real_data # real cd plot and open rocket cd arrays
        self.use_cd_file = use_cd_file
        self.drag_function = drag_function
        self.thrust_function = thrust_function
        # allow the the class instance
        self.real_data = real_data
        # Set this to False to silence solver output
        self.logging_enabled = True

        if use_cd_file == True:
            self.drag_function = fetch_cd_function
            print("--------------------------------I USED THE CD FILE !!!")
        else:
            self.drag_function = V2_rocket_drag_function
            print("--------------------------------I USED THE V2_ROCKET FUNCTION !!!")


    
    def get_flight_state(self, stages, last_timestamp):
        active_stage = None
        stage_start_time = 0
        for stage in stages:
            stage_end_time = stage_start_time+ stage.burn_time
            if last_timestamp < stage_end_time:
                current_stage = stage
                break
            stage_start_time = stage_end_time + stage.separation_lag if stage.separation_lag is not None else stage_end_time
        if current_stage is None:
            # Rocket has landed
            return None
        stage_time = last_timestamp - stage_start_time
        stage_mass = current_stage.motor_burn_rate * (stage_time if stage_time < stage.burn_time else stage.burn_time)
        stage_thrust = current_stage.max_thrust if stage_time < stage.burn_time else 0
        
        return RocketStageState(current_stage, last_timestamp, stage_mass, stage_thrust)


    def calculate_properties(accel_x,accel_y,accel_z,altitude,timestamp):
        try:
            atmosphere = Atmosphere(altitude)
        except ValueError: atmosphere = None # no atmosphere
        state, active_stage = self.get_flight_state(stages, timestamp)
        # Calc mass
        stage.fuel_mass_vector[i] = stage.fuel_mass
        mass = stage.total_mass
        # Calculate thrust vector
        if i > 100 and np.all(stage.speed[:, 0] == 0):
            # The rocket might have landed.
            thrust = np.array([0, 0, 0])
        elif stage.has_separated:
            # The stage has separated, so there is no thrust
            thrust = np.array([0, 0, 0])
        else:
            # The stage is still attached and may have thrust
            if stage == self.stages[-1]:
                # This is the upper stage, so determine the amount of time
                # that the engine has been burning for
                engine_burn_time = stage.time[i] - stage.burn_start_time
                
                # Determine the thrust of the current active stage
                thrust_magnitude = 0
                if stage.fuel_mass >= 0:
                    if engine_burn_time < stage.burn_time:
                        thrust_magnitude = stage.thrust

        

       
    def load_real_data(self):
        """ Import Data Collected from Real Flight"""
        # TR2 Recorded Flight 1 (single stage)
        real_data = Real_Data()
        file = "C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/TR2_EasyMega_Flight_Data.csv"
        # time data
        time_rec, _ = real_data.read_csv_col(file, 3)   # remember this function returns tuple
        time_rec = real_data.resample_array(time_rec,TIME_STEP) # refitted to match the simulation TIME_STEP
        # acceleration data for the x, y, and z axes
        accel_x,_ = real_data.read_csv_col(file,15)
        accel_x = real_data.resample_array(accel_x,TIME_STEP) 

        accel_y,_ = real_data.read_csv_col(file,16)
        accel_y = real_data.resample_array(accel_y,TIME_STEP)

        accel_z,_ = real_data.read_csv_col(file,17)
        accel_z = real_data.resample_array(accel_z,TIME_STEP)

        # altitude
        alt_recorded,_ = real_data.read_csv_col(file,8)
        alt_recorded = real_data.resample_array(alt_recorded,TIME_STEP) 
        # velocity
        V_recorded,_ = real_data.read_csv_col(file,10)
        V_recorded = real_data.resample_array(V_recorded,TIME_STEP) 

        photograph = calculate_properties(accel_x,accel_y,accel_z,i)


    def calculate_fuel_mass(timestamp, fuel_mass, burn_rate):
        initial_timestamp = 0
        time_elapsed = timestamp - initial_timestamp
        fuel_mass = fuel_mass - burn_rate * time_elapsed

        return max(0, fuel_mass) # Ensure fuel mass is never negative
    


    def calculate_position_vector(starting_point, direction, acceleration_x, acceleration_y, acceleration_z, time_elapsed):
        lat, lon, alt = starting_point
        phi = direction # phi is the angle between the direction of motion and the equator
        r = alt # r is the distance from the center of the earth
        theta = lon # theta is the angle between the direction of motion and the prime meridian
        
        # Calculate the new position
        new_lat = lat + r * cos(phi) * cos(theta) * time_elapsed
        new_lon = lon + r * cos(phi) * sin(theta) * time_elapsed
        new_alt = alt + r * sin(phi) * time_elapsed
        lla = lla[0]
        
        # Update the velocity based on the acceleration
        vel_x = acceleration_x * time_elapsed
        vel_y = acceleration_y * time_elapsed
        vel_z = acceleration_z * time_elapsed
        velocity = np.array[vel_x,vel_y,vel_z]
        
        return (new_lat, new_lon, new_alt), velocity



        # Copy the data from the real stages into the simulation stages
        for i, stage in enumerate(self.stages):
            stage.position[:, :len(real_stages[i].position[0])] = real_stages[i].position
            stage.velocity[:, :len(real_stages[i].velocity[0])] = real_stages[i].velocity
            stage.lla_vector[:, :len(real_stages[i].lla_vector[0])] = real_stages[i].lla_vector
            stage.surface_position[:, :len(real_stages[i].surface_position[0])] = real_stages[i].surface_position
            stage.burn_start_time = real_stages[i].burn_start_time
            stage.fuel_mass = real_stages[i].fuel_mass
            stage.fuel_mass_vector[:len(real_stages[i].fuel_mass_vector)] = real_stages[i].fuel_mass_vector
            stage.has_separated = real_stages[i].has_separated
            stage.has_impacted_ground = real_stages[i].has_impacted_ground

        




        # calculate velocity using the trapezoidal rule
        vx = np.cumsum((ax[:-1] + ax[1:]) / 2 * np.diff(t))
        vy = np.cumsum((ay[:-1] + ay[1:]) / 2 * np.diff(t))
        vz = np.cumsum((az[:-1] + az[1:]) / 2 * np.diff(t))

        # calculate position using the trapezoidal rule
        x = np.cumsum((vx[:-1] + vx[1:]) / 2 * np.diff(t))
        y = np.cumsum((vy[:-1] + vy[1:]) / 2 * np.diff(t))
        z = np.cumsum((vz[:-1] + vz[1:]) / 2 * np.diff(t))





    def run_investigation(self):
        # set single use flag
        looking_for_apogee = True
        apogee_time = None
        # set initial position and velocity of each stage
        for stage in self.stages:
            stage.position[:, 0] = self.launchsite.position_ECEF
            stage.velocity[:, 0] = self.launchsite.initial_velocity         # TODO: add real velocity. If you have real velocity use it, else iterate.

            # initial lat-long-alt and surface position calculated
            lat, long, alt = stage.get_lla(0)
            stage.lla_vector[:, 0] = np.array([lat, long, alt])
            stage.surface_position[:, 0] = lla2ecef(lat, long, 0)

        i = 0 # timestamp for sim start ?
        active_stage_index = 0
        last_progress_print = time() - 10

        while i < N_TIME_INTERVALS - 1:
            if i in REAL_DATA_TIME_INTERVALS:
                self.load_real_data(i)
            i += 1 

            # active stage represents the lowest connected stage
            # i.e the stage that can currently provide thrust
            active_stage: Stage = self.stages[active_stage_index]
            # upper stage is the last item in the stages list
            upper_stage: Stage = self.stages[-1]

            # stop solving if the upper stage has impacted the ground
            if upper_stage.has_impacted_ground:
                self.log(f"[{upper_stage.name}] Impacted ground")

                for stage in self.stages:
                    self.log(f"[{stage.name}] range is {round(stage._range[-1] / 1e3, 1)} km")
                    altitude = stage.get_lla_position_vector()[2, :]
                    apogee = np.amax(altitude)
                    # print(len(stage.time)) #  DEBUG
                    
                    self.log(f"[{stage.name}] Apogee is {round(apogee, 1)} m at {stage.time[-1]} seconds")
                    
                break

            # 'start the motor' if not started yet  
            if active_stage.burn_start_time is None:
                active_stage.burn_start_time = upper_stage.time[i - 1] # iterate

            # iterate through each stage. Forces are evaluated for each stage
            # and stages are skipped if they are joined to other stages.

            # for multiple-stage "bodies", trajectory is calculated on the 
            # upper stage and then copied to lower stages
            for stage in self.stages:
                # if the stage has impacted the ground then skip it
                if stage.has_impacted_ground:
                    continue

                # determine altitude using ecef2lla function.
                # we pass in the "position index"
                # a position index of i-1 represents the previous position
                lla = stage.lla_vector[:, i - 1] # iterate
                latitude, longitude, altitude = lla[0], lla[1], lla[2]

                # update the store about the fuel mass of each stage.
                # for the purposes of data visualisation and understanding 
                # how the fuel was burned
                stage.fuel_mass_vector[i] = stage.fuel_mass

                # by default each stage has no
                thrust_magnitude = 0
                if stage.has_separated:
                    # if the stage has separated then for the rest of this
                    # time interval, only consider the dynamics of this stage
                    stages_to_consider = [stage]
                elif stage == upper_stage:
                    # otherwise, the stage is connected to other stages
                    # and hence we must consider all stages that are above the 
                    # active stage.

                    # for example, we must calculate mass by summming the 
                    # mass of all stages in the list below
                    stages_to_consider = self.stages[active_stage_index::]

                    # determine the amount of time that the engine on the 
                    # current active stage has been burning for
                    engine_burn_time = active_stage.time[i] - active_stage.burn_start_time

                    # if the stage has fuel left then continue
                    if active_stage.fuel_mass >= 0:
                        # only create thrust if the stage has time left to burn
                        if engine_burn_time < active_stage.burn_time:
                            # set the thrust of the currently considered body
                            # to the thrust of the active (lower) stage
                            thrust_magnitude = active_stage.thrust
                else:
                    # this path is taken by stages that do not need to have a 
                    # new solution made for their trajectory. Example would be 
                    # a middle or bottom stage during the first stage burn.

                    # trajectories are calculated from the uppermost stage of 
                    # each body of stages defined in stages_to_consider
                    continue

                # remaining stages are separated stages or upper stage
                # calculations for multiple stages connected together are 
                # done on the upper stage and copied to lower stages

                # determine mass

                mass = sum([_stage.total_mass for _stage in stages_to_consider])

                # determine atmospheric properties
                try:
                    atmosphere = Atmosphere(altitude)
                except ValueError:
                    # we have left the atmosphere
                    atmosphere = None

                # determine thrust vector
                if i == 1 and np.all(stage.velocity[:, 0] == 0):    # TODO  use real velocity 
                    # if on the launch pad then the thrust is oriented in the 
                    # same direction as the launchsite
                    # unless the stage has an initial velocity
                    thrust_vector = self.launchsite.orientation_ECEF * thrust_magnitude
                elif self.thrust_function is not None and thrust_magnitude > 0:
                    thrust_vector = self.thrust_function(stages_to_consider, i - 1)
                else:
                    # otherwise assume that the thrust is oriented in the 
                    # same direction as the current velocity
                    thrust_vector = thrust_magnitude * normalise_vector(stage.velocity[:, i - 1])

                # determine gravity vector
                gravity_vector = stage.get_gravity_vector(i - 1, mass)

                # determine drag vector
                mach_number = stage.get_mach_number(i, atmosphere)
                if mach_number is None:
                    Cd = 0
                else:
                    Cd = self.drag_function(stages_to_consider, mach_number, self.real_data)
                A = max([_stage.cross_sectional_area for _stage in stages_to_consider])
                if Cd == 0:
                    # coefficient of drag = 0 corresponds to no drag
                    drag_vector = np.array([0, 0, 0]) # pos neg pos
                else:
                    velocity = stage.velocity[:, i - 1]
                    velocity_magnitude = np.sqrt(velocity.dot(velocity))
                    if velocity_magnitude != 0:
                        drag_direction = - normalise_vector(stage.velocity[:, i - 1])
                    else:
                        drag_direction = 0
                    drag_vector = 1 / 2 * atmosphere.density * A * Cd * velocity_magnitude ** 2 * drag_direction
                    # self.log(f"[{self.name}] [{stage.time[i]}] DRAG {active_stage.name} with vector  {drag_vector} ") # ---------------
                # calculate centrifugal force

                # omega represents rotation vector pointed along Z axis
                omega = np.array([0, 0, (2 * np.pi) / 86400])
                r = stage.position[:, i - 1]
                v_r = stage.velocity[:, i - 1]

                f_centrifugal = np.cross(- mass * omega, np.cross(omega, r))
                f_coriolis = np.cross(-2 * mass * omega, v_r)

                # add up forces acting on the rocket
                sum_forces = drag_vector + gravity_vector + thrust_vector + f_centrifugal + f_coriolis 
                # + chute_drag_vector
                # determine acceleration (F=ma)
                acceleration = sum_forces / mass
                
                dt = stage.time[i] - stage.time[i - 1]
                
                # apogee time
                if looking_for_apogee==True and i>10:
                    prev_alt = stage.get_lla_position_vector()[2, i - 2]
                    current_alt = stage.get_lla_position_vector()[2, i-1]
                    if prev_alt > current_alt:
                        apogee_time = stage.time[i-1]
                        # self.log(f"-----------------------------------Apogee detected at time [{apogee_time}]{prev_alt}]{current_alt}].") # DEBUG
                        looking_for_apogee = False           

                # propagate forward velocity and position based on current acceleration
                stage.acceleration[:, i] = np.copy(acceleration)
                stage.velocity[:, i] = stage.velocity[:, i - 1] + acceleration * dt
                stage.position[:, i] = stage.position[:, i - 1] + stage.velocity[:, i] * dt

                # update data stores of other useful variables about the flight
                stage.drag_vectors[:, i] = drag_vector
                stage.thrust_vectors[:, i] = thrust_vector
                stage.gravity_vectors[:, i] = gravity_vector

                lat, long, alt = stage.get_lla(i)
                stage.lla_vector[:, i] = np.array([lat, long, alt])

                # approximate the distance downrange
                stage.surface_position[:, i] = lla2ecef(latitude, longitude, 0)
                stage._range[i] = stage._range[i - 1] + \
                                  np.linalg.norm(stage.surface_position[:, i] - stage.surface_position[:, i - 1])     

                if stage == upper_stage:
                    # copy trajectory data to the connected stages
                    for other_connected_stage in stages_to_consider[:-1]:
                        other_connected_stage.acceleration[:, i] = np.copy(stage.acceleration[:, i])
                        other_connected_stage.velocity[:, i] = np.copy(stage.velocity[:, i])
                        other_connected_stage.position[:, i] = np.copy(stage.position[:, i])
                        other_connected_stage.drag_vectors[:, i] = np.copy(drag_vector)
                        other_connected_stage.thrust_vectors[:, i] = np.copy(thrust_vector)
                        other_connected_stage.gravity_vectors[:, i] = np.copy(gravity_vector)
                        other_connected_stage.lla_vector[:, i] = np.copy(stage.lla_vector[:, i])
                        other_connected_stage.surface_position[:, i] = np.copy(stage.surface_position[:, i])
                        other_connected_stage._range[i] = stage._range[i]

                    # trigger stage separation
                    engine_burn_time = active_stage.time[i] - active_stage.burn_start_time
                    if active_stage.burn_time + active_stage.separation_time < engine_burn_time and active_stage != upper_stage:
                        self.log(f"[{self.name}] [{stage.time[i]}] Stage {active_stage.name} separated!")
                        active_stage.has_separated = True
                        active_stage_index += 1

                # calculate decrease in mass due to fuel burn
                if thrust_magnitude > 0:
                    active_stage.fuel_mass -= (thrust_magnitude * dt) / (GRAVITY * active_stage.specific_impulse)
                    active_stage.total_mass = active_stage.dry_mass + active_stage.fuel_mass

                if altitude < 0 and stage.time[i - 1] > 1:
                    # if the upper stage hits the ground before the other stages are done solving, truncate those as well
                    if stage == upper_stage:
                        stages_to_truncate = list(
                            filter(
                                lambda x: x.has_impacted_ground == False,
                                self.stages
                            )
                        )
                    else:
                        stages_to_truncate= [stage]

                    for _stage in stages_to_truncate:
                        # stage has impacted ground, truncate data stores and skip this stage
                        # this prevents the graphs from having zeros on the end
                        _stage.position = stage.position[:, 0:i]
                        _stage.velocity = stage.velocity[:, 0:i]
                        _stage.acceleration = stage.acceleration[:, 0:i]
                        _stage.gravity_vectors = stage.gravity_vectors[:, 0:i]
                        _stage.thrust_vectors = stage.thrust_vectors[:, 0:i]
                        _stage.drag_vectors = stage.drag_vectors[:, 0:i]
                        _stage.lla_vector = stage.lla_vector[:, 0:i]
                        _stage.surface_position = stage.surface_position[:, 0:i]
                        _stage._range = stage._range[0:i]
                        _stage.time = stage.time[0:i]
                        _stage.fuel_mass_vector = stage.fuel_mass_vector[0:i]
                        _stage.has_impacted_ground = True

            # print out current solving progress to the user
            if time() - last_progress_print > 0.1:
                percentage = round(100 * (i / N_TIME_INTERVALS))
                print(f"Investigation progress: [{percentage}%]")
                last_progress_print = time()

 