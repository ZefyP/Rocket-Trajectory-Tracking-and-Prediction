# -*- coding: utf-8 -*-

from .stage import Stage
from typing import List, Callable
import numpy as np
from .drag import V2_rocket_drag_function
from .launcher import Launcher
from .ambiance import Atmosphere
from .utils import normalise_vector, lla2ecef
from .constants import *
from time import time
import matplotlib.pyplot as plt
import pickle

plt.style.use("ggplot")


class Rocket:
    # name of the rocket
    name: str

    # country operating the rocket
    country: str

    # list of stages of the rocket
    # index 0 is the first (lower) stage, last item represents the "warhead"
    stages: List[Stage] = []

    # user defined drag characteristics of rocket
    # takes a function with the following parameters:
    # 0: List[Stage] - list of connected stages to calculate drag for
    # 1: float - current mach number of body
    # the function must return a float, the drag coefficient
    drag_function: Callable[[List[Stage], float], float]

    # user defined thrust characteristics of rocket 
    # returns a thrust vector in Newtons
    # 0: List[Stage] - list of connected stages to calculate thrust for
    # 1: int - current 'position index' of body
    # the function must return an ECEF thrust vector
    
    
    
    thrust_function: Callable[[List[Stage], int], np.ndarray]

    launcher: Launcher

    def __init__(self, name: str, country: str, launcher: Launcher,
                 drag_function: Callable[[List[Stage], float], float] = V2_rocket_drag_function,
                 thrust_function: Callable[[List[Stage], int], np.ndarray] = None):
        self.name = name
        self.country = country
        self.launcher = launcher

        self.drag_function = drag_function
        self.thrust_function = thrust_function

        # set this to False to silence solver output
        self.logging_enabled = True

    def run_simulation(self):
        # set initial position and velocity of each stage
        for stage in self.stages:
            stage.position[:, 0] = self.launcher.position_ECEF
            stage.velocity[:, 0] = self.launcher.initial_velocity

            # initial lat-long-alt and surface position calculated
            lat, long, alt = stage.get_lla(0)
            stage.lla_vector[:, 0] = np.array([lat, long, alt])
            stage.surface_position[:, 0] = lla2ecef(lat, long, 0)

        i = 0
        active_stage_index = 0
        last_progress_print = time() - 10

        while i < N_TIME_INTERVALS - 1:
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
                    self.log(f"[{stage.name}] apogee is {round(apogee / 1000, 1)} km")
                break

            # 'start the motor' if not started yet  
            if active_stage.burn_start_time is None:
                active_stage.burn_start_time = upper_stage.time[i - 1]

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
                lla = stage.lla_vector[:, i - 1]
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
                if i == 1 and np.all(stage.velocity[:, 0] == 0):
                    # if on the launch pad then the thrust is oriented in the 
                    # same direction as the launcher
                    # unless the stage has an initial velocity
                    thrust_vector = self.launcher.orientation_ECEF * thrust_magnitude
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
                    Cd = self.drag_function(stages_to_consider, mach_number)
                A = max([_stage.cross_sectional_area for _stage in stages_to_consider])
                if Cd == 0:
                    # coefficient of drag = 0 corresponds to no drag
                    drag_vector = np.array([0, 0, 0])
                else:
                    velocity = stage.velocity[:, i - 1]
                    velocity_magnitude = np.sqrt(velocity.dot(velocity))
                    if velocity_magnitude != 0:
                        drag_direction = - normalise_vector(stage.velocity[:, i - 1])
                    else:
                        drag_direction = 0
                    drag_vector = 1 / 2 * atmosphere.density * A * Cd * velocity_magnitude ** 2 * drag_direction

                # calculate centrifugal force

                # omega represents rotation vector pointed along Z axis
                omega = np.array([0, 0, (2 * np.pi) / 86400])
                r = stage.position[:, i - 1]
                v_r = stage.velocity[:, i - 1]

                f_centrifugal = np.cross(- mass * omega, np.cross(omega, r))
                f_coriolis = np.cross(-2 * mass * omega, v_r)

                # add up forces acting on the rocket
                sum_forces = drag_vector + gravity_vector + thrust_vector + f_centrifugal + f_coriolis

                # determine acceleration (F=ma)
                acceleration = sum_forces / mass

                dt = stage.time[i] - stage.time[i - 1]

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
                    # copy trajectory data to connected stages
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
                        stages_to_truncate = [stage]

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

                last_progress_print = time()

    def plot_altitude(self):
        for stage in self.stages:
            altitude = stage.get_lla_position_vector()[2, :]
            time = stage.time
            plt.plot(time, altitude, label=stage.name)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Altitude (metres)")

        apogee = np.amax(altitude)
        plt.axhline(y=apogee, color='r', linestyle='-')
        self.log(f"[{self.name}] Apogee is {apogee / 1000}km")
        plt.show()

    def plot_altitude_range(self):
        for stage in self.stages:
            lla_position_vector = stage.get_lla_position_vector()
            altitude = lla_position_vector[2, :]
            
            plt.plot(stage._range / 1000, altitude / 1000, label=stage.name)
        
        # # plot the apogee
        # apogee = np.amax(altitude)
        # plt.xlim(0, apogee)
        # plt.axhline(y=apogee, color='r', linestyle='-')
        #     #self.log(f"[{self.name}] Apogee is {apogee / 1000}km")
        #     # plt.show()

        plt.xlabel("Range (km)")
        plt.ylabel("Altitude (km)")
        plt.show()

    def log(self, string_to_log):
        if self.logging_enabled:
            print(string_to_log)

    def plot_thrust_magnitude(self):
        for stage in self.stages:
            time = stage.time
            thrust_magnitude = np.linalg.norm(stage.thrust_vectors, axis=0)
            plt.plot(time, thrust_magnitude, label=stage.name)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Thrust Magnitude (Newtons)")
        plt.show()

    def plot_fuel_mass(self):
        for stage in self.stages:
            plt.plot(stage.time, stage.fuel_mass_vector, label=stage.name)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Fuel Mass (kg)")
        plt.show()

    def plot_speed(self):
        for stage in self.stages:
            speed = np.linalg.norm(stage.velocity, axis=0)
            plt.plot(stage.time, speed, label=stage.name)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speed (m/s)")
        plt.show()

    def plot_speed_and_altitude(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for stage in self.stages:
            altitude = stage.get_lla_position_vector()[2, :]
            speed = np.linalg.norm(stage.velocity, axis=0)
            ax1.plot(stage.time, speed, "g")
            ax2.plot(stage.time, altitude, "b")

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Speed (m/s)", color="g")
        ax2.set_ylabel("Altitude (m)", color="b")
        plt.grid(False)

        plt.show()

    def plot_all(self):
        self.plot_altitude()
        self.plot_thrust_magnitude()
        self.plot_fuel_mass()
        self.plot_speed()
        self.plot_speed_and_altitude()
        self.plot_altitude_range()

    def save(self, filename):
        with open(filename, "wb") as outp:
            pickle.dump(self, outp, -1)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as inp:
            return pickle.load(inp)
