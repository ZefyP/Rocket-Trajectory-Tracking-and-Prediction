# -*- coding: utf-8 -*-

from typing import Callable, List
from .rocket import Rocket
from multiprocessing import Pool
from .utils import ecef2lla
import matplotlib.pyplot as plt
import pickle
import numpy as np
from .constants import mpl_colours
import os


class MonteCarloSim():
    missile_generator_function: Callable
    results: List[Rocket]
    n_simulations: int
    kml_output_folder: str
    colours = ["royalblue", "lime", "coral", "gold"]
    linewidth = 0.1
    CSV_folder = None

    def __init__(self, missile_generator_function, n_simulations, kml_output_folder=None, kml_downsample=1,
                 CSV_folder=None):
        self.missile_generator_function = missile_generator_function
        self.n_simulations = n_simulations
        self.kml_output_folder = kml_output_folder

        if kml_output_folder is not None:
            if not os.path.isdir(kml_output_folder):
                os.mkdir(kml_output_folder)
        if CSV_folder is not None:
            if not os.path.isdir(CSV_folder):
                os.mkdir(CSV_folder)

        self.kml_downsample = kml_downsample
        self.CSV_folder = CSV_folder

    def _sim(self, n):
        mis = self.missile_generator_function(n)
        mis.run_simulation()

        for stage in mis.stages:
            if self.kml_output_folder is not None:
                stage.export_KML(
                    f"{self.kml_output_folder}/{mis.name}_{stage.name}_{n}.kml",
                    extrude=False, downsample_factor=self.kml_downsample
                )

            if self.CSV_folder is not None:
                stage.export_CSV(
                    f"{self.CSV_folder}/{mis.name}_{stage.name}_{n}.csv"
                )
        return mis

    def run(self, processes=None):
        with Pool(maxtasksperchild=10, processes=processes) as pool:
            results = pool.map_async(self._sim, range(0, self.n_simulations))
            self.results = results.get()
            return self.results

    def save(self, filename):
        with open(filename, "wb") as outp:
            pickle.dump(self, outp, -1)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as inp:
            return pickle.load(inp)

    def plot_all(self):
        self.plot_impact_distribution()
        self.plot_altitude_time()
        self.plot_altitude_range()
        self.plot_speed_time()
        self.plot_speed_altitude()

    def plot_impact_distribution(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        data = {}
        for missile in self.results:
            for stage in missile.stages:
                plot_name = f"{missile.name} {stage.name}"
                final_pos = stage.position[:, -1]
                final_lat, final_long, final_alt = ecef2lla(final_pos[0], final_pos[1], final_pos[2])
                data.setdefault(plot_name, {}).setdefault("latitudes", []).append(final_lat)
                data.setdefault(plot_name, {}).setdefault("longitudes", []).append(final_long)

        i = 0
        for name in data.keys():
            latitudes = data[name]["latitudes"]
            longitudes = data[name]["longitudes"]
            plt.scatter(longitudes, latitudes, label=name, color=mpl_colours[i])
            i += 1

        ax.axis("equal")
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title(f"Impact distribution")
        plt.legend()
        plt.show()

    def plot_altitude_time(self):
        for missile in self.results:
            i = 0
            for stage in missile.stages:
                altitude = stage.get_lla_position_vector()[2, :]
                time = stage.time
                plt.plot(time, altitude / 1000, lw=self.linewidth, color=mpl_colours[i])
                i += 1
        plt.xlabel("Time (seconds)")
        plt.ylabel("Altitude (km)")
        plt.legend([stage.name for stage in self.results[0].stages])
        plt.show()

    def plot_altitude_range(self):
        for missile in self.results:

            i = 0
            for stage in missile.stages:
                lla_position_vector = stage.get_lla_position_vector()
                altitude = lla_position_vector[2, :]
                plt.plot(stage._range / 1000, altitude / 1000, lw=self.linewidth, color=mpl_colours[i])
                i += 1
        plt.xlabel("Range (km)")
        plt.ylabel("Altitude (km)")
        plt.legend([stage.name for stage in self.results[0].stages])
        plt.show()

    def plot_speed_time(self):
        for missile in self.results:
            i = 0
            for stage in missile.stages:
                speed = np.linalg.norm(stage.velocity, axis=0)
                plt.plot(stage.time, speed, lw=self.linewidth, color=mpl_colours[i])
                i += 1
        plt.legend([stage.name for stage in self.results[0].stages])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Velocity magnitude (m/s)")
        plt.show()

    def plot_speed_altitude(self):
        for missile in self.results:
            i = 0
            for stage in missile.stages:
                altitude = stage.get_lla_position_vector()[2, :]
                apogee = np.amax(altitude)
                apogee_index = int(np.where(altitude == apogee)[0])
                altitude = altitude[apogee_index::]
                speed = np.linalg.norm(stage.velocity, axis=0)
                speed = speed[apogee_index::]
                plt.plot(speed, altitude, lw=self.linewidth, color=mpl_colours[i])
                i += 1
        plt.xlabel("Velocity magnitude (m/s)")
        plt.ylabel("Altitude (metres)")
        plt.title("Descent phase velocity (post-apogee)")
        plt.legend([stage.name for stage in self.results[0].stages])
        plt.show()
