# -*- coding: utf-8 -*-

from .rocket import Rocket
from typing import Callable
import numpy as np

# abstract base classes
from abc import ABC, abstractmethod

from multiprocessing import Queue, Pool, Process
import os
import matplotlib.pyplot as plt
import queue
from .utils import lla2ecef


class Target(ABC):
    """
    Base class for defining optimisation 'targets'
    """
    error: float
    min_error: float = None
    error_threshold: float

    def __init__(self, error_threshold):
        self.error_threshold = error_threshold

    @abstractmethod
    def calculate_error(self, missile: Rocket) -> float:
        pass

    def target_met(self) -> bool:
        return self.min_error is not None and \
               self.min_error < self.error_threshold

    def is_best(self, missile):
        self.error = self.calculate_error(missile)
        if self.min_error is None:
            self.min_error = self.error
            return True
        if self.error < self.min_error:
            self.min_error = self.error
            return True
        else:
            return False


class ApogeeRangeTarget(Target):
    target_apogee: float
    target_range: float

    def __init__(self, target_apogee, target_range, error_threshold):
        Target.__init__(self, error_threshold)
        self.target_apogee = target_apogee
        self.target_range = target_range

    def calculate_error(self, missile: Rocket):
        altitude = missile.stages[-1].lla_vector[2, :]
        apogee = np.amax(altitude)
        _range = missile.stages[-1]._range[-1]
        apogee_error = self.target_apogee - apogee
        range_error = self.target_range - _range

        error = np.sqrt(apogee_error ** 2 + range_error ** 2)
        return error


class LocationTarget(Target):
    target_position: np.ndarray

    def __init__(self, target_latitude, target_longitude, error_threshold):
        Target.__init__(self, error_threshold)
        self.target_position = lla2ecef(target_latitude, target_longitude, 0)

    def calculate_error(self, missile: Rocket):
        landing_position = missile.stages[-1].position[:, -1]
        position_error = self.target_position - landing_position
        error = np.linalg.norm(position_error)
        return error


class Optimiser:
    missile_generator: Callable[[float, Rocket], Rocket]
    target: Target

    plot_queue = Queue(10)
    best_missile: Rocket = None
    std_multiplier = 1
    missiles_since_last_solution = 0
    kml_output_folder: str

    def __init__(
            self,
            missile_generator: Callable[[float], Rocket],
            target: Target,
            kml_output_folder=None
    ):
        self.missile_generator = missile_generator
        self.target = target
        self.kml_output_folder = kml_output_folder

        if kml_output_folder is not None:
            if not os.path.isdir(kml_output_folder):
                os.mkdir(kml_output_folder)

    @staticmethod
    def create_plot(q):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, (1, 2))
        ax.set_xlabel("Range (km)")
        ax.set_ylabel("Altitude (km)")

        ax2 = fig.add_subplot(133)
        i = 0
        ax2.set_xlabel("Iteration number")
        ax2.set_ylabel("Error score")

        best_lines = []
        while True:
            try:
                missile, min_error = q.get_nowait()
            except queue.Empty:
                plt.pause(1)
                continue
            altitude = missile.stages[-1].lla_vector[2, :] / 1000
            _range = missile.stages[-1]._range / 1000
            if missile.is_best:
                for line in best_lines:
                    line.set(color="lightgray")
                    best_lines.remove(line)
            line, = ax.plot(_range, altitude,
                            "-" if missile.is_best else "--",
                            color="red" if missile.is_best else "lightgray"
                            )
            i += 1
            if missile.is_best:
                best_lines.append(line)
                ax2.scatter(i, min_error)
            if len(ax.lines) > 32:
                ax.lines.pop(0)
            fig.canvas.draw()
            fig.canvas.flush_events()

    @staticmethod
    def run_simulation(missile):
        missile.run_simulation()
        return missile

    def run(self, processes=os.cpu_count() - 1):
        plotter = Process(target=self.create_plot, args=(self.plot_queue,))
        plotter.daemon = True
        plotter.start()

        pool = Pool(processes=processes)
        n = 0
        while not self.target.target_met():
            missiles = [
                self.missile_generator(self.std_multiplier, self.best_missile) \
                for x in range(0, processes)
            ]
            results = pool.map_async(self.run_simulation, missiles)
            print(f"Started {len(missiles)} simulations")
            missiles = results.get()
            print("Simulation batch complete")

            for missile in missiles:
                if self.target.is_best(missile):
                    missile.is_best = True
                    self.best_missile = missile
                    self.missiles_since_last_solution = 0

                else:
                    missile.is_best = False
                    self.missiles_since_last_solution += 1

                self.plot_queue.put((missile, self.target.error))

            if self.kml_output_folder is not None:
                for stage in self.best_missile.stages:
                    stage.export_KML(f"{self.kml_output_folder}/{self.best_missile.name}_{stage.name}{n}.kml",
                                     extrude=False)
                n += 1

            if self.missiles_since_last_solution > 20:
                self.std_multiplier *= 0.5
                print(f"\n\n\n!!!!!!\nNew STD multiplier: {self.std_multiplier}")

            print(f"Error: {self.target.min_error}")
            self.best_missile.save(f"{self.kml_output_folder}/best_missile.mis")
        pool.close()
        pool.join()

        self.best_missile.plot_altitude_range()
