# -*- coding: utf-8 -*-

from typing import List
from .rocket import Rocket
import numpy as np
import matplotlib.pyplot as plt
from .utils import ecef2aer, lla2ecef
from .constants import mpl_colours


class Observer:
    name: str
    missiles: List[Rocket]
    position: np.ndarray

    def __init__(self, name, missiles: List[Rocket],
                 latitude, longitude, altitude):
        self.name = name
        self.missiles = missiles
        self.position = lla2ecef(latitude, longitude, altitude)

    def plot_target_aer(self):
        """
        Plots azmiuth, elevation and range of targets
        """

        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312, sharex=ax1)
        ax3 = plt.subplot(313, sharex=ax1)

        linewidth = 1 if len(self.missiles) < 10 else 0.05
        for missile in self.missiles:
            i = 0
            for stage in missile.stages:
                azimuth = []
                elevation = []
                _range = []
                for x in range(0, stage.position.shape[1]):
                    az, el, r = ecef2aer(self.position, stage.position[:, x])
                    azimuth.append(az)
                    elevation.append(el)
                    _range.append(r)

                ax1.plot(stage.time, azimuth, color=mpl_colours[i], lw=linewidth)
                ax2.plot(stage.time, elevation, color=mpl_colours[i], lw=linewidth)
                ax3.plot(stage.time, _range, color=mpl_colours[i], lw=linewidth)
                i += 1
        ax3.set_xlabel("Time (seconds)")

        ax1.set_ylabel("Azimuth (deg)")
        ax2.set_ylabel("Elevation (deg)")
        ax3.set_ylabel("Range (metres)")

        ax1.set_title(f"{self.name} - Azimuth, Elevation and Range plot")

        plt.show()
