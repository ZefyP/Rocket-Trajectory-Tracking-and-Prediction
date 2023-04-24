# -*- coding: utf-8 -*-

from .stage import Stage
from typing import List


def V2_rocket_drag_function(stages: List[Stage], mach: float) -> float:
    # Drag function for V2
    # derived from Sutton, "Rocket Propulsion Elements", 7th ed, p108

    drag_coefficient: float = 0

    if mach > 5:
        drag_coefficient = 0.15
    elif mach > 1.8 and mach <= 5:
        drag_coefficient = -0.03125 * mach + 0.30625
    elif mach > 1.2 and mach <= 1.8:
        drag_coefficient = -0.25 * mach + 0.7
    elif mach > 0.8 and mach <= 1.2:
        drag_coefficient = 0.625 * mach - 0.35
    elif mach <= 0.8:
        drag_coefficient = 0.15

    return drag_coefficient
