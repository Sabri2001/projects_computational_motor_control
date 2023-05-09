"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def exercise_2a_swim(timestep):
    """[Project 1] Exercise 2a Swimming

    In this exercise we need to implement swimming for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different swimming drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference

    # PERSONAL NOTES
    # phase lag: now 2pi/8 => 8 oscillators form complete S-shape, check in papers whether go for fewer or more
    # drive: 3 to 5
    # check speed/torque + energy? they ask for quantity representing both speed/power at same time, eg ratio?
    # check lecture notes/papers?
    pass
    return


def exercise_2b_walk(timestep):
    """[Project 1] Exercise 2a Walking

    In this exercise we need to implement walking for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different walking drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference

    # PERSONAL NOTES
    # phase lag: now 2pi/8 => 8 oscillators form complete S-shape, check in papers whether go for fewer or more
    # Q: not supposed to try changing with drive are we? cf. supp material
    # drive: 1 to 3
    pass
    return


def exercise_test_walk(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference

    # PERSONAL NOTES:
    # Q: just describe qualitatively? or also grid search?
    # Q: what's the diff between this function (+ below) with those above?

    pass
    return


def exercise_test_swim(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_2a_swim(timestep=1e-2)
    exercise_2b_walk(timestep=1e-2)

