"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from math import pi
import argparse


def exercise_2a_swim(timestep, gui, save = False):
    """[Project 1] Exercise 2a Swimming

    In this exercise we need to implement swimming for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different swimming drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference

    # PERSONAL NOTES
    # phase lag: now 2pi/8 => 8 oscillators form complete S-shape, according to salamander k: 0.5 -> 1.5
    # drive: 3 to 5 => freq 0.9 to 1.3 and R 0.391 to 0.521
    # check speed/torque 
    # -> TODO: power/speed -> similar to cost of transport (energy/distance)

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            phase_lag_body=phase_lag_body,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
        )
        for drive in np.linspace(3, 5, 4)
        for phase_lag_body in [pi/8,2*pi/8,3*pi/8]
    ]

    # Grid search
    os.makedirs('./logs/exo2a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exo2a/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless= not gui,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        if save:
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    return


def exercise_2b_walk(timestep, gui, save=False):
    """[Project 1] Exercise 2a Walking

    In this exercise we need to implement walking for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different walking drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference

    # PERSONAL NOTES
    # drive: 1 to 3

    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive = drive,
            phase_lag_body=phase_lag_body,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
        )
        for drive in np.linspace(1, 3, 4)
        for phase_lag_body in [pi/8,2*pi/8,3*pi/8]
    ]

    # Grid search
    os.makedirs('./logs/exo2b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exo2b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless= not gui,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        if save:
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    """Parse args"""
    parser = argparse.ArgumentParser(description=(
        'CMC lab'
    ))
    parser.add_argument(
        '--save', '--save-simulations', '-s',
        help='Save simulations data',
        dest='save',
        action='store_true'
    )
    args, _ = parser.parse_known_args()
    exercise_2a_swim(timestep=1e-2, gui=False, save=args.save)
    exercise_2b_walk(timestep=1e-2, gui=False, save=args.save)
