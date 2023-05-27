"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from math import pi


def exercise_4b_transition(timestep, record=False, save=False, gui=False):
    """[Project 1] 4b Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
    # Use exercise_example.py for reference
    # Additional hints:
    # sim_parameters = SimulationParameters(
    #     ...,
    #     spawn_position=[4, 0, 0.0],
    #     spawn_orientation=[0, 0, np.pi],
    # )
    # _sim, _data = simulation(
    #     sim_parameters=sim_parameters,
    #     arena='amphibious',
    #     fast=True,
    #     record=True,
    #     record_path='walk2swim',  # or swim2walk
    # )

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[-1, 0, 0.0],  # Robot position in [m]
                # note: get into water at x = 0 m
            spawn_orientation=[0, 0, pi/2],  # Orientation in Euler angles [rad]
            drive = 2,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            transition = True,
        )
    ]

    # Grid search
    os.makedirs('./logs/4b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/4b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',
            fast=True,  # For fast mode (not real-time)
            headless=not gui,  # For headless mode (No GUI, could be faster)
            record=record,  # Record video
            record_path="videos/4b/walk2swim", # video saving path
            camera_id=0  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        if save:
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    return


def exercise_4c_transition(timestep, record=False, save=False, gui=False):
    """[Project 1] 4b Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
    # Use exercise_example.py for reference
    # Additional hints:
    # sim_parameters = SimulationParameters(
    #     ...,
    #     spawn_position=[4, 0, 0.0],
    #     spawn_orientation=[0, 0, np.pi],
    # )
    # _sim, _data = simulation(
    #     sim_parameters=sim_parameters,
    #     arena='amphibious',
    #     fast=True,
    #     record=True,
    #     record_path='walk2swim',  # or swim2walk
    # )

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[4, 0.0, 0.2],  # Robot position in [m]
                # note: get into water at x = 0 m
            spawn_orientation=[0, 0, -pi/2],  # Orientation in Euler angles [rad]
            drive = 2,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            transition = True,
        )
    ]

    # Grid search
    os.makedirs('./logs/4c/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/4c/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',
            fast=True,  # For fast mode (not real-time)
            headless=not gui,  # For headless mode (No GUI, could be faster)
            record=record,  # Record video
            record_path="videos/4c/swim2walk", # video saving path
            camera_id=0  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        if save:
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    #exercise_4b_transition(timestep=1e-2, record=False, save=False, gui=True)
    exercise_4c_transition(timestep=1e-2, record=False, save=False, gui=True)
