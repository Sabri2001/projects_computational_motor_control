"""[Project1] Exercise 5: Turning while Swimming & Walking, Backward Swimming & Walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from math import pi


def exercise_5a_swim_turn(timestep):
    """[Project1] Exercise 5a: Turning while swimming"""

    # Use exercise_example.py for reference
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 4,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, not used now
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            turn=2,  # Another example
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        # data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # # Log simulation parameters
        # with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
        #     pickle.dump(sim_parameters, param_file)
    return


def exercise_5b_swim_back(timestep):
    """[Project1] Exercise 5b: Backward Swimming"""
    # Use exercise_example.py for reference
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 4,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, not used now
            phase_lag_body=-2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            turn=0,  # Another example
        )
    ]

    # Grid search
    # os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
    #     data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
    #     # Log simulation parameters
    #     with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
    #         pickle.dump(sim_parameters, param_file)
    return


def exercise_5c_walk_turn(timestep):
    """[Project1] Exercise 5c: Turning while Walking"""

    # Use exercise_example.py for reference
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=60,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 2,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, not used now
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            turn=0,  # Another example
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        # data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # # Log simulation parameters
        # with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
        #     pickle.dump(sim_parameters, param_file)
    return


def exercise_5d_walk_back(timestep):
    """[Project1] Exercise 5d: Backward Walking"""
    # Use exercise_example.py for reference
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=60,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 2,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, not used now
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            turn=0,  # Another example
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        # data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # # Log simulation parameters
        # with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
        #     pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    # exercise_5a_swim_turn(timestep=1e-2)
    # exercise_5b_swim_back(timestep=1e-2)
    exercise_5c_walk_turn(timestep=1e-2)
    # exercise_5d_walk_back(timestep=1e-2)
