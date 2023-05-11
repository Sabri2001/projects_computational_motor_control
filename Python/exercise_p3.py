"""[Project1] Exercise 3: Limb and Spine Coordination while walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from math import pi


def exercise_3a_coordination(timestep):
    """[Project 1] Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and leg oscillators.

    """
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 2,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, nominal ampli?, not used now
            phase_lag_body=pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 2,
            # turn=0,  # Another example -> no used now
            # ...
        )
        # for drive in np.linspace(1, 3, 4)
        # for phase_lag_body_limb in np.linspace(-1,1,6)
    ]

    # Grid search
    os.makedirs('./logs/exo3a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exo3a/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        # Uncomment if wanna save
        # data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # # Log simulation parameters
        # with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
        #     pickle.dump(sim_parameters, param_file)
    return


def exercise_3b_coordination(timestep):
    """[Project 1] Exercise 3b Limb and Spine coordination

    This exercise explores how spine amplitude affects coordination.

    Run the simulations for different walking drives and body amplitude.

    Normally body ampli a fun of drive with fixed param (see paper)
     -> ditch those? just assume independent now? same for exo2?

    """
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 2,
            # amplitudes=[1, 2, 3],  # Just an example -> not used yet
            phase_lag_body=pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            # turn=0,  # Another example -> no used now
            # ...
        )
        # for drive in np.linspace(1, 3, 4)
        # for amplitudes in np.linspace(...) ?? -> 0° to 30° as in paper for eg (0° as asked in instructions)?
    ]

    # Grid search
    os.makedirs('./logs/exo3b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exo3b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        # Uncomment if wanna save
        # data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # # Log simulation parameters
        # with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
        #     pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    exercise_3a_coordination(timestep=1e-2)
    # exercise_3b_coordination(timestep=1e-2)
