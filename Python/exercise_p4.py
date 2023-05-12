"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def exercise_4a_transition(timestep):
    """[Project 1] 4a Transitions

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
            duration=60,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[4, 0, 0.0],  # Robot position in [m]
            spawn_orientation=[0, 0, np.pi],  # Orientation in Euler angles [rad]
            # drive=drive,  # An example of parameter part of the grid search
            drive = 2,
            # amplitudes=[1, 2, 3],  # Just an example -> don't know what stands for, not used now
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = 0.0,
            # turn=0,  # Another example -> no used now
            # ...
        )
        # for drive in np.linspace(3, 4, 2)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    exercise_4a_transition(timestep=1e-2)

