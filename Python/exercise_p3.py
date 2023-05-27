"""[Project1] Exercise 3: Limb and Spine Coordination while walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from math import pi


def exercise_3a_coordination(timestep, gui, save=False):
    """[Project 1] Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and leg oscillators.

    """
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=40,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive = drive,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            phase_lag_body_limb = phase_lag_body_limb,
        )
        for drive in np.linspace(1, 3, 4)
        for phase_lag_body_limb in np.linspace(-1,1,5)+pi # cf. Salamandra II
    ]

    # Grid search
    os.makedirs('./logs/3a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/3a/simulation_{}.{}'
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


def exercise_3b_coordination(timestep, gui, save=False):
    """[Project 1] Exercise 3b Limb and Spine coordination

    This exercise explores how spine amplitude affects coordination.

    Run the simulations for different walking drives and body amplitude.

    Normally body ampli a fun of drive with fixed param (see paper)
     -> ditch those? just assume independent now? same for exo2?

    """
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=40,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive = drive,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            spine_nominal_amplitude = amplitude,
            ampli_depends_on_drive = False
        )
        for drive in np.linspace(1, 3, 4)
        for amplitude in np.linspace(0, pi/6, 7) # 0°:5°:30° as in Salamandra II for eg (0° as asked in instructions)
    ]

    # Grid search
    os.makedirs('./logs/3b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/3b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water'
            fast=True,  # For fast mode (not real-time)
            headless=not gui,  # For headless mode (No GUI, could be faster)
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
    #exercise_3a_coordination(timestep=1e-2, gui = True, save=False)
    exercise_3b_coordination(timestep=1e-2, gui = True, save=False)
