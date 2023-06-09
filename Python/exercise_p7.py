"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from math import pi


def exercise_7_transition(timestep, record=False, save=False, gui=False):
    # Parameters
    parameter_set = [
        # Water to land
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[4, 0.0, 0.2],  # Robot position in [m]
                # note: get into water at x = 0 m
            spawn_orientation=[0, 0, -pi/2],  # Orientation in Euler angles [rad]
            drive = 2,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            force_transition=True, # transition using ground reaction forces
        ),
        # Land to water
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[-1, 0.0, 0.0],  # Robot position in [m]
                # note: get into water at x = 0 m
            spawn_orientation=[0, 0, pi/2],  # Orientation in Euler angles [rad]
            drive = 2,
            phase_lag_body=2*pi/8,  # or np.zeros(n_joints) for example
            force_transition=True, # transition using ground reaction forces
        )
    ]

    # Grid search
    os.makedirs('./logs/7/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/7/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',
            fast=True,  # For fast mode (not real-time)
            headless=not gui,  # For headless mode (No GUI, could be faster)
            record=record,  # Record video
            record_path="videos/7/swim2walk", # video saving path
            camera_id=0  # camera type: 0=top view, 1=front view, 2=side view,
        )
        data.sensors.contacts.array[:,0:4,2]
        # Log robot data
        if save:
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    exercise_7_transition(timestep=1e-2, record=False, save=True, gui=True)
