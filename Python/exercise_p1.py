"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from math import pi
import plot_results


def run_network(duration, update=False, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag_body=2*pi/8,
        phase_lag_body_limb = 0.0,
        turn=None,
    )
    pylog.info(
        'DONE: Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    drive = [6*i/(n_iterations-1) for i in range(n_iterations)]
    state = SalamandraState.salamandra_robot(n_iterations) 
        # initialise empty array (or rather, SalamandraState -> cf. salamndra_simulation.data.py object) 
        # in which history of states will be written
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    # osc_left = np.arange(8) -> not used
    # osc_right = np.arange(8, 16)
    # osc_legs = np.arange(16, 20)

    # Logs: initialisation
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)

    # derivative phase
    dphases_log = np.zeros([
        n_iterations,
        20
    ])
    dphases_log[0, :] = network.get_dphase(iteration=0)

    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)

    amplitude_rates_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs

    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)

    outputs = np.zeros([
        n_iterations,
        len(network.outputs(iteration=0))
    ])
    outputs[0, :] = network.outputs(iteration=0)


    pylog.info(
        'DONE: Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive = drive[i]
                )
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        dphases_log[i+1, :] = network.get_dphase(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        amplitude_rates_log[i+1, :] = network.robot_parameters.nominal_amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
        outputs[i+1, :] = network.outputs(iteration=i+1)
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    pylog.info('DONE: Implement plots')

    # Plotting oscillator patterns
    # Note: artificially offset curves for visualising walking/swimming patterns
    plot_results.plot_oscillator_patterns(times, outputs, drive, dphases_log)

    # Plotting oscillator properties
    plot_results.plot_oscillator_properties(times, outputs, drive, freqs_log, amplitudes_log)

    # Plot radius and frequency against drive
    plot_results.plot_drive_effects(drive, dphases_log, amplitude_rates_log)

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=40, update=True)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())
