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
    pylog.warning(
        'DONE: Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    drive = [6*i/(n_iterations-1) for i in range(n_iterations)]
    state = SalamandraState.salamandra_robot(n_iterations) 
        # initialise empty array (or rather, SalamandraState -> cf. salamndar_simulation.data.py object) 
        # in which history of states will be written
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    osc_left = np.arange(8) # TODO: what are those for??
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)

    # Logs: initialisation
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)

    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)

    amplitude_rates_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ]) # my addition, to check ampli_rates
    
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

    # comment below pass to run file
    # pylog.warning('Remove the pass to run your code!!')
    # pass

    pylog.warning(
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
    pylog.warning('DONE: Implement plots')
    # Plotting the phases
    # plt.figure()
    # plt.plot(times, phases_log[:, 0], label='Body phase')
    # plt.plot(times, phases_log[:, 16], label='Limb phase')
    # plt.xlabel('Time')
    # plt.ylabel('Phase')
    # plt.legend()

    # Plotting the amplitudes
    # plt.figure()
    # plt.plot(times, amplitudes_log[:, 0], label='Body amplitude')
    # plt.plot(times, amplitudes_log[:, 16], label='Limb amplitude')
    # plt.xlabel('Times')
    # plt.ylabel('Amplitude')
    # plt.legend()

    # # Plotting the nominal amplitudes
    plt.figure(figsize=(8,4))
    plt.plot(drive, amplitude_rates_log[:, 0], label='Body', color='black')
    plt.plot(drive, amplitude_rates_log[:, 16], label='Limb', color='black', linestyle='dashed')
    plt.xlabel('drive')
    plt.ylabel('R')
    plt.legend()
    plt.tight_layout()

    # # Plotting the frequencies
    plt.figure(figsize=(8,4))
    plt.plot(drive, freqs_log[:, 0], label='Body', color='black')
    plt.plot(drive, freqs_log[:, 16], label='Limb', color='black', linestyle='dashed')
    plt.xlabel('drive')
    plt.ylabel(r"$\nu$ [Hz]")
    plt.legend()
    plt.tight_layout()

    # Plotting oscillator outputs
    # Note: artificially offset curves for visualising walking/swimming patterns
    # Spine oscillators 0 to 7 (left side, head to tail)
    # plt.figure()
    # plt.plot(times, outputs[:, 0],'b',label='Osc_output0')
    # plt.plot(times, outputs[:, 1]-2,'b', label='Osc_output1')
    # plt.plot(times, outputs[:, 2]-4,'b', label='Osc_output2')
    # plt.plot(times, outputs[:, 3]-6,'b', label='Osc_output3')
    # plt.plot(times, outputs[:, 4]-8,'g', label='Osc_output4')
    # plt.plot(times, outputs[:, 5]-10,'g', label='Osc_output5')
    # plt.plot(times, outputs[:, 6]-12,'g', label='Osc_output6')
    # plt.plot(times, outputs[:, 7]-14,'g', label='Osc_output7')

    # # Front limbs (left then right)
    # plt.plot(times, outputs[:, 16]-18,'b',label='Osc_output16')
    # plt.plot(times, outputs[:, 18]-20,'g', label='Osc_output18')

    # TODO: adjust red/dashed lines for visualisation

    # plt.xlabel('Time')
    # plt.ylabel('X')
    # plt.legend()

    plot_results.plot_oscillator_patterns(times, outputs, drive) # TODO plot frequencies
    plot_results.plot_properties(times, outputs, drive, freqs_log, amplitudes_log)

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
    exercise_1a_networks(plot=not save_plots    ())