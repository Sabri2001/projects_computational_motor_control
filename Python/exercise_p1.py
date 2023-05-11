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

    # # Plotting the nominal amplitudes: that's good
    # plt.figure()
    # plt.plot(times, amplitude_rates_log[:, 0], label='Body nominal amplitude')
    # plt.plot(times, amplitude_rates_log[:, 16], label='Limb nominal amplitude')
    # plt.xlabel('Times')
    # plt.ylabel('Nominal amplitude')
    # plt.legend()

    # # Plotting the motor position outputs
    # plt.figure()
    # plt.plot(times, outputs_log[:, 0], label='Spine motor')
    # plt.plot(times, outputs_log[:, 8], label='Limb shoulder motor')
    # plt.xlabel('Time')
    # plt.ylabel('Motor Position')
    # plt.legend()

    # Plot x
    plt.figure()
    plt.plot(times, outputs_log[:, 0], label='Spine')
    plt.plot(times, outputs_log[:, 8], label='Limb')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.legend()

    # # Plotting the frequencies
    # plt.figure()
    # plt.plot(times, freqs_log[:, 0], label='Body frequency')
    # plt.plot(times, freqs_log[:, 16], label='Limb frequency')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.legend()

    # Plot for visualising walking/swimming patterns
    # => artificially offset curves + only show spine motors
    # plt.figure()
    # # 8 spine command plots
    # plt.plot(times, outputs_log[:, 0],'b',label='Motor0')
    # plt.plot(times, outputs_log[:, 1]-2,'b', label='Motor1')
    # plt.plot(times, outputs_log[:, 2]-4,'b', label='Motor2')
    # plt.plot(times, outputs_log[:, 3]-6,'b', label='Motor3')
    # plt.plot(times, outputs_log[:, 4]-8,'g', label='Motor4')
    # plt.plot(times, outputs_log[:, 5]-10,'g', label='Motor5')
    # plt.plot(times, outputs_log[:, 6]-12,'g', label='Motor6')
    # plt.plot(times, outputs_log[:, 7]-14,'g', label='Motor7')
    # # red lines for gait visualisation
    # plt.plot(np.array([13.79,13.99,14.88,15.09]),np.array([0.74,-5.33,-7.26,-13.28]),'r')
    # plt.plot(np.array([24.37,25.23]),np.array([0.87,-13.13]),'r')
    # plt.xlabel('Time')
    # plt.ylabel('Motor Position')
    # plt.legend()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=10, update=True)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots    ())
