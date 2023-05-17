"""Plot results"""

import pickle
import numpy as np
from requests import head
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import motor_output
import matplotlib.colors as colors
from math import pi

def plot_oscillator_patterns(times, outputs, drive, vlines=[0,20,40], walk_timesteps=[16, 17], swim_timesteps=[26, 27]):
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, height_ratios=[2, 0.5, 1, 1])

    # Body oscillators
    axes[0].set_ylabel('x Body')
    # Spine oscillators 0 to 7 (left side, head to tail)
    for i in range(8):
        color = 'b' if i < 4 else 'g' # Blue for trunk, green for tail
        # Assign single label for body and limb
        label = None
        if i == 0:
            label = 'Trunk'
        elif i == 4:
            label = 'Tail'
        axes[0].plot(times, outputs[:, i]-i*2, color, label=label)
        axes[0].text(-1.5, outputs[0, i]-i*2, f'$x_{i + 1}$', fontsize='small')
    # Remove the y-ticks
    axes[0].yaxis.set_tick_params(labelleft=False)
    axes[0].set_yticks([])
    # Add trunk and tail labels
    axes[0].legend()

    # Plot the red line for walking
    xs = [walk_timesteps[0], walk_timesteps[0], walk_timesteps[1], walk_timesteps[1]]
    ys = [outputs[walk_timesteps[0], 0], outputs[walk_timesteps[0], 3]-3*2, outputs[walk_timesteps[1], 4]-4*2, outputs[walk_timesteps[1], 7]-7*2]
    axes[0].plot(xs, ys, 'r')
    # Plot the red line for swimming
    ys = [outputs[swim_timesteps[0], 0], outputs[walk_timesteps[1], 7]-7*2]
    axes[0].plot(swim_timesteps, ys, 'r')

    # Limb oscillators
    axes[1].set_ylabel('x Limb')
    # Plot both oscillators
    axes[1].plot(times, outputs[:, 16], 'b')
    axes[1].text(-1.5, outputs[0, 16], "$x_{17}$", fontsize='small')
    axes[1].plot(times, outputs[:, 18]-2, 'g')
    axes[1].text(-1.5, outputs[0, 18]-2, "$x_{19}$", fontsize='small')
    # Remove the y-ticks
    axes[1].yaxis.set_tick_params(labelleft=False)
    axes[1].set_yticks([])
    # Equal axis size as previous
    axes[1].axis('equal')

    # Frequency ??
    axes[2].set_ylabel('Freq [Hz]')
    axes[2].text(0,0.9,'todo')

    # Drive
    axes[3].set_ylabel('drive d')
    axes[3].plot(times, drive, 'black')

    # Label time axes
    axes[-1].set_xlabel('Time [s]')
    # Add gray dashed lines to all plots
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.vlines(vlines, ymin, ymax, 'gray', 'dashed', alpha=0.2)

    # Correct the layout
    plt.tight_layout()

def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data, label=None, color=None):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot (interpolation)

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    plt.show()


def max_distance(link_data, nsteps_considered=None):
    """Compute max distance"""
    if not nsteps_considered:
        nsteps_considered = link_data.shape[0]
    com = np.mean(link_data[-nsteps_considered:], axis=1)

    # return link_data[-1, :]-link_data[0, 0]
    return np.sqrt(
        np.max(np.sum((link_data[:, :]-link_data[0, :])**2, axis=1)))


def compute_speed(links_positions, links_vel, nsteps_considered=200):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    links_pos_xy = links_positions[-nsteps_considered:, :, :2]
    joints_vel_xy = links_vel[-nsteps_considered:, :, :2]
    time_idx = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []
    com_pos = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :9, 0]
        y = links_pos_xy[idx, :9, 1]

        pheadtail = links_pos_xy[idx][0]-links_pos_xy[idx][8]  # head - tail
        pcom_xy = np.mean(links_pos_xy[idx, :9, :], axis=0)
        vcom_xy = np.mean(joints_vel_xy[idx], axis=0)

        covmat = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:, largest_index]

        ht_direction = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        com_pos.append(pcom_xy)
        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return np.mean(speed_forward), np.mean(speed_lateral)


def sum_torques(joints_data):
    """Compute sum of torques"""
    return np.sum(np.abs(joints_data[:, :]))


def cost_of_transport(joints_torques, joints_velocities):
    """Compute mean """
    instant_powers = joints_torques*joints_velocities
    


def main(files, plot=True):
    """Main"""

    speed_vec  = []
    for file_name in files:
        # Load data
        data = SalamandraData.from_file(file_name+'.h5')
        with open(file_name+'.pickle', 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        # timestep = times[1] - times[0]
        # amplitudes = parameters.amplitudes
        # phase_lag_body = parameters.phase_lag_body
        # osc_phases = data.state.phases()
        # osc_amplitudes = data.state.amplitudes()
        links_positions = np.array(data.sensors.links.urdf_positions())
        links_vel = np.array(data.sensors.links.com_lin_velocities())
        # head_positions = links_positions[:, 0, :]
        # tail_positions = links_positions[:, 8, :]
        # joints_positions = data.sensors.joints.positions_all() # check oscillation or ramp (for power)!
        # joints_velocities = data.sensors.joints.velocities_all()
        # joints_torques = data.sensors.joints.motor_torques_all()

        # Metrics (scalar)
        # Note: use dir() to know metrics than can be applied to object
        # print("Total torque: ", sum_torques(joints_torques))
        # speed_vec.append(compute_speed(links_positions, links_vel)[0]) # only axial speed here

    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    # head_positions = np.asarray(head_positions)
    # plt.figure('Positions')
    # plot_positions(times, head_positions)
    # plt.figure('Trajectory')
    # plot_trajectory(head_positions)

    # 2D plot for grid search (NOTE: update parameter ranges+labels)
    # param_range1 = np.linspace(3,5,4)
    # param_range2 = [pi/8, 2*pi/8, 3*pi/8]
    # results = np.array([[i,j,0] for i in param_range1 for j in param_range2])
    # results[:,2] = np.array(speed_vec)
    # print(results)
    # plot_2d(results,["Drive [-]","Wave number k [-]","Mean speed [m/s]"]) # param1, param2, metric

    # Show plots
    # if plot:
    #     plt.show()
    # else:
    #     save_figures()


def test_2D():
    """Test 2D grid search plot"""
    results = np.array([[i,j*pi/8,0] for i in np.linspace(3,5,4) for j in range(1,4)])
    results[:,2] = results[:,0] + results[:,1]
    print(results)
    plot_2d(results,["x","y","z"])


def test_cost_of_transport():
    """Test 2D grid search plot"""
    

if __name__ == '__main__':
    # main(plot=save_plots()) -> that's for saving plots
    file_names = [f'./logs/exo2a/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/exo2b/simulation_{i}' for i in range(12)]
    main(files=file_names, plot=False)
