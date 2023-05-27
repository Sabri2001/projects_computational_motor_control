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
from math import pi,cos

def plot_oscillator_patterns(times, outputs, drive, freqs, vlines=[0,20,40], walk_timesteps=[16, 17], swim_timesteps=[23, 24]):
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
    xs = [walk_timesteps[0]+1.07, walk_timesteps[0]+1.2, walk_timesteps[1]+1, walk_timesteps[1]+1.12]
    ys = [0.75+outputs[walk_timesteps[0], 0], 0.75+outputs[walk_timesteps[0], 3]-3*2, 0.75+outputs[walk_timesteps[1], 4]-4*2, 0.75+outputs[walk_timesteps[1], 7]-7*2]
    axes[0].plot(xs, ys, 'r')
    # Plot the red line for swimming
    ys = [0.8+outputs[swim_timesteps[0], 0], 0.8+outputs[walk_timesteps[1], 7]-7*2]
    axes[0].plot([swim_timesteps[0]+0.4, swim_timesteps[1]+0.25], ys, 'r')

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

    # Frequency
    axes[2].set_ylabel('$\\frac{\\dot{\\theta}}{2\\pi}$ [Hz]')
    axes[2].plot(times, freqs[:, 0], color='black', label='spine')
    axes[2].plot(times, freqs[:, 16], color='black', linestyle='dashed', label='limb')
    axes[2].legend()

    # Drive
    axes[3].set_ylabel('drive d')
    axes[3].plot(times, drive, 'black',)
    axes[3].axhline(y=6, xmin=0, xmax=3, color="orange", linewidth=1, zorder=0)
    axes[3].axhline(y=3, xmin=0, xmax=3, color="orange", linewidth=1, zorder=0)
    axes[3].axhline(y=0, xmin=0, xmax=3, color="orange", linewidth=1, zorder=0)
    axes[3].text(0.25, 1.5, 'Walking', fontsize=14)
    axes[3].text(20.25, 4.15, 'Swimming', fontsize=14)

    
    # Label time axes
    axes[-1].set_xlabel('Time [s]')
    # Add gray dashed lines to all plots
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.vlines(vlines, ymin, ymax, 'gray', 'dashed', alpha=0.2)

    # Correct the layout
    plt.tight_layout()

def plot_oscillator_properties(times, outputs, drive, freqs_log, amplitudes_log, vlines=[0,20,40], walk_timesteps=[16, 17], swim_timesteps=[26, 27]):
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, height_ratios=[1, 1, 1, 1])

    # Body oscillators
    axes[0].set_ylabel('$x$')
    # Plot spine and limb oscillations
    axes[0].plot(times, outputs[:, 0], color='black')
    axes[0].plot(times, outputs[:, 16]-2,color='black', linestyle='dashed')
    # Remove the y-ticks
    axes[0].yaxis.set_tick_params(labelleft=False)
    axes[0].set_yticks([])

    # Frequencies
    axes[1].set_ylabel('$\\frac{\\dot{\\theta}}{2\\pi}$ [Hz]')
    # Plot both oscillators
    axes[1].plot(times, freqs_log[:, 0], color='black')
    axes[1].plot(times, freqs_log[:, 16], color='black', linestyle='dashed')
    
    axes[2].set_ylabel('r')
    # Amplitudes
    axes[2].plot(times, amplitudes_log[:, 0],  color='black')
    axes[2].plot(times, amplitudes_log[:, 16], color='black', linestyle='dashed')

    # Drive
    axes[3].set_ylabel('drive d')
    axes[3].plot(times, drive, 'black')
    axes[3].axhline(y=6, xmin=0, xmax=3, color="black", linestyle='dotted', linewidth=1, zorder=0)
    axes[3].axhline(y=3, xmin=0, xmax=3, color="black", linestyle='dotted', linewidth=1, zorder=0)
    axes[3].axhline(y=0, xmin=0, xmax=3, color="black", linestyle='dotted', linewidth=1, zorder=0)
    
    # Label time axes
    axes[-1].set_xlabel('Time [s]')
    # Add gray dashed lines to all plots
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.vlines(vlines, ymin, ymax, 'gray', 'dashed', alpha=0.2)

    # Correct the layout
    plt.tight_layout()

def plot_drive_effects(drive, freqs, amplitudes):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Frequencies
    axes[0].set_ylabel('$\\nu$ [Hz]')
    axes[0].plot(drive, freqs[:, 0], color='black', label='spine')
    axes[0].plot(drive, freqs[:, 16], color='black', linestyle='dashed', label='limb')
    # Add spine and limb labels
    axes[0].legend()
    
    axes[1].set_ylabel('R')
    # Amplitudes
    axes[1].plot(drive, amplitudes[:, 0],  color='black')
    axes[1].plot(drive, amplitudes[:, 16], color='black', linestyle='dashed')
    
    # Label drive axes
    axes[-1].set_xlabel('drive')

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


def mean_power(joints_torques, joints_velocities):
    """Mean instantaneous power"""
    return np.mean(joints_torques*joints_velocities)


def power_speed_ratio(joints_torques, joints_velocities, links_positions, links_vel):
    """Compute mean power/mean speed"""
    power = mean_power(joints_torques, joints_velocities)
    speed = np.linalg.norm(np.array(compute_speed(links_positions, links_vel))) # speed norm
    return power/speed


def wrap_to_pi(angle):
    """
    Returns equivalent angle in range -pi to pi
    """
    while angle > 3*pi:
        angle -= 2*pi

    while angle <= -pi:
        angle += 2*pi
    return angle


def wrap_to_2pi(angle):
    """
    Returns equivalent angle in range 0 to 2*pi
    """
    while angle >= 2*pi:
        angle -= 2*pi

    while angle < 0:
        angle += 2*pi
    return angle


vector_wrap_to_2pi = np.vectorize(wrap_to_2pi)


def spine_cmd(phase,ampli):
    return ampli*(1+cos(phase))


def limb_cmd(phase,ampli): 
    return ampli*cos(phase)-1 # -1 offset for plotting


def plot_phase_force_vs_time(times, phases, ground_forces):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Frequencies
    axes[0].set_ylabel('Phase [rad]')
    axes[0].plot(times, phases, label=['limb 1', 'limb 2', 'limb 3', 'limb 4'])
    # Add spine and limb labels
    axes[0].legend()
    
    axes[1].set_ylabel('Force [N]')
    # Amplitudes
    axes[1].plot(times, ground_forces, label=['limb 1', 'limb 2', 'limb 3', 'limb 4'])
    
    # Label drive axes
    axes[-1].set_xlabel('Time [s]')

    # Correct the layout
    plt.tight_layout()


def plot_phase_vs_force(phases, ground_forces):
    plt.figure()
    # Frequencies
    plt.ylabel('Phase [rad]')
    plt.plot(ground_forces, phases, label=['limb 1', 'limb 2', 'limb 3', 'limb 4'])
    # Add spine and limb labels
    plt.legend()
    
    # Label drive axes
    plt.xlabel('Grf [N]')

    plt.show()


def threshold_grf(ground_forces):
    return np.clip(ground_forces,a_min=7,a_max=None)


def main(files, plot=True):
    """Main"""

    speed_vec  = []
    torque_vec = []
    power_speed_vec = []   

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
        timestep = times[1] - times[0]
        # amplitudes = parameters.amplitudes
        # phase_lag_body = parameters.phase_lag_body
        osc_phases = np.array(data.state.phases())
        osc_amplitudes = data.state.amplitudes()
        links_positions = np.array(data.sensors.links.urdf_positions()) # shape(1000,17,3) -> 3 for xyz
        links_vel = np.array(data.sensors.links.com_lin_velocities()) # shape(1000,17,3) -> 3 for xyz  
            # 0 -> 8: spine joints (from head to tail), 9 -> 17: leg joints
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_positions = np.array(data.sensors.joints.positions_all()) # shape (1000,16)
        joints_velocities = np.array(data.sensors.joints.velocities_all()) # shape (1000,16)
            # Note: checked that this is relative velocity
        joints_torques = np.array(data.sensors.joints.motor_torques_all()) # shape (1000,16)
        # Fix:
        joints_positions[:, [1,5]] *= -1
        joints_velocities[:, [1,5]] *= -1
        joints_torques[:, [1,5]] *= -1

        ground_forces = np.array(data.sensors.contacts.array[:,0:4,2])

        # Metrics (scalar)
        # Note: use dir() to know metrics than can be applied to object
        torque_vec.append(sum_torques(joints_torques))
        speed_vec.append(compute_speed(links_positions, links_vel)[0]) # only axial speed here
        power_speed_vec.append(power_speed_ratio(joints_torques, joints_velocities, links_positions, links_vel))

    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Exo4: output commands
    # plt.figure("Oscillators")
    # plt.plot(times,list(map(spine_cmd,osc_phases[:,0],osc_amplitudes[:,0])), 'b', label="Spine") # phase lag within first half of spine
    # plt.plot(times,list(map(limb_cmd,osc_phases[:,16],osc_amplitudes[:,16])), 'g', label="Limb") # phase lag within first half of spine
    # plt.text(-3.5, spine_cmd(osc_phases[0,0],osc_amplitudes[0,0])+0.4, f'$x_{0 + 1}$', fontsize='large')
    # plt.text(-3.5, limb_cmd(osc_phases[0,16],osc_amplitudes[0,16]), f'$x_{1 + 1}$', fontsize='large')
    # plt.gca().yaxis.set_tick_params(labelleft=False)
    # plt.gca().set_yticks([])
    # plt.xlabel("Time [s]")
    # plt.legend()

    # Plot Traj/Positions
    # head_positions = np.asarray(head_positions)
    # plt.figure('Positions')
    # plot_positions(times, head_positions)
    # plt.figure('Trajectory')
    # plot_trajectory(head_positions)
    # plt.show()

    # Exo5: plot spine angles
    # plt.figure("Spine angle")
    # plt.plot(times, joints_positions[:,0], 'b')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Spine angle [rad]")
        
    # Plot body phase lags 
    # plt.figure("Oscillators")
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,0]-osc_phases[:,1])), 'b', label="Trunk") # phase lag within first half of spine
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,1]-osc_phases[:,2])), 'b') # phase lag within first half of spine
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,2]-osc_phases[:,3])), 'b') # phase lag within first half of spine
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,3]-osc_phases[:,4])), 'r', label="Trunk to tail") # phase lag between first and second half of spine 
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,4]-osc_phases[:,5])), 'g', label="Tail") # phase lag within second half of spine 
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,5]-osc_phases[:,6])), 'g') # phase lag within second half of spine
    # plt.plot(times,list(map(wrap_to_pi,osc_phases[:,6]-osc_phases[:,7])), 'g') # phase lag within second half of spine
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Body phase lag [rad]")

    # 2D plot for grid search speed metric (NOTE: should update x/y labels + ranges)
    # param_range1 = np.linspace(1,3,4) 
    # param_range2 = np.linspace(-1,1,5)+pi
    # results = np.array([[i,j,0] for i in param_range1 for j in param_range2])
    # results[:,2] = np.array(speed_vec)
    # print(results)
    # plot_2d(results,["Drive [-]","Phase offset between limbs and body [rad]","Mean speed [m/s]"]) # param1, param2, metric
    
    # # 2D plot for grid search torque metric (NOTE: should update x/y labels + ranges)
    # param_range1 = np.linspace(1,3,4)
    # param_range2 = [pi/8, 2*pi/8, 3*pi/8] 
    # results = np.array([[i,j,0] for i in param_range1 for j in param_range2])
    # results[:,2] = np.array(torque_vec)
    # print(results)
    # plot_2d(results,["Drive [-]","Phase lag body [rad]","Total torque [N m]"]) # param1, param2, metric

    # # 2D plot for grid search CoT metric (NOTE: should update x/y labels + ranges)
    # param_range1 = np.linspace(1,3,4) 
    # param_range2 = [pi/8, 2*pi/8, 3*pi/8] 
    # results = np.array([[i,j,0] for i in param_range1 for j in param_range2])
    # results[:,2] = np.array(power_speed_vec)
    # print(results)
    # plot_2d(results,["Drive [-]","Phase lag body [rad]", "Power Speed Ratio [J/m]"]) # param1, param2, metric

    # Plot limb phases vs. ground forces
    # CAREFUL: limb order chosen in network not same as paper (and grf), so order shift
    mod_osc_phases = vector_wrap_to_2pi(osc_phases[:,18]) # mod 2pi
    filter_grf = threshold_grf(ground_forces[:,1]) # filtered (thresh 7)
    #plot_phase_force_vs_time(times, mod_osc_phases, filter_grf)
    plot_phase_vs_force(mod_osc_phases, filter_grf) 
    plt.show()

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    

if __name__ == '__main__':
    # file_names = [f'./logs/2a/simulation_{i}' for i in range(12)]
    # file_names = [f'./logs/2b/simulation_{i}' for i in range(12)]
    # file_names = [f'./logs/3a/simulation_{i}' for i in range(20)]
    # file_names = [f'./logs/3b/simulation_{i}' for i in range(28)]
    # file_names = [f'./logs/4b/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/4c/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/5a/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/5b/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/5c/simulation_{i}' for i in range(1)]
    # file_names = [f'./logs/5d/simulation_{i}' for i in range(1)]
    file_names = [f'./logs/6a/simulation_{i}' for i in range(1)]
    main(files=file_names, plot=True)
