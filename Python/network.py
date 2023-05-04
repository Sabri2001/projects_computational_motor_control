"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters : RobotParameters, loads, contact_sens):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    dstate: <np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]

    freq = robot_parameters.frequencies
    amplitudes_rate = robot_parameters.amplitudes_rate
    nominal_amplitudes = robot_parameters.nominal_amplitudes

    # Implement equation here
    dtheta = np.zeros(n_oscillators)
    print(n_oscillators)
    
    for i in range(n_oscillators):
        dtheta[i] = 2*np.pi*freq[i]
        w = robot_parameters.set_coupling_weights(i)
        phi = robot_parameters.set_phase_bias(i)
        for j in range(n_oscillators):
            dtheta[i] += amplitudes[j]*w[i,j] *np.sin(phases[j]-phases[i]-phi[i,j])
    
    dr = np.zeros(n_oscillators)
    print(np.size(amplitudes_rate))
    print(np.size(dr))
    for k in range(n_oscillators):
        dr[k] = amplitudes_rate[k]*(nominal_amplitudes[k]-amplitudes[k])
    
    return np.concatenate([dtheta, dr])


def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    motor_outputs: <np.array>
        Motor outputs for joint in the system.

    """
    # Last 4 oscillators define output of each limb.
    # Each limb has 2 degree of freedom
    # Implement equation here
    q_body = np.zeros(8)
    for i in range(8):
        q_body[i] = amplitudes[i]*(1+np.cos(phases[i])) - amplitudes[i+8]*(1+np.cos(phases[i+8]))

    q_shoulder = np.zeros(4)
    q_wrist = np.zeros(4)
    for i in range(4):
        q_shoulder[i] = amplitudes[i+16]*np.cos(phases[i+16])
        q_wrist[i] = amplitudes[i+16]*np.sin(phases[i+16])

    # 16 + 4 oscillators
    # spine motors: 8 -> Mapped from phases[:8] & phases[8:16] and amplitudes[:8] & amplitudes[8:16]
    # leg motors: 8 -> Mapped from phases[16:20] and amplitudes[16:20] with cos(shoulder) & sin(wrist) for each limb
    # output -> spine output for motor (head to tail) + leg output (Front Left
    # shoulder, Front Left wrist, Front Right, Hind Left, Hind right)
    # phases = np.append(
    #    np.zeros_like(phases)[
    #        :16], np.zeros_like(phases)[
    #        16:20])
    # amplitudes = np.append(
    #    np.zeros_like(amplitudes)[
    #        :16], np.zeros_like(amplitudes)[
    #        16:20])
    return np.concatenate([q_body,q_shoulder,q_wrist])


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state):
        super().__init__()
        self.n_iterations = n_iterations
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.rand(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None, contact_sens=None):
        """Step"""
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads, contact_sens)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        output = self.state.amplitude(iteration=iteration)*(1+np.cos(self.state.phases(iteration=iteration)))
        return output

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        oscillator_output = motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )
        return oscillator_output

