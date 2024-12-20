"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters
from scipy.signal import lfilter


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

    freq = robot_parameters.freqs # nu
    amplitudes_rate = robot_parameters.amplitudes_rate # a
    nominal_amplitudes = robot_parameters.nominal_amplitudes # R
    weights = robot_parameters.coupling_weights # w
    phi = robot_parameters.phase_bias # phi

    # exo6
    N = np.zeros(n_oscillators)
    N[16:20] = contact_sens
    n = 10
    b = [1.0/n]*n
    a = 1
    N_filtered = lfilter(b,a,N)
    N_filtered = np.clip(N,7,None)
    sigma = robot_parameters.weights_contact_limb

    # Implement equation here
    dphase = np.zeros(n_oscillators) # init dphase
    
    for i in range(n_oscillators):
        dphase[i] = 2*np.pi*freq[i]
        if robot_parameters.feedback and i in range(16,20):
            # print("intrins: ", dphase[i])
            # print("Feedback: ", sigma*N_filtered[i]*np.cos(phases[i]))
            dphase[i] += sigma*N_filtered[i]*np.cos(phases[i])
        for j in range(n_oscillators):
            dphase[i] += amplitudes[j]*weights[i,j]*np.sin(phases[j]-phases[i]-phi[i,j])

    
    dr = np.zeros(n_oscillators) # init dr

    for k in range(n_oscillators):
        dr[k] = amplitudes_rate[k]*(nominal_amplitudes[k]-amplitudes[k])
    
    return np.concatenate([dphase, dr])


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
    # Each limb has 2 degrees of freedom
    # Implement equation here
    q_body = np.zeros(8)
    for i in range(8):
        q_body[i] = amplitudes[i]*(1+np.cos(phases[i])) - amplitudes[i+8]*(1+np.cos(phases[i+8]))

    q_leg = np.array([])
    
    for i in [16,18,17,19]:
        q_leg = np.append(q_leg,amplitudes[i]*np.cos(phases[i])) # shoulder
        q_leg = np.append(q_leg,amplitudes[i]*np.sin(phases[i])) # wrist

    # 16 + 4 oscillators
    # spine motors: 8 -> Mapped from phases[:8] & phases[8:16] and amplitudes[:8] & amplitudes[8:16]
    # leg motors: 8 -> Mapped from phases[16:20] and amplitudes[16:20] with cos(shoulder) & sin(wrist) for each limb
    # output -> spine output for motor (head to tail) + leg output (Front Left
    # shoulder, Front Left wrist, Front Right, Hind Left, Hind right)
    return np.concatenate([q_body,q_leg])


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
        # Replace your oscillator phases here (rather, initial value)
        np.random.seed(0)
        self.state.set_phases( # set_phases: writes phases of this iteration into state array
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
        output = np.zeros([20])
        output[:16] = self.state.amplitudes(iteration=iteration)[:16]*(1+np.cos(self.state.phases(iteration=iteration)[:16]))
        output[16:] = self.state.amplitudes(iteration=iteration)[16:]*+np.cos(self.state.phases(iteration=iteration)[16:])
        return output

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        oscillator_output = motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )
        return oscillator_output


    def get_dphase(self, iteration=None):
        """Get motor position"""
        freq = self.robot_parameters.freqs
        dphase = 2*np.pi*freq
        phases = self.state.phases(iteration=iteration)
        amplitudes = self.state.amplitudes(iteration=iteration)
        phi = self.robot_parameters.phase_bias
        weights = self.robot_parameters.coupling_weights
        for i in range(20):
            for j in range(20):
                dphase[i] += amplitudes[j]*weights[i,j]*np.sin(phases[j]-phases[i]-phi[i,j])

        return dphase/2/np.pi
