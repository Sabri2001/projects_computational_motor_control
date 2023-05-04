"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.feedback_gains_swim = np.zeros(self.n_oscillators)
        self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # print("GPGS: {}".format(gps[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))

    def set_frequencies(self, parameters):
        """Set frequencies"""
        self.frequencies = 2.0*np.ones(self.n_oscillators)
        return self.frequencies

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        for i in range(self.n_body_joints):
            self.coupling_weights[i,i+1] = 10.0 #parameters.axial_weights
            self.coupling_weights[i+1,i] = 10.0
            self.coupling_weights[i,i+8] = 10.0
            self.coupling_weights[i+8,i] = 10.0 # parameters.contralateral_weights
        return self.coupling_weights

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        for i in range(self.n_body_joints):
            self.phase_bias[i,i+1] = -2*np.pi/8
            self.phase_bias[i+1,i] = 2*np.pi/8
            self.phase_bias[i,i+8] = np.pi
            self.phase_bias[i+8,i] = np.pi
        return  self.phase_bias

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.amplitudes_rate = 20.0*np.ones(self.n_oscillators)
        return self.amplitudes_rate

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        self.nominal_amplitudes = 1.0*np.ones(self.n_oscillators)
        return self.nominal_amplitudes