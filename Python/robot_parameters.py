"""Robot parameters"""

import numpy as np
from farms_core import pylog
from math import pi


class RobotParameters(dict): # inherits from dict class
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases # init to None!! (looks like unused though)
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
        self.rates = np.zeros(self.n_oscillators) # WHAT IS THIS??
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.amplitudes_rate = np.zeros(self.n_oscillators)
        self.feedback_gains_swim = np.zeros(self.n_oscillators)
        self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain

        # our additions
        self.phase_lag_body = parameters.phase_lag_body

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_amplitudes_rate(parameters)  # a_i

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

    def set_phase_bias(self, parameters):
        """Set coupling weights"""
        # Body oscillators, left side
        for i in range(int(self.n_oscillators_body/2)):
            if i != self.n_oscillators_body/2-1: # not at the end of the spinal cord
                # CLARIFY CONVENTION??
                self.phase_bias[i,i+1] = -self.phase_lag_body #parameters.axial_weights
                self.phase_bias[i+1,i] = self.phase_lag_body
            self.phase_bias[i,i+self.n_body_joints] = pi # parameters.contralateral_weights
            self.phase_bias[i+self.n_body_joints,i] = pi

        # Body oscillators, right side
        for i in range(int(self.n_oscillators_body/2), self.n_oscillators_body):
            if i != self.n_oscillators_body-1: # not at the end of the spinal cord
                self.phase_bias[i,i+1] = -self.phase_lag_body
                self.phase_bias[i+1,i] = self.phase_lag_body

        # Limb oscillators, left side
        for i in range(self.n_oscillators_body, int(self.n_oscillators_body+self.n_legs_joints/2)):
            if i != self.n_oscillators_body+self.n_legs_joints/2-1:
                self.phase_bias[i,i+1] = pi
                self.phase_bias[i+1,i] = pi
            self.phase_bias[i,int(i+self.n_legs_joints/2)] = pi
            self.phase_bias[int(i+self.n_legs_joints/2),i] = pi


        # Limb oscillators, right side
        for i in range(int(self.n_oscillators_body+self.n_legs_joints/2), self.n_oscillators):
            if i != self.n_oscillators-1:
                self.phase_bias[i,i+1] = pi
                self.phase_bias[i+1,i] = pi

        # Connections from limb to spine (strong)
        self.phase_bias[0:4,16] = 0.0
        self.phase_bias[4:8,17] = 0.0
        self.phase_bias[8:12,18] = 0.0
        self.phase_bias[12:16,19] = 0.0

    def set_coupling_weights(self, parameters):
        """Set phase bias"""
        # Body oscillators, left side
        for i in range(int(self.n_oscillators_body/2)):
            if i != self.n_oscillators_body/2-1: # not at the end of the spinal cord
                self.coupling_weights[i,i+1] = 10.0 #parameters.axial_weights
                self.coupling_weights[i+1,i] = 10.0
            self.coupling_weights[i,i+self.n_body_joints] = 10.0 # parameters.contralateral_weights
            self.coupling_weights[i+self.n_body_joints,i] = 10.0 

        # Body oscillators, right side
        for i in range(int(self.n_oscillators_body/2), self.n_oscillators_body):
            if i != self.n_oscillators_body-1: # not at the end of the spinal cord
                self.coupling_weights[i,i+1] = 10.0
                self.coupling_weights[i+1,i] = 10.0

        # Limb oscillators, left side
        for i in range(self.n_oscillators_body, int(self.n_oscillators_body+self.n_legs_joints/2)):
            if i != self.n_oscillators_body+self.n_legs_joints/2-1:
                self.coupling_weights[i,i+1] = 10.0
                self.coupling_weights[i+1,i] = 10.0
            self.coupling_weights[i,int(i+self.n_legs_joints/2)] = 10.0
            self.coupling_weights[int(i+self.n_legs_joints/2),i] = 10.0


        # Limb oscillators, right side
        for i in range(int(self.n_oscillators_body+self.n_legs_joints/2), self.n_oscillators):
            if i != self.n_oscillators-1:
                self.coupling_weights[i,i+1] = 10.0
                self.coupling_weights[i+1,i] = 10.0

        # Connections from limb to spine (strong) -> QUESTION: SUPPOSED TO PUT THOSE CONNECTIONS THAT PROBABLY DON'T EXIST?
        self.coupling_weights[0:4,16] = 30.0
        self.coupling_weights[4:8,17] = 30.0
        self.coupling_weights[8:12,18] = 30.0
        self.coupling_weights[12:16,19] = 30.0

    def set_frequencies(self, parameters):
        """Set frequencies"""
        # Body oscillator
        if parameters.drive >= 1.0 and parameters.drive <= 5.0:
            self.freqs[:self.n_oscillators_body] = 0.2*parameters.drive + 0.3
        else: # saturation
            self.freqs[:self.n_oscillators_body] = 0.0 

        # Limb oscillator
        if parameters.drive >= 1.0 and parameters.drive <= 3.0:
            self.freqs[self.n_oscillators_body:] = 0.2*parameters.drive + 0.0
        else: # saturation
            self.freqs[self.n_oscillators_body:] = 0.0

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        # Body oscillator
        if parameters.drive >= 1.0 and parameters.drive <= 5.0:
            self.nominal_amplitudes[:self.n_oscillators_body] = 0.065*parameters.drive + 0.196
        else: # saturation
            self.nominal_amplitudes[:self.n_oscillators_body] = 0.0 

        # Limb oscillator
        if parameters.drive >= 1.0 and parameters.drive <= 3.0:
            self.nominal_amplitudes[self.n_oscillators_body:] = 0.131*parameters.drive + 0.131
        else: # saturation
            self.nominal_amplitudes[self.n_oscillators_body:] = 0.0 

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.amplitudes_rate = 20.0*np.ones(self.n_oscillators)
