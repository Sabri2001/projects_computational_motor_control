"""Simulation parameters"""


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.initial_phases = None
        self.position_body_gain = 0.6  # default do not change
        self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...

        # Disruptions
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0

        # Tegotae
        self.weights_contact_body = 0.0
        self.weights_contact_limb_i = 0.0
        self.weights_contact_limb_c = 0.0

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

