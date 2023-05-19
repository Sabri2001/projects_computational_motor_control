"""[Project1] Exercise 6: Walking with sensory feedback"""

import os
import pickle
import numpy as np
import matplotlib.animation as manimation
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
# import farms_pylog


def exercise_6a_phase_relation(timestep):
    """Exercise 6a - Relationship between phase of limb oscillator & swing-stance
    (Project 2 Question 1)

    This exercise helps in understanding the relationship between swing-stance
    during walking and phase of the limb oscillators.

    Implement rigid spine with limb movements to understand the relationship.
    Hint:
        - Use the spine's nominal amplitude to make the spine rigid.

    Observer the limb phase output plot versus ground reaction forces.
        - Apply a threshold on ground reaction forces to remove small signals
        - Identify phase at which the limb is in the middle of stance
        - Identify which senstivity function is better suited for implementing
        tegotae feedback for our CPG-controller.
        - Identify if weights for contact feedback should be positive  or negative
        to get the right coordination

    """
    # Use exercise_example.py for reference
    pass
    return


def exercise_6b_tegotae_limbs(timestep):
    """Exercise 6b - Implement tegotae feedback
    (Project 2 Question 4)

    This exercise explores the role of local limb feedback. Such that the feedback
    from the limb affects only the oscillator corresponding to the limb.

    Keep the spine rigid and straight to observed the effect of tegotae feedback

    Implement uncoupled oscillators by setting the following:
    weights_body2body = 30
    weights_limb2body = 0
    weights_limb2limb = 0

    Implement only local sensory feedback(same limb) by setting:
    weights_contact_limb_i = (explore positive and negative range of values)

    Hint:
    - Apply weights in a small range, such that the feedback values are not greater than
    the oscilltor's intrinsic frequency
    - Observer the oscillator output plot. Check the phase relationship between
    limbs.

    """

    # Use exercise_example.py for reference
    pass
    return


def exercise_6c_tegotae_spine(timestep):
    """Exercise 6c - Effect of spine undulation with tegotae feedback
    (Project 2 Question 5)

    This exercise explores the role of spine undulation and how
    to combine tegotae sensory feedback with spine undulations.

    We will implement the following cases with tegotae feedback:

    1. spine undulation, with no limb to body coupling, no limb to limb coupling
    2. spine undlation, with limb to body coupling, no limb to limb coupling
    3. spine undlation, with limb to body coupling, with limb to limb coupling

    Comment on the three cases, how they are similar and different.
    """
    return


def exercise_6d_open_vs_closed(timestep):
    """Exercise 6d - Open loop vs closed loop behaviour
    (Project 2 Question 6)

    This exercise explores the differences in open-loop vs closed loop.

    Implement the following cases
    1. Open loop: spine undulation, with limb to body coupling, no limb to limb coupling
    2. Open loop: spine undlation, with limb to body coupling, with limb to limb coupling
    3. Closed loop: spine undulation, with limb to body coupling, no limb to limb coupling
    4. Closed loop: spine undlation, with limb to body coupling, with limb to limb coupling

    Comment on the three cases, how they are similar and different.
    """
    return


if __name__ == '__main__':
    exercise_6a_phase_relation(timestep=1e-2)
    exercise_6b_tegotae_limbs(timestep=1e-2)
    exercise_6c_tegotae_spine(timestep=1e-2)
    exercise_6d_open_vs_closed(timestep=1e-2)

