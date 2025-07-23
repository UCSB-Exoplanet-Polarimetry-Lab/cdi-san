"""making deformable mirror influence functions"""
from prysm.mathops import np

def gaussian_influence_function(r, actuator_pitch):
    """creates a gaussian influence function

    Parameters
    ----------
    r : ndarray
        array of radial coordinates in DM actuator command space, mm
    actuator_pitch : float
        pitch of the gaussian actuators, mm

    Returns
    -------
    ndarray
        array containing a centered gaussian influence function.
        intended as an input into prysm.x.DM
    """
    return np.exp(-r**2 / (2 * actuator_pitch))


def sinc_influence_function(x, y, actuator_pitch):
    """creates a sinc influence function

    Parameters
    ----------
    x : ndarray
        array of coordinates in DM actuator command space, mm
    y : ndarray
        array of coordinates in DM actuator command space, mm
    actuator_pitch : float
        pitch of the gaussian actuators, mm

    Returns
    -------
    ndarray
        array containing a centered sinc influence function.
        intended as an input into prysm.x.DM
    """

    return np.sinc(x / actuator_pitch) * np.sinc(y / actuator_pitch)
