
import numpy as np
from matplotlib import pyplot as plt


def getpolicy(Q):
    """ Get best policy matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """

    # POLICY: The best action for each state (x, y), meaning the action INDEX with the highest Q-value.

    P = np.argmax(Q, axis=-1) # action axis

    return P


def getvalue(Q):
    """ Get best value matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """

    # VALUE: The best value for each state (x, y), meaning the maximum Q-value.
    
    V = np.max(Q, axis=-1) # action axis

    return V

