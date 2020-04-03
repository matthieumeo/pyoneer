# ############################################################################
# math.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Miscellaneous mathematical functions.
"""

import numpy as np


def sign(x: np.ndarray) -> np.ndarray:
    """
    Sign function, supports complex vectors too.
    :param x: np.ndarray
    Vector.
    :return: np.ndarray
    Sign of the vector.

    Note: For complex case see the book "A Mathematical Introduction to Compressive Sensing" pages 72 and 484.
    """
    if np.isreal(x):
        return np.sign(x)
    else:
        y = x / np.abs(x)
        y[np.abs(x) == 0] = 0
        return x


def sinc(t: np.ndarray, M: int, period: float) -> np.ndarray:
    """
    Cardinal sine.
    :param t: np.ndarray
    :param M: int
    :param period: float
    :return: np.ndarray
    """
    val = np.sin((2 * M + 1) * np.pi * t / period) / np.sin(np.pi * t / period) / (2 * M + 1)
    return val
