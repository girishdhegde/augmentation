import math

import torch 
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def get_sine(f, t=None, n=1000, stepsize=None):
    """ 
    Args:
        f: Frequency in Hz
        t: Total duration in seconds
        n: Total samples
        stepsize: deltat 
    Returns:
        sin(2pi.f.[0, t/n, 2t/n, ..., t])
    """

    if t is None:
        t = stepsize*n

    return torch.sin(2*(math.pi)*f*torch.linspace(0, t, n))


def dft():
    pass


if __name__ == '__main__':
    sine1 = get_sine(1, 2, 1000)
    plt.plot(sine1)
    plt.show()