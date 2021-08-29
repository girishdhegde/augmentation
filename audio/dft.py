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


def dft(f):
    """
    Args:
        f [List]: List of amplitude of any signal
    Returns:
        Fourier transform of sequence: F[n] = Sum(f(k).e-(j.2pi.nk/N))
        Magnitude(F[n])
        Phase([F[n]])
    """
    N = len(f)
    e = torch.tensor(math.e, dtype=f.dtype)
    j = torch.tensor(1j)
    
    F = torch.stack(
        [
            (f * (e**(-j*2*math.pi*n*torch.arange(N)/N))).sum() 
            for n in range(N)
        ]
    )
    return  F, torch.abs(F), torch.atan2(F.imag, F.real)


def to_actual_frequency(F, T):
    """Fourier transform to actual frequency in Hz

    Args:
        F [torch.tensor]: Fourier tensor of length N - F(z)
        T [int]: Total Time interval in Sec.
    """
    magnitude = torch.abs(F)

if __name__ == '__main__':
    N = 1000
    T = 3
    f = [1, 3, 5, 4]
    
    signal = get_sine(f[0], T, N)
    for fi in f:
        signal = signal + get_sine(fi, T, N)

    plt.plot(signal)
    plt.show()

    F, mag, phase = dft(signal)

    print(mag[:N//2].argmax())

    plt.plot(mag[:N//2])
    plt.show()
