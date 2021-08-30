import math

import torch 
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


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
    return  F, torch.abs(F)/N, torch.atan2(F.imag, F.real)


def idft(F):
    """
    Args:
        F [List]: Fourier transform of some signal
    Returns:
        Reconstructed signal: f[n] = Sum(F(k).e(j.2pi.nk/N))
    """
    N = len(f)
    e = torch.tensor(math.e, dtype=f.dtype)
    j = torch.tensor(1j)
    
    f = torch.stack(
        [
            (F * (e**(j*2*math.pi*n*torch.arange(N)/N))).sum() 
            for n in range(N)
        ]
    )/N
    return  f


def topk(F, T=1, top=None):
    """Func to get top components(in actual frequency(Hz)) from Fourier transform 

    Args:
        F [torch.tensor]: Fourier tensor of length N - F(z)
        T [int]: Total Time interval in Sec.
        top [int]: Number of top frequencies to return
    Returns:
        top_freq, top_mag
    """
    N = F.shape[0]
    if top is None:
        top = N

    magnitude = torch.abs(F[:N//2])/N
    top_mag, top_freq = torch.topk(magnitude, top)
    top_freq = torch.div(top_freq, T, rounding_mode='floor')

    return top_freq, top_mag


def get_sine(f, t=None, n=1000, stepsize=None, scale=1., shift=0):
    """ 
    Args:
        f: Frequency in Hz
        t: Total duration in seconds
        n: Total samples
        stepsize: deltat 
        scale: scaling factor
        shift: dc component
    Returns:
        sin(2pi.f.[0, t/n, 2t/n, ..., t])
    """
    if f == 0:
        return scale*torch.ones(n) + shift

    if t is None:
        t = stepsize*n

    return scale*torch.sin(2*(math.pi)*f*torch.linspace(0, t, n)) + shift


def get_square(f, t, n=1000, scale=1, shift=0):
    """ 
    Args:
        f: Frequency in Hz
        t: Total duration in seconds
        n: Total samples
        scale: scaling factor
        shift: dc component
    """

    if t is None:
        t = stepsize*n
    wavelength =  n//t
    half_wavelength = wavelength//2

    wave = torch.ones(n)
    for i in range(half_wavelength, n, wavelength):
        wave[i: i+half_wavelength] = torch.zeros(half_wavelength)

    return scale*wave + shift


# def reconstruct(F, T):
#     """ Reconstruct signal given fourier transform
#     Args:
#         F [Complex tensor]: Fourier coefficients
#         T [int]: Time period in sec.
#     """ 
#     an, bn = F.real, F.imag
#     N = an.shape[0]
#     an[0]/2 + torch.sin(2*(math.pi)*f*torch.linspace(0, T, N)/N)


if __name__ == '__main__':
    # Eg1: Combination of Sinusoidals
    N = 1000
    T = 3
    f = [1, 3, 7, 4]
    k = 5
    
    signal = 2 + get_sine(f[0], T, N)
    for fi in f:
        signal = signal + get_sine(fi, T, N)

    F, mag, phase = dft(signal)
    # plt.plot(mag[:N//2])
    # plt.show()

    freq, mag = topk(F, T, k)
    top_signals = [m*get_sine(fi, T, N) for m, fi in zip(mag, freq)]
    reconstructed = sum(top_signals)

    print(f'Signal consists of topk frequencies {freq} with the magnitude {mag} resp.')

    fig, axs = plt.subplots(3, 1, figsize=(28, 28))
    axs[0].set_title('Original')
    x = torch.linspace(0, T, steps=1000)
    axs[0].plot(x, signal)
    axs[1].set_title(f'top {k} components')
    for sgnl in top_signals:
        axs[1].plot(x, sgnl)
    axs[2].set_title(f'Reconstructed')
    axs[2].plot(x, reconstructed)

    plt.tight_layout(pad=10)
    plt.show()


    # Eg2: Square wave
    N = 1000
    T = 2
    f = 1
    k = 10
    
    signal = get_square(f, T, N, 3., 1.)

    F, mag, phase = dft(signal)

    freq, mag = topk(F, T, k)
    top_signals = [m*get_sine(fi, T, N) for m, fi in zip(mag, freq)]
    reconstructed = sum(top_signals)

    print(f'Signal consists of topk frequencies {freq} with the magnitude {mag} resp.')

    fig, axs = plt.subplots(3, 1, figsize=(28, 28))
    axs[0].set_title('Original')
    x = torch.linspace(0, T, steps=1000)
    axs[0].plot(x, signal)
    axs[1].set_title(f'top {k} components')
    for sgnl in top_signals:
        axs[1].plot(x, sgnl)
    axs[2].set_title(f'Reconstructed')
    axs[2].plot(x, reconstructed)

    plt.tight_layout(pad=10)
    plt.show()
