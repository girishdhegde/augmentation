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
            (f*(e**(-j*2*math.pi*n*torch.arange(N)/N))).sum() 
            for n in range(N)
        ]
    )
    return  F, torch.abs(F)/N, torch.atan2(F.imag, F.real)


def idft(F, top=None):
    """
    Args:
        F [List]: Fourier transform of some signal
        top [Optional[int]]: If given only top components will be used for reconstruction
    Returns:
        Reconstructed signal: f[n] = Sum(F(k).e(j.2pi.nk/N))
    """
    N = F.shape[0]
    if top is None:
        top = N//2
    top = top*2

    magnitude = torch.abs(F)
    top_mag, top_idx = torch.topk(magnitude, top)

    e = torch.tensor(math.e, dtype=F.dtype)
    j = torch.tensor(1j)
    
    f = torch.stack(
        [
            (F[top_idx]*(e**(j*2*math.pi*n*top_idx/N))).sum() 
            for n in range(N)
        ]
    )/N

    return  f


def get_Hz_scale(f, T):
    """
    Args:
        f: Frequency vector
        T: Time period in seconds
    """
    return f/(T*f.shape[0])


def reconstruct(F, T, top=None):
    """ Func. to reconstruct signal from Fourier coefficient
    Args:
        F [Tensor[Complex]]: Fourier coeficients
        T [int]: Timie duration in sec.
        top [Optional[int]]: Number of 'top' components to consider 
    """
    freq, mag, ph = topk(F, T, top)
    N = F.shape[0]
    
    top_signals = [get_cos(fi, T, N, scale=mi, phase=pi) for mi, fi, pi in zip(mag, freq, ph)]
    reconstructed = sum(top_signals)

    return reconstructed, top_signals


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
        top = N//2

    magnitude = torch.abs(F[:N//2])/(N//2)  # Only considering real part -> N//2
    magnitude[0] = magnitude[0]/2
    phase = torch.atan2(F.imag, F.real)
    top_mag, top_freq = torch.topk(magnitude, top)
    top_phase = phase[top_freq]
    top_freq = torch.div(top_freq, T, rounding_mode='floor')

    return top_freq, top_mag, top_phase


def get_sine(f, t=None, n=1000, stepsize=None, scale=1., shift=0, phase=0):
    """ 
    Args:
        f: Frequency in Hz
        t: Total duration in seconds
        n: Total samples
        stepsize: deltat 
        scale: scaling factor
        shift: dc component
        phase: Phase shift (phi)
    Returns:
        sin(2pi.f.[0, t/n, 2t/n, ..., t] + phase)
    """
    if f == 0:
        return scale*torch.ones(n) + shift

    if t is None:
        t = stepsize*n

    return scale*torch.sin(2*(math.pi)*f*torch.linspace(0, t, n) + phase) + shift


def get_cos(f, t=None, n=1000, stepsize=None, scale=1., shift=0, phase=0):
    """ 
    Args:
        f: Frequency in Hz
        t: Total duration in seconds
        n: Total samples
        stepsize: deltat 
        scale: scaling factor
        shift: dc component
        phase: Phase shift (phi)
    Returns:
        cos(2pi.f.[0, t/n, 2t/n, ..., t] + phase)
    """
    if f == 0:
        return scale*torch.ones(n) + shift

    if t is None:
        t = stepsize*n

    return scale*torch.cos(2*(math.pi)*f*torch.linspace(0, t, n) + phase) + shift


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
    wavelength =  n//(f*t)
    half_wavelength = wavelength//2

    wave = torch.ones(n)
    for i in range(half_wavelength, n, wavelength):
        wave[i: i+half_wavelength] = torch.zeros(half_wavelength)

    return scale*wave + shift


if __name__ == '__main__':
    # Eg1: Combination of Sinusoidals
    N = 1000
    T = 3
    f = [0, 1, 3, 7, 4]
    m = [2, 1, 0.7, 0.2, .8]
    k = 5
    
    signal = get_sine(f[0], T, N)
    if len(f) > 1:
        for fi, mi in zip(f[1:], m[1:]):
            signal = signal + get_sine(fi, T, N, scale=mi)
    
    F, mag, phase = dft(signal)
    reconstructed, top_signals = reconstruct(F, T, k)
    # reconstructed = idft(F, top=k)
    hz_scale = torch.arange(F.shape[0]//2)/T
    plt.plot(hz_scale, mag[:N//2])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.title('Magnitude Spectrum')
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(28, 28))
    axs[0].set_title('Original')
    x = torch.linspace(0, T, steps=N)
    axs[0].plot(x, signal)
    axs[1].set_title(f'top {k} components')
    for sgnl in top_signals:
        axs[1].plot(x, sgnl)
    axs[2].set_title(f'Reconstructed')
    axs[2].plot(x, reconstructed)

    plt.tight_layout(pad=10)
    plt.show()

    ####################################################################################
    # Eg2: Square wave
    ####################################################################################
    N = 1000
    T = 2
    f = 2
    k = 10
    
    signal = get_square(f, T, N, 3., 1.)
    F, mag, phase = dft(signal)
    reconstructed, top_signals = reconstruct(F, T, k)
    # reconstructed = idft(F, top=k)

    hz_scale = torch.arange(F.shape[0]//2)/T
    plt.plot(hz_scale, mag[:N//2])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.title('Magnitude Spectrum')
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(28, 28))
    axs[0].set_title('Original')
    x = torch.linspace(0, T, steps=N)
    axs[0].plot(x, signal)
    axs[1].set_title(f'top {k} components')
    for sgnl in top_signals:
        axs[1].plot(x, sgnl)
    axs[2].set_title(f'Reconstructed')
    axs[2].plot(x, reconstructed)
    plt.tight_layout(pad=10)
    plt.show()
