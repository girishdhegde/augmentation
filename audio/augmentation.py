import random
import glob

import torch 
import torch.nn as nn
import torchaudio 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import sounddevice as sd


def rms(samples):
    """Root Mean Square (RMS)."""
    return torch.sqrt((samples**2).mean())


def calculate_desired_noise_rms(signal_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on: 
    - https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    - https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    ---------------------------------------------------------------------------    
    Args:
        signal_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
        snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    ---------------------------------------------------------------------------
    snr = rms(signal).to(db)/rms(noise).to(db)
    rms(noise) = rms(signal)/snr.to(rms)
    to(rms) = 10**(db/20)
    hence:
        noise_rms = signal_rms/(10**(snr/20))
    """
    return signal_rms/(10**(snr/20))


def rms_normalize(samples):
    """Power-normalize samples
    Taken from:
    https://github.com/asteroid-team/torch-audiomentations/blob/master/torch_audiomentations/utils/io.py
    Parameters
    ----------
    samples : (channel, time) Tensor
        Single or multichannel samples
    Returns
    -------
    samples: (channel, time) Tensor
        Power-normalized samples
    """
    rms = samples.square().mean(dim=1).sqrt()
    return (samples.t() / (rms + 1e-8)).t()


def db_to_amplitude(db):
    return 10**(db/20)  #  db = 20log(amplitude)


class Shift(nn.Module):
    def __init__(
        self,
        min_shif = -0.5,
        max_shift = 0.5,
        shift_unit = "fraction",
        rollover = True,
        p = 0.5,
        sample_rate = None,
    ):
        """
        Args:
            min_shif (float, optional): Defaults to -0.5.
            max_shift (float, optional): Defaults to 0.5.
            shift_unit (str, optional): unit of min_shift and max_shift.
                "fraction": Fraction of the total sound length
                "samples": Number of audio samples
                "seconds": Number of seconds
            rollover (bool, optional): Defaults to True.
            p (float, optional): Defaults to 0.5.
            sample_rate ([type], optional): Defaults to None.
        """
        super().__init__()
        self.min_shif = min_shif
        self.max_shif = max_shif
        self.shift_unit = shift_unit
        self.rollover = rollover
        self.p = p
        self.sample_rate = sample_rate

        if shift_unit == 'fraction':
            self.get_shift = self._fraction_shift
        elif shift_unit == 'seconds':
            s = int(random.uniform(
                    self.min_shif*self.sample_rate, self.max_shif*self.sample_rate
                ))
            self.get_shift = lambda x: s
        elif shift_unit == 'samples':
            s = int(random.uniform(self.min_shif, self.max_shif))
            self.get_shift = lambda x: s

    def _fraction_shift(self, x):
        n = x.shape[-1]
        return int(random.uniform(self.min_shif*n, self.max_shif*n))

    def forward(self, x):
        """
        Args:
            x ([Tensor]): Waveform of shape [ch, n] or [n]
        """
        if random.random() < self.p:
            shifts = self.get_shift(x)
            x = torch.roll(x, shifts, dims=-1)
            if not self.rollover:
                if shifts > 0:
                    x[..., :shifts] = 0.
                elif shifts < 0: 
                    x[..., -shifts:] = 0.
        return x

 
class Gain(nn.Module):
    def __init__(
        self,
        min_gain_db = -18.0,
        max_gain_db = 6.0,
        p = 0.5,
    ):
        """ Multiply the audio by a random amplitude factor to reduce or increase the volume. This
        technique can help a model become somewhat invariant to the overall gain of the input audio.

        Args:
            min_gain_db (float, optional): Defaults to -18.0.
            max_gain_db (float, optional): Defaults to 6.0.
            p (float, optional): Defaults to 0.5.
        """
        super().__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p

    def forward(self, x):
        """
        Args:
            x ([Tensor]): Waveform of shape [ch, n] or [n]
        """
        if random.random() < self.p:
            x = x*db_to_amplitude(torch.randn(x.shape[-1]))
        return x


class GaussianNoise(nn.Module):
    def __init__(
        self,
        min_snr,
        max_snr,
        p=0.5
    ):
        """
        Args:
            min_snr ([float/int]): minimum signal to noise ratio
            max_snr ([float/int]): maximum signal to noise ratio
            p (float, optional): [description]. Defaults to 0.5.
        """
        super().__init__()
        
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def forward(self, x):
        """
        Args:
            x ([Tensor]): Waveform of shape [ch, n] or [n]
        """
        if random.random() < self.p:
            xrms = rms(x)
            noiserms = calculate_desired_noise_rms(
                xrms, 
                random.uniform(self.min_snr, self.max_snr)
            )
            x = x + noiserms*torch.randn_like(x)
        return x
 
 
class BackgroundNoise(nn.Module):
    def __init__(
        self,
        min_snr,
        max_snr,
        noise,
        p=0.5
    ):
        """
        Args:
            min_snr ([float/int]): minimum signal to noise ratio
            max_snr ([float/int]): maximum signal to noise ratio
            noise (Path/List/Tensor, optional): Noise audio
                - Path: Directory path to .wav noise files
                - List: List noise files path
                - Tensor: [total, ch, n] Audio Tensor
            p (float, optional): [description]. Defaults to 0.5.
        Based on:
            https://github.com/asteroid-team/torch-audiomentations/blob/master/torch_audiomentations/utils/io.py
        """
        super().__init__()
        
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

        if isinstance(noise, string):
            self.noise = glob.glob(f'{noise}/*.wav')
            self.is_file = True
        if isinstance(noise, (list, tuple, set)):
            self.noise = noise
            self.is_file = True
        if isinstance(noise, torch.Tensor):
            self.noise = noise
            self.is_file = False

    def random_background(self, audio):
        pieces = []
        ch, missing_num_samples = audio.shape
        while missing_num_samples > 0:
            if self.is_file:
                noise = random.choice(self.noise)
                noise, sr = torchaudio.load(noise)
            else: 
                noise = random.choice(self.noise)
            background_ch, background_num_samples = noise.shape
            if background_ch < ch:
                noise =  repeat(noise, '1 n -> b n', b=ch)

            if background_num_samples > missing_num_samples:
                sample_offset = random.randint(
                    0, background_num_samples - missing_num_samples
                )
                background_samples = noise[..., sample_offset: missing_num_samples]
                missing_num_samples = 0
            else:
                background_samples = noise
                missing_num_samples -= background_num_samples

            pieces.append(background_samples)

        #  the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        #  the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        #  (this simplifies "apply_transform" logic)
        return rms_normalize(
            torch.cat([rms_normalize(piece) for piece in pieces], dim=1)
        )

    def forward(self, x):
        """
        Args:
            x ([Tensor]): Waveform of shape [ch, n]
        """
        if random.random() < self.p:
            xrms = rms(x)
            noiserms = calculate_desired_noise_rms(
                xrms, 
                random.uniform(self.min_snr, self.max_snr)
            )
            noise = random_backround(x)
            x = x + noiserms*torch.randn_like(x)
        return x
 