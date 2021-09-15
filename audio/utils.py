import matplotlib
import matplotlib.pyplot as plt
import time
import os 

import torch 
import torchaudio 
import librosa

import numpy as np
import sounddevice as sd


# Taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None, save=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None, save=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

    if save is not None:
        plt.savefig(save)
    plt.show()


def play_audio(waveform, sample_rate, save=None):
    if save is not None:
        torchaudio.save(save, waveform, sample_rate)
    waveform = waveform.numpy()
    sd.play(waveform[0], sample_rate, blocking=True)


def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, save=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show()

def get_spectrogram(
    waveform,
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)

def plot_pitch(waveform, sample_rate, pitch, save=None):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time,  waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln2 = axis2.plot(
        time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

    axis2.legend(loc=0)

    if save is not None:
        plt.savefig(save)
    plt.show()


if __name__ == '__main__':
    audio_path = '../data/train.wav'
    metadata = torchaudio.info(audio_path)
    print(metadata)

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform[0].view(1, -1)

    print_stats(waveform, sample_rate=sample_rate)
    plot_waveform(waveform, sample_rate, save='../output/waveform.png')
    plot_specgram(waveform, sample_rate)
    play_audio(waveform, sample_rate, )