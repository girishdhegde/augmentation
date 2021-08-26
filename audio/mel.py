import time
import os 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import sounddevice as sd

import torch 
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from utils import print_stats, plot_waveform, plot_specgram, play_audio, plot_spectrogram


def get_hann_filter(N=512):
    N -= 1
    return 0.2 * (1 - (torch.cos(2*np.pi*torch.arange(N + 1)/N)))


def framing(signal, win_length, hop_length):
    length = signal.shape
    if len(length) > 1:
        length = length[1]
        signal = signal[0]
    else: 
        length = length[0]

    frames = []
    for start in range(0, length - win_length, hop_length):
        frames.append(signal[start: start + win_length])

    return torch.stack(frames)


def windowing(frames, window=None):
    if window is None:
        window = get_hann_filter(N=frames.shape[1])
    return frames*window[None, :]


def mel_spectogram(waveform, win_length, hop_length):
    frames = framing(waveform, win_length, hop_length)
    windows = windowing(frames)

    # fig, axes = plt.subplots(1, 5, figsize=(24, 24))
    # for i, ax in enumerate(axes):
    #     ax.plot(windows[i])
    #     # ax.axis('off')
    #     ax.set_xticklabels(())
    #     ax.set_yticklabels(())

    # # fig.suptitle('Windowing(Framing(Waveform))')
    # # fig.tight_layout()
    # plt.savefig('../output/windowing.png')
    # plt.show()

    freq = torch.fft.rfft(windows, dim=1, )
    freq = torch.abs(freq.T) * 2

    spec = (freq)/torch.sum(get_hann_filter(win_length))

    spec_in_db = 20*torch.log10(spec/torch.finfo(torch.float32).max)

    return freq, spec, spec_in_db



def get_mel_from_hertz(hertz):
    return 2595 * np.log10(1 + (hertz/ 700))

def get_hertz_from_mel(mel):
    return 700 * (10**(mel / 2595) - 1)


if __name__ == '__main__':
    audio_path = '../data/voice.wav'
    win_length = 25 # in ms
    hop_length = 10 # in ms
    n_fft = 512


    metadata = torchaudio.info(audio_path)
    print(metadata)

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform[0].view(1, -1)

    print_stats(waveform, sample_rate=sample_rate)
    plot_waveform(waveform, sample_rate, save='../output/waveform.png')
    # # plot_specgram(waveform, sample_rate)
    play_audio(waveform, sample_rate, )

    win_length = int((win_length*sample_rate)/1000)
    hop_length = int((hop_length*sample_rate)/1000)

    freq, spec, spec_in_db = mel_spectogram(waveform, win_length, hop_length)

    fig = plt.figure()
    plt.imshow(spec, cmap='inferno')
    plt.gca().invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Frequency(Color - Amplitude)')
    plt.savefig('../data/spectogram.png')
    plt.show()

    fig = plt.figure()
    plt.imshow(spec_in_db, cmap='inferno')
    plt.gca().invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Frequency(Color - Amplitude in db)')
    plt.savefig('../data/spectogram_in_db.png')
    plt.show()

    mel_range = get_mel_from_hertz(torch.arange(spec_in_db.shape[0]))

    fig = plt.figure()
    librosa.display.specshow(spec_in_db.numpy())
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency(Color - Amplitude in db)')
    plt.colorbar()
    plt.savefig('../data/mel_spectogram.png')
    plt.show()



    # mel_spectrogram = T.MelSpectrogram(
    # sample_rate=sample_rate,
    # n_fft=win_length,
    # win_length=win_length,
    # hop_length=hop_length,
    # center=True,
    # pad_mode="reflect",
    # power=2.0,
    # norm='slaney',
    # onesided=True,
    # )

    # melspec = mel_spectrogram(waveform)
    # plot_spectrogram(
    #     melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')