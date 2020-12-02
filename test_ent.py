import argparse
import os
import utils

import torch
import numpy as np

from test import predict_song
from waveunet import Waveunet
from loss import compute_power_spectral_entropy as entropy

def main(args):
    # MODEL

    m = args.m 
    r = args.r 

    input1 = load_song(args, args.input1)[:, :, 1024:2048]
    print(input1.shape)
    input2 = load_song(args, args.input2)[:, :, 1024:2048]
    input3 = load_song(args, args.input3)[:, :, 1024:2048]
    input4 = load_song(args, args.input4)[:, :, 1024:2048]

    print(entropy(input1))
    print(entropy(input2))
    print(entropy(input3))
    print(entropy(input4))
    print("===========")
    mix1 = torch.cat([input1, input1], dim=2)
    mix2 = torch.cat([input1, input2], dim=2)
    mix3 = torch.cat([input1, input3], dim=2)
    mix4 = torch.cat([input1, input4], dim=2)
    print(entropy(mix1))
    print(entropy(mix2))
    print(entropy(mix3))
    print(entropy(mix4))

def load_song(args, path):

    # Load mixture in original sampling rate
    mix_audio, mix_sr = utils.load(path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
        mix_audio = np.tile(mix_audio, [args.channels, 1])
    else:
        assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = utils.resample(mix_audio, mix_sr, args.sr)
    
    return torch.from_numpy(mix_audio).unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["vocals", "bass", "drums", "other"],
                        help="List of instruments to separate (default: \"vocals bass drums other\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet/model',
                        help='Reload a previously trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    parser.add_argument('--input1', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--input2', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--input3', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--input4', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--target', type=str, default=None, help="Output path (same folder as input path if not set)")

    parser.add_argument('--difference_output', type=int, default=0,
                        help="Train last instrument as difference of input and sum of other instruments (1 for True and 0 for False)")

    parser.add_argument('--m', type=int, default=0,
                        help="Train last instrument as difference of input and sum of other instruments (1 for True and 0 for False)")

    parser.add_argument('--r', type=float, default=0,
                        help="Train last instrument as difference of input and sum of other instruments (1 for True and 0 for False)")

    args = parser.parse_args()

    main(args)
