import argparse
import os
import utils

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import soundfile
import librosa

from waveunet import Waveunet

def tsne_compute_output(model, inputs, inst):
    all_outputs = [0, 0]

    if model.separate:
            output = model.tsne_forward(inputs, inst)
            all_outputs = output.detach().clone()
    else:
        all_outputs = model.tsne_forward(inputs)

    return all_outputs


def tsne_predict(audio, model, inst):
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    outputs = {key: np.zeros(audio.shape, np.float32) for key in model.instruments}

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            outputs = tsne_compute_output(model, curr_input, inst).cpu().numpy()

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs

def tsne_predict_song(args, audio_path, model, inst):
    model.eval()

    # Load mixture in original sampling rate
    mix_audio, mix_sr = utils.load(audio_path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = utils.resample(mix_audio, mix_sr, args.sr)

    sources = tsne_predict(mix_audio, model, inst)

    return sources

def main(args):
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    if args.cuda:
        model = utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("Loading model from checkpoint " + str(args.load_model))
    state = utils.load_model(model, None, args.load_model, args.cuda)
    print('Step', state['step'])

    output1 = tsne_predict_song(args, args.input1, model, args.tsne_inst).squeeze(0)
    output2 = tsne_predict_song(args, args.input2, model, args.tsne_inst).squeeze(0)
    output3 = tsne_predict_song(args, args.input3, model, args.tsne_inst).squeeze(0)

    output = np.concatenate((output1, output2, output3), axis=0)

    print("debug1: " + str(output1.shape) + ", " + str(output.shape))

    tsne_output = TSNE(learning_rate=100).fit_transform(output)

    labels = ['#ff0000' for i in range(len(output1))] + ['#00ff00' for i in range(len(output2))] + ['#0000ff' for i in range(len(output3))]
    plt.scatter(tsne_output[:, 0], tsne_output[:, 1], c=labels, s=5)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--tsne_inst', type=str, default="bass",
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
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
    parser.add_argument('--output', type=str, default=None, help="Output path (same folder as input path if not set)")

    args = parser.parse_args()

    main(args)
