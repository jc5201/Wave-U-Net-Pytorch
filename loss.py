import torch
import numpy as np

import torch.nn as nn
import utils

def compute_si_sdr(inputs: torch.Tensor, targets: torch.Tensor):
    # shape : batch, channel, length
    shape = inputs.shape
    eps = 1e-4
    mean_inputs = torch.mean(inputs, dim=2, keepdim=True)
    mean_targets = torch.mean(targets, dim=2, keepdim=True)
    zero_mean_inputs = inputs - mean_inputs
    zero_mean_targets = targets - mean_targets

    s_targets = torch.unsqueeze(inputs, dim=1)  # batch, 1, channel, length
    s_inputs = torch.unsqueeze(targets, dim=2)  # batch, channel, 1, length

    pair_wise_dot = torch.sum(s_inputs * s_targets, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_targets ** 2, dim=3, keepdim=True) + eps  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_targets / s_target_energy  # [B, C, C, T]

    e_noise = s_inputs - pair_wise_proj  # [B, C, C, T]

    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + eps)
    pair_wise_si_snr = -10 * torch.mean(torch.log10(pair_wise_si_snr + eps))  # [B, C, C]
    return pair_wise_si_snr

def compute_L1_entropy(n_fft) :
    def compute(inputs, targets):
        # shape : batch, channel, length
        L1 = nn.L1Loss()

        return L1(inputs, targets) + compute_entropy(inputs, targets, n_fft)
    return compute

def compute_entropy(inputs, targets, n_fft):
    eps = 0.001

    mono_inputs = torch.mean(inputs, dim=1)
    mono_targets = torch.mean(targets, dim=1)
    spec_inputs = torch.stft(mono_inputs, n_fft=n_fft, center=False, normalized=False, onesided=True)[:, :, :, 0]       # [B, N(freq), F(frame)]
    spec_targets = torch.stft(mono_targets, n_fft=n_fft, center=False, normalized=False, onesided=True)[:, :, :, 0]

    repeat_inputs = torch.unsqueeze(spec_inputs.permute(0, 2, 1), 2)    # [B, F, 1, N]
    repeat_inputs = repeat_inputs.repeat(1, 1, repeat_inputs.shape[1], 1) # [B, F, F, N]
    repeat_inputs = repeat_inputs.view(repeat_inputs.shape[0], -1, repeat_inputs.shape[3])

    repeat_targets = spec_targets.permute(0, 2, 1)    # [B, F, N]
    repeat_targets = repeat_targets.repeat(1, repeat_targets.shape[1], 1) # [B, F*F, N]

    #distance = torch.mean((repeat_inputs - repeat_targets) ** 2, dim=2).view(spec_inputs.shape[0], spec_inputs.shape[2], spec_inputs.shape[2])
    distance = ((spec_inputs - spec_targets) ** 2)
    distance = distance + torch.ones_like(distance) * eps
    entropy = torch.log(torch.pow(distance, -1))
    entropy_sum = torch.mean(torch.mean(entropy, dim=2), dim=1) * -1

    return entropy_sum

