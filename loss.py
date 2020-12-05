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

def compute_L1_time() :
    def computet(inputs, targets):
        # shape : batch, channel, length
        L1 = nn.L1Loss()

        sss = torch.stft(torch.mean(inputs, dim=1), n_fft=400)
        # [B, N, F, 2]
        time_diff = torch.abs(sss[:, :, :-1, 0] - sss[:, :, 1:, 0])
        time_diff = torch.mean(torch.mean(time_diff, dim=2), dim=1)

        result = L1(inputs, targets) + 0.1 * time_diff
        return torch.mean(result)
    return computet

def compute_multi_scale_spectral_loss() :
    def computec(inputs, targets):
        loss = []
        n_ffts = [2048, 512, 128, 32]
        for n_fft in n_ffts:
            spec_inputs = torch.stft(torch.mean(inputs, dim=1), n_fft=n_fft)[:, :, :, 0]
            spec_targets = torch.stft(torch.mean(targets, dim=1), n_fft=n_fft)[:, :, :, 0]
            # [B, N, F]
            L1 = torch.mean(torch.mean(torch.abs(spec_inputs - spec_targets), dim=2), dim=1)
            L1_log = torch.mean(torch.mean(torch.abs(torch.log(spec_inputs) - torch.log(spec_targets)), dim=2), dim=1)
            loss.append(L1 + L1_log)
        return torch.mean(torch.stack(loss, dim=1))
    return computec

def compute_L1_entropy(m, r) :
    def compute(inputs, targets):
        # shape : batch, channel, length
        L1 = nn.L1Loss()

        #concatted = torch.cat([inputs, targets], dim=2)
        result = L1(inputs, targets) + 0.01 * compute_power_spectral_entropy(inputs)
        return torch.mean(result)
    return compute

def compute_spatial_entropy(inputs, targets, n_fft):
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

def compute_approximate_entropy(inputs, m, r):
    phi_m = compute_ap_phi(inputs, m, r)    
    phi_m1 = compute_ap_phi(inputs, m + 1, r)    
    return phi_m - phi_m1

def compute_ap_phi(inputs, m, r):
    mono_inputs = torch.mean(inputs, dim=1)
    # [B, T]
    clip_inputs = torch.stack([mono_inputs[:, i:i+m] for i in range(0, mono_inputs.shape[1] - m)], dim=1)
    # [B, T-m, m]
    ref_inputs = torch.unsqueeze(clip_inputs[:, :, 0], 2)
    ref_inputs = ref_inputs.repeat(1, 1, m)

    dif_inputs = clip_inputs - ref_inputs

    c = []
    for i in range(dif_inputs.shape[1]):
        tmp = torch.unsqueeze(dif_inputs[:, i, :], 1).repeat(1, dif_inputs.shape[1], 1)
        c_i = torch.sum((tmp - dif_inputs) ** 2, dim=2)
        c_i = torch.sum(torch.sigmoid(torch.ones_like(c_i) * r - c_i), dim=1)
        c.append(c_i)
    p = torch.mean(torch.log(torch.stack(c, dim=1)), dim=1)

    return p

def compute_wiener_entropy(inputs):
	# [B, C, T]
	sss = torch.stft(torch.mean(inputs, dim=1), n_fft=400)
	# [B, N, F, 2]
	sss = sss[:, :, :, 0] ** 2 + sss[:, :, :, 1] ** 2
	# [B, N, F]
	geo = torch.exp(torch.mean(torch.log(sss), dim=1))
	# [B, F]
	mean = torch.mean(sss, dim=1)
	eps = 0.00001 * torch.ones_like(mean)
	wiener = geo / (mean + eps)
	return torch.mean(wiener, dim=1)	

def compute_power_spectral_entropy(inputs):
	# [B, C, T]
	sss = torch.stft(torch.mean(inputs, dim=1), n_fft=400)
	# [B, N, F, 2]
	sss = sss[:, :, :, 0] 
	# [B, N, F]
	spec = torch.mean(sss ** 2, dim=1)
	# [B, F]
	summ = torch.unsqueeze(torch.sum(spec, dim=1), dim=1).repeat(1, spec.shape[1])
	prob = spec / summ 
	entropy = torch.sum(prob * torch.log(prob), dim=1) * -1
	return entropy	

