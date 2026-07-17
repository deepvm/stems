import math
import mlx.core as mx

def reflect_pad_1d(x: mx.array, left: int, right: int) -> mx.array:
    """Implement PyTorch-like 1D reflection padding on the last dimension."""
    pad_left = x[..., 1:left+1][..., ::-1]
    pad_right = x[..., -right-1:-1][..., ::-1]
    return mx.concatenate([pad_left, x, pad_right], axis=-1)

def mlx_stft(x: mx.array, n_fft: int, hop_length: int, window: mx.array) -> mx.array:
    """Perform Short-Time Fourier Transform (STFT) natively on GPU using MLX.
    
    Matches PyTorch's torch.stft(..., center=True, normalized=True) exactly.
    Input shape: [B, C, T]
    Output shape: [B, C, freqs, frames]
    """
    B, C, T = x.shape
    le = int(math.ceil(T / hop_length))
    pad_amount = hop_length // 2 * 3
    
    # 1. Reflect pad (matching Demucs pre-STFT padding)
    left_pad = pad_amount
    right_pad = pad_amount + le * hop_length - T
    padded_x = reflect_pad_1d(x, left_pad, right_pad)
    
    # 2. Center pad (reflect pad of n_fft // 2 matching PyTorch center=True)
    center_pad = n_fft // 2
    padded_x_c = reflect_pad_1d(padded_x, center_pad, center_pad)
    
    # 3. Frame extraction via striding (constant time O(1) zero-copy)
    F_num = le + 4
    shape = (B, C, F_num, n_fft)
    strides = (C * padded_x_c.shape[-1], padded_x_c.shape[-1], hop_length, 1)
    frames = mx.as_strided(padded_x_c, shape=shape, strides=strides)
    
    # 4. Apply window
    windowed = frames * window
    
    # 5. FFT (normalized by sqrt(n_fft) matching PyTorch normalized=True)
    z = mx.fft.rfft(windowed) / math.sqrt(n_fft)
    
    # 6. Transpose to [B, C, freqs, frames]
    z = mx.transpose(z, (0, 1, 3, 2))
    
    # 7. Remove last frequency bin and slice time (matching Demucs _spec)
    z = z[..., :-1, :]
    z = z[..., 2:2+le]
    return z

def mlx_istft(z: mx.array, n_fft: int, hop_length: int, window: mx.array, length: int) -> mx.array:
    """Perform Inverse STFT natively on GPU using MLX OLA.
    
    Matches PyTorch's torch.istft(..., center=True, normalized=True) exactly.
    Input shape: [..., freqs, frames] (complex)
    Output shape: [..., length]
    """
    *batch_dims, freqs, frames = z.shape
    
    # 1. Pad freq: add back the last frequency bin (zeros)
    z_pad = mx.concatenate([z, mx.zeros((*batch_dims, 1, frames), dtype=z.dtype)], axis=-2)
    
    # 2. Pad time: add 2 frames on each side (matching Demucs _ispec)
    z_pad = mx.concatenate([
        mx.zeros((*batch_dims, freqs + 1, 2), dtype=z.dtype),
        z_pad,
        mx.zeros((*batch_dims, freqs + 1, 2), dtype=z.dtype),
    ], axis=-1)
    
    # 3. Transpose to [..., F_num, freqs]
    # For arbitrary batch dimensions:
    batch_rank = len(batch_dims)
    z_pad = mx.transpose(z_pad, (*range(batch_rank), batch_rank + 1, batch_rank))
    
    # 4. Inverse FFT (IRFFT) and multiply by sqrt(n_fft)
    windowed_frames = mx.fft.irfft(z_pad) * math.sqrt(n_fft)
    
    # 5. Apply synthesis window
    windowed_frames = windowed_frames * window
    
    # 6. Overlap-add (OLA) via 4-group parallel mapping
    F_num = windowed_frames.shape[-2]
    N = math.prod(batch_dims)
    wf_flat = windowed_frames.reshape(N, F_num, n_fft)
    
    overlap_factor = n_fft // hop_length
    target_length = hop_length * (F_num - 1) + n_fft
    
    reconstructed_signals = []
    for k in range(overlap_factor):
        idx = list(range(k, F_num, overlap_factor))
        group_frames = wf_flat[:, idx, :]
        group_sig = group_frames.reshape(N, -1)
        
        left_zeros = mx.zeros((N, k * hop_length))
        right_len = target_length - (left_zeros.shape[-1] + group_sig.shape[-1])
        right_zeros = mx.zeros((N, right_len))
        
        group_reconstructed = mx.concatenate([left_zeros, group_sig, right_zeros], axis=-1)
        reconstructed_signals.append(group_reconstructed)
        
    summed = sum(reconstructed_signals)
    
    # 7. Normalize by the window sum of squares
    window_sq = window ** 2
    ola_denom = mx.zeros((target_length,))
    for k in range(overlap_factor):
        idx = list(range(k, F_num, overlap_factor))
        group_win = mx.broadcast_to(window_sq, (len(idx), n_fft)).reshape(-1)
        
        left_z = mx.zeros((k * hop_length,))
        right_z = mx.zeros((target_length - (left_z.shape[-1] + group_win.shape[-1]),))
        
        group_denom = mx.concatenate([left_z, group_win, right_z], axis=-1)
        ola_denom = ola_denom + group_denom
        
    normalized = summed / mx.maximum(ola_denom, 1e-8)
    
    # 8. Trim center padding (n_fft // 2 + pad_amount)
    pad_amount = hop_length // 2 * 3
    total_left_trim = n_fft // 2 + pad_amount
    
    final_x = normalized[..., total_left_trim : total_left_trim + length]
    return final_x.reshape(*batch_dims, length)
