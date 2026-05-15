from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionChunkDataset(Dataset):
    """
    On-the-fly dataset over per-clip feature files.

    Loads all clips into memory as (T, D) normalised tensors.  Windows of
    length H+1 are constructed on the fly so no pre-chunked file is needed
    and H can be changed freely without re-preprocessing.

    Each item is:
      chunk : (H, D) float32 — normalised future frames (prediction target)
      s_t   : (D,)  float32 — conditioning state (frame immediately before chunk)
    """

    def __init__(self, features_dir: Path, norm_stats_path: Path, H: int) -> None:
        self.H = H
        stats      = np.load(norm_stats_path)
        mean, std  = stats["mean"], stats["std"]   # (D,)
        self.mean  = mean.astype(np.float32)
        self.std   = std.astype(np.float32)

        self.clips: list[torch.Tensor] = []
        offsets = [0]

        for f in sorted(Path(features_dir).glob("*.npz")):
            raw    = np.load(f)["features"].astype(np.float32)   # (T, D)
            normed = torch.from_numpy((raw - mean) / std)
            self.clips.append(normed)
            Tp = max(0, normed.shape[0] - H)   # valid start positions (0..Tp-1)
            offsets.append(offsets[-1] + Tp)

        self.offsets = offsets
        self.D = self.clips[0].shape[1] if self.clips else 0

        # Per-feature std of one-step deltas — used to renormalise s_{t+1}-s_t
        # for the delta-mode isometry loss (deltas are small and not unit var).
        deltas = torch.cat([c[1:] - c[:-1] for c in self.clips], dim=0)
        self.delta_std = deltas.std(dim=0).clamp_min(1e-6)   # (D,)

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_idx = bisect.bisect_right(self.offsets, idx) - 1
        t        = idx - self.offsets[clip_idx]
        clip     = self.clips[clip_idx]           # (T, D)
        s_t      = clip[t]                        # (D,)
        chunk    = clip[t + 1 : t + self.H + 1]  # (H, D)
        return chunk, s_t
