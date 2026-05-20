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
      chunk : (H, D)      float32 — normalised future frames (prediction target)
      cond  : (n_cond, D) float32 — n_cond frames immediately preceding the chunk
                                    (padded by repeating the first available frame
                                     when the clip is too short)
    """

    def __init__(self, features_dir: Path, norm_stats_path: Path,
                 H: int, n_cond: int = 1) -> None:
        self.H      = H
        self.n_cond = n_cond
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

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_idx = bisect.bisect_right(self.offsets, idx) - 1
        t        = idx - self.offsets[clip_idx]
        clip     = self.clips[clip_idx]           # (T, D)
        chunk    = clip[t + 1 : t + self.H + 1]  # (H, D)

        # n_cond frames ending at t (inclusive); pad at start if clip is too short
        start  = max(0, t - self.n_cond + 1)
        frames = clip[start : t + 1]              # (min(n_cond, t+1), D)
        if frames.shape[0] < self.n_cond:
            pad   = frames[:1].expand(self.n_cond - frames.shape[0], -1)
            frames = torch.cat([pad, frames], dim=0)
        cond = frames                             # (n_cond, D)
        return chunk, cond
