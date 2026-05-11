from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionChunkDataset(Dataset):
    """
    Wraps storage/data/vae/chunks_H<H>_s<s>.npz.

    Each item is (chunk, s_t) where:
      chunk : (H, D) float32 — normalised state sequence
      s_t   : (D,)  float32 — first frame, used as prior input
    """

    def __init__(self, chunks_path: Path, norm_stats_path: Path) -> None:
        raw   = np.load(chunks_path)["chunks"]                # (N, H, D)
        stats = np.load(norm_stats_path)
        mean, std = stats["mean"], stats["std"]               # (D,)
        normed = (raw - mean) / std
        self.chunks = torch.from_numpy(normed).float()        # (N, H, D)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]   # (H, D)
        s_t   = chunk[0]           # (D,)
        return chunk, s_t
