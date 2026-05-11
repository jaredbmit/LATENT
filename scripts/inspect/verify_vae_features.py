"""Verification checks for extracted VAE features.

Runs a suite of assertions and diagnostics to validate that
storage/data/vae/ outputs are numerically correct and consistent
with the raw motion data.

Usage:
    uv run python scripts/inspect/verify_vae_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RAW_ROOT  = Path("storage/data/mocap/Tennis")
VAE_ROOT  = Path("storage/data/vae")
FEAT_DIR  = VAE_ROOT / "features"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global failures
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures += 1


def warn(name: str, detail: str = "") -> None:
    print(f"  [{WARN}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Load outputs
# ---------------------------------------------------------------------------
print("=" * 64)
print("Loading VAE outputs")
print("=" * 64)

feat_files = sorted(FEAT_DIR.glob("*.npz"))
raw_files  = sorted(p for p in RAW_ROOT.rglob("*.npz") if "__MACOSX" not in p.parts)
check("feature files exist", len(feat_files) > 0, f"found {len(feat_files)}")
check("feature file count matches raw", len(feat_files) == len(raw_files),
      f"{len(feat_files)} vs {len(raw_files)}")

chunks_path = VAE_ROOT / "chunks_H64_s1.npz"
stats_path  = VAE_ROOT / "norm_stats.npz"
check("chunks file exists",    chunks_path.exists())
check("norm_stats file exists", stats_path.exists())

chunks = np.load(chunks_path)["chunks"]   # (N, H, D)
stats  = np.load(stats_path)
mean, std = stats["mean"], stats["std"]

feat_arrays = [np.load(f)["features"] for f in feat_files]
all_feats   = np.concatenate(feat_arrays, axis=0)

print(f"\n  chunks shape : {chunks.shape}")
print(f"  total frames : {all_feats.shape[0]:,}")

# ---------------------------------------------------------------------------
# 1. Shape checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("1. Shape and dtype")
print("=" * 64)

N, H, D = chunks.shape
check("chunk H = 64",   H == 64,   f"H={H}")
check("chunk D = 80",   D == 80,   f"D={D}")
check("chunk N > 0",    N > 0,     f"N={N}")
check("chunks dtype float32", chunks.dtype == np.float32, str(chunks.dtype))
check("mean shape (D,)",  mean.shape == (D,))
check("std shape (D,)",   std.shape  == (D,))

total_frames = sum(f.shape[0] for f in feat_arrays)
expected_N   = sum(max(f.shape[0] - H + 1, 0) for f in feat_arrays)
check("chunk count matches stride-1 formula", N == expected_N,
      f"got {N}, expected {expected_N}")

# ---------------------------------------------------------------------------
# 2. NaN / Inf
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("2. NaN / Inf")
print("=" * 64)

check("no NaN in features",  not np.isnan(all_feats).any())
check("no Inf in features",  not np.isinf(all_feats).any())
check("no NaN in chunks",    not np.isnan(chunks).any())
check("no Inf in chunks",    not np.isinf(chunks).any())
check("no NaN in mean",      not np.isnan(mean).any())
check("no zero std",         (std > 0).all(), f"min_std={std.min():.2e}")

# ---------------------------------------------------------------------------
# 3. Gravity vector is unit-norm in root frame
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("3. Gravity vector norm (should be 1.0)")
print("=" * 64)

grav = all_feats[:, 1:4]
norms = np.linalg.norm(grav, axis=1)
check("gravity norm mean ≈ 1.0", abs(norms.mean() - 1.0) < 1e-4,
      f"mean={norms.mean():.6f}")
check("gravity norm max error < 1e-4", (np.abs(norms - 1.0) < 1e-4).all(),
      f"max_err={np.abs(norms - 1.0).max():.2e}")

# ---------------------------------------------------------------------------
# 4. Cross-validate features against raw data (first file)
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("4. Cross-validation against raw qpos / site_xpos (file 0)")
print("=" * 64)

raw0  = np.load(raw_files[0], allow_pickle=True)
feat0 = feat_arrays[0]
qpos0 = np.asarray(raw0["qpos"], dtype=np.float64)

# Root height
check("root height == qpos z",
      np.allclose(feat0[:, 0], qpos0[:, 2], atol=1e-5),
      f"max_err={np.abs(feat0[:, 0] - qpos0[:, 2]).max():.2e}")

# Joint angles (feats 10:39 vs qpos 7:36)
check("joint angles match qpos",
      np.allclose(feat0[:, 10:39], qpos0[:, 7:], atol=1e-5),
      f"max_err={np.abs(feat0[:, 10:39] - qpos0[:, 7:]).max():.2e}")

# Gravity z-component in root frame: for an upright robot, gravity_z ≈ -1
# During tennis it varies, but the mean z should be negative.
check("gravity_z mean < 0 (robot mostly upright)",
      feat0[:, 3].mean() < 0,
      f"mean_gz={feat0[:, 3].mean():.3f}")

# Feet should be below root in root frame (z < 0) most of the time
left_foot_z  = feat0[:, 68 + 2]   # left_foot z in root frame
right_foot_z = feat0[:, 68 + 5]   # right_foot z in root frame
check("left foot mostly below root (z < 0)",
      (left_foot_z < 0).mean() > 0.95,
      f"{(left_foot_z < 0).mean()*100:.1f}% frames")
check("right foot mostly below root (z < 0)",
      (right_foot_z < 0).mean() > 0.95,
      f"{(right_foot_z < 0).mean()*100:.1f}% frames")

# Right palm height — during tennis strokes can be anywhere, just check finite
right_palm_z = feat0[:, 68 + 11]
check("right palm z finite", np.isfinite(right_palm_z).all())

# ---------------------------------------------------------------------------
# 5. Heading invariance: linvel heading should have near-zero lateral
#    mean when averaged over a long straight-motion stretch.
#    Weaker check: linvel magnitude in heading frame == global linvel magnitude
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("5. Heading-frame linear velocity")
print("=" * 64)

linvel_heading = feat0[:, 4:7]
linvel_global  = np.diff(qpos0[:, :3], axis=0) * 50.0
speed_feat     = np.linalg.norm(linvel_heading[:-1], axis=1)
speed_raw      = np.linalg.norm(linvel_global, axis=1)
check("linvel magnitude preserved under heading rotation",
      np.allclose(speed_feat, speed_raw, atol=1e-4),
      f"max_err={np.abs(speed_feat - speed_raw).max():.2e}")

# linvel_z in heading frame == global linvel_z (rotation is only around Z)
check("linvel z unchanged by heading rotation",
      np.allclose(linvel_heading[:-1, 2], linvel_global[:, 2], atol=1e-4),
      f"max_err={np.abs(linvel_heading[:-1, 2] - linvel_global[:, 2]).max():.2e}")

# ---------------------------------------------------------------------------
# 6. Chunk boundary integrity — no window crosses clip boundaries
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("6. Chunk boundary integrity")
print("=" * 64)

# Reconstruct expected chunk starts per clip and verify against stored chunks
clip_starts  = []
clip_offset  = 0
for feat in feat_arrays:
    T = feat.shape[0]
    for start in range(0, T - H + 1, 1):
        clip_starts.append((clip_offset, start))
    clip_offset += T

# First chunk should equal feat_arrays[0][0:H]
check("first chunk matches first clip frames 0:H",
      np.allclose(chunks[0], feat_arrays[0][:H], atol=1e-6))

# Last chunk of clip 0 should equal feat_arrays[0][T0-H:T0]
T0 = feat_arrays[0].shape[0]
n_chunks_clip0 = T0 - H + 1
check("last chunk of clip 0 matches clip 0 tail",
      np.allclose(chunks[n_chunks_clip0 - 1], feat_arrays[0][T0 - H:], atol=1e-6))

# First chunk of clip 1 should equal feat_arrays[1][0:H]
check("first chunk of clip 1 matches clip 1 head",
      np.allclose(chunks[n_chunks_clip0], feat_arrays[1][:H], atol=1e-6))

# Consecutive chunks within a clip differ only by a 1-frame shift
# (overlapping region should be identical)
c0 = chunks[0]          # frames [0, H)
c1 = chunks[1]          # frames [1, H+1)
check("consecutive chunks overlap correctly (stride=1)",
      np.allclose(c0[1:], c1[:-1], atol=1e-6))

# ---------------------------------------------------------------------------
# 7. Normalisation stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("7. Normalisation stats")
print("=" * 64)

computed_mean = all_feats.mean(axis=0)
computed_std  = all_feats.std(axis=0)

check("saved mean matches computed",
      np.allclose(mean, computed_mean, atol=1e-4),
      f"max_err={np.abs(mean - computed_mean).max():.2e}")
check("saved std matches computed",
      np.allclose(std, np.clip(computed_std, 1e-6, None), atol=1e-4),
      f"max_err={np.abs(std - np.clip(computed_std, 1e-6, None)).max():.2e}")

normed = (all_feats - mean) / std
check("normalised mean ≈ 0", np.abs(normed.mean(axis=0)).max() < 1e-3,
      f"max={np.abs(normed.mean(axis=0)).max():.2e}")
check("normalised std ≈ 1",  np.abs(normed.std(axis=0) - 1).max() < 1e-3,
      f"max={np.abs(normed.std(axis=0) - 1).max():.2e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
if failures == 0:
    print(f"All checks passed.")
else:
    print(f"{failures} check(s) FAILED.")
print("=" * 64)
