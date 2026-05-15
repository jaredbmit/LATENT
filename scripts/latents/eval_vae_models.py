"""Evaluate trained VAE checkpoints and compare across runs.

For each run reports (in normalised feature space):

  One-step metrics (teacher-forced, H=1) — comparable across all models:
    recon_1      MSE from posterior mean
    kl_1         KL(q || p)
    iso_1        pairwise isometry loss on posterior mean
    prior_1      MSE decoding from prior mean (noise-free)

  Multi-step free-running metrics (p=0, H=L from config):
    recon_L      mean per-step MSE over an L-step free-running rollout
    iso_L        mean per-step isometry loss
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from motion_latent.paths import LATENTS_ROOT
from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.model import MotionVAE, isometry_loss, kl_two_gaussians

PATTERN = "mvae_cond_*_L[0-9]*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_one_step(model: MotionVAE, cfg: dict) -> dict:
    ds = MotionChunkDataset(cfg["features"], cfg["stats"], H=1)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    recon = kl = iso = prior = n = 0.0
    with torch.no_grad():
        for chunk, s_t in dl:
            s_t    = s_t.to(device)
            s_next = chunk[:, 0].to(device)
            r, mu_q, lv_q, mu_p, lv_p = model(s_next, s_t)
            b = len(s_t)
            recon += F.mse_loss(r, s_next).item() * b
            kl    += kl_two_gaussians(mu_q, lv_q, mu_p, lv_p).item() * b
            iso   += isometry_loss(mu_q, s_next, model.latent_dim).item() * b
            r_p    = model.decode(model.prior_mean(s_t), s_t)
            prior += F.mse_loss(r_p, s_next).item() * b
            n += b
    return {"recon_1": recon / n, "kl_1": kl / n, "iso_1": iso / n, "prior_1": prior / n}


def eval_rollout(model: MotionVAE, cfg: dict) -> dict:
    """Free-running (p=0) L-step rollout evaluation."""
    L = cfg.get("L", 1)
    ds = MotionChunkDataset(cfg["features"], cfg["stats"], H=L)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    delta_std = ds.delta_std.to(device) if cfg.get("iso_mode") == "delta" else None

    recon_sum = iso_sum = n = 0.0
    with torch.no_grad():
        for chunk, s_t in dl:
            chunk, s_t = chunk.to(device), s_t.to(device)
            B = len(s_t)
            _, metrics = model.rollout_loss(
                chunk, s_t,
                p=0.0,
                beta=cfg.get("beta", 0.001),
                alpha=cfg.get("alpha", 0.0),
                iso_mode=cfg.get("iso_mode", "target"),
                delta_std=delta_std,
            )
            recon_sum += metrics["recon"] * B
            iso_sum   += metrics["iso"] * B
            n += B
    return {"recon_L": recon_sum / n, "iso_L": iso_sum / n}


rows = []
for run_dir in sorted(LATENTS_ROOT.glob(PATTERN)):
    if not (run_dir / "model.pt").exists():
        print(f"SKIP {run_dir.name} — no model.pt")
        continue
    print(f"evaluating {run_dir.name} ...", flush=True)
    model, cfg = MotionVAE.from_run(run_dir, device)
    m1 = eval_one_step(model, cfg)
    mL = eval_rollout(model, cfg)
    rows.append({
        "run": run_dir.name,
        "L": cfg.get("L", 1),
        "residual": cfg.get("residual", False),
        "alpha": cfg.get("alpha", 0.0),
        "iso_mode": cfg.get("iso_mode", "-"),
        **m1,
        **mL,
    })

# Sort: L first, then residual, then alpha
rows.sort(key=lambda r: (r["L"], r["residual"], r["alpha"], r["iso_mode"]))

hdr = f"{'run':<42} {'L':>2} {'res':>3} {'alpha':>5} {'iso_mode':>9} | {'recon_1':>8} {'kl_1':>8} {'iso_1':>8} {'prior_1':>8} | {'recon_L':>8} {'iso_L':>8}"
print()
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['run']:<42} {r['L']:>2} {str(r['residual'])[0]:>3} {r['alpha']:>5.2f} {r['iso_mode']:>9} | "
        f"{r['recon_1']:>8.4f} {r['kl_1']:>8.4f} {r['iso_1']:>8.4f} {r['prior_1']:>8.4f} | "
        f"{r['recon_L']:>8.4f} {r['iso_L']:>8.4f}"
    )
