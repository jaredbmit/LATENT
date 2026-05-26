"""Train a latent diffusion model (DDPM, DiT denoiser) over ChunkVAE latents.

Reads:
  storage/runs/<vae_run>/chunk_diffusion_dataset.npz
    latents : (N, latent_len, latent_dim)

Writes:
  storage/runs/<run_name>/model.pt
  storage/runs/<run_name>/norm_stats.npz  (channel-wise mean/std, shape (latent_dim,))
  storage/runs/<run_name>/config.json

Usage:
  uv run python scripts/diffusion/train_diffusion.py
  uv run python scripts/diffusion/train_diffusion.py --run_name diff_base --epochs 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import time

import numpy as np
import torch
import torch.nn.functional as F

from motion_latent.paths import RUNS_ROOT
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.schedule import cosine_schedule


class EMA:
    """Exponential moving average of model weights, kept for sampling only.

    The optimizer trains the raw weights; this maintains a parallel running
    average that is never fed back into training. Sample/save from `.shadow`.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)   # buffers like int counters: just track


def train(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.fp32_precision   = 'tf32'  # enable TF32 on Ampere+
    torch.backends.cudnn.conv.fp32_precision    = 'tf32'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- Load latents ---
    dataset_path = RUNS_ROOT / args.vae_run / "chunk_diffusion_dataset.npz"
    data    = np.load(dataset_path)
    latents = data["latents"].astype(np.float32)    # (N, latent_len, latent_dim)

    latent_len, latent_dim = latents.shape[1], latents.shape[2]
    print(f"latents: {latents.shape}  latent_len={latent_len}  latent_dim={latent_dim}")

    # Channel-wise normalisation across all samples and positions.
    flat        = latents.reshape(-1, latent_dim)
    latent_mean = flat.mean(axis=0)                     # (latent_dim,)
    latent_std  = np.maximum(flat.std(axis=0), 1e-4)   # (latent_dim,)

    z_norm = (latents - latent_mean) / latent_std      # (N, latent_len, latent_dim)
    print(f"normalised: mean={z_norm.mean():.4f}  std={z_norm.std():.4f}")

    # Dataset fits in GPU memory — keep it resident to avoid CPU→GPU transfers per batch.
    z_gpu     = torch.from_numpy(z_norm).float().to(device)
    n_samples = z_gpu.shape[0]
    n_batches = n_samples // args.batch_size

    # --- Schedule ---
    schedule = cosine_schedule(args.T)

    # --- Model ---
    model = MotionDiT(
        latent_len=latent_len, latent_dim=latent_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ff_mult=args.ff_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    # model = torch.compile(model)

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ema       = EMA(model, args.ema_decay)

    ab = schedule.alphas_bar.to(device)   # (T+1,)

    out_dir = RUNS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    steps_since_log = 0

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        perm = torch.randperm(n_samples, device=device)
        for bi in range(n_batches):
            z0 = z_gpu[perm[bi * args.batch_size : (bi + 1) * args.batch_size]]
            B  = z0.shape[0]

            t   = torch.randint(1, args.T + 1, (B,), device=device)
            eps = torch.randn_like(z0)
            ab_t = ab[t].view(B, 1, 1)
            z_t  = ab_t.sqrt() * z0 + (1 - ab_t).sqrt() * eps

            with torch.autocast('cuda', dtype=torch.bfloat16):
                x0_hat = model(z_t, t)
                loss   = F.mse_loss(x0_hat, z0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            loss_sum += loss.item()
            steps_since_log += 1

        scheduler.step()

        if epoch % args.log_every == 0:
            elapsed     = time.perf_counter() - t0
            ms_per_step = 1000 * elapsed / steps_since_log
            print(f"epoch {epoch:5d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"loss={loss_sum/n_batches:.5f}  {ms_per_step:.1f}ms/step")
            t0 = time.perf_counter()
            steps_since_log = 0

    # --- Save ---
    # EMA weights are the ones to sample from; keep the raw weights alongside.
    torch.save(ema.shadow,         out_dir / "model.pt")
    torch.save(model.state_dict(), out_dir / "model_raw.pt")
    np.savez(out_dir / "norm_stats.npz", mean=latent_mean, std=latent_std)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({
        "latent_len":  latent_len,
        "latent_dim":  latent_dim,
        "model_type":  "motion_dit",
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae_run",   type=str, default="v2/cvae_v2",
                    help="ChunkVAE run dir containing chunk_diffusion_dataset.npz")
    ap.add_argument("--run_name",  type=str, default="v2/diff_base")
    ap.add_argument("--T",         type=int, default=1000,
                    help="Number of diffusion timesteps.")
    ap.add_argument("--d_model",   type=int, default=256)
    ap.add_argument("--n_heads",   type=int, default=4)
    ap.add_argument("--n_layers",  type=int, default=8)
    ap.add_argument("--ff_mult",   type=int, default=4)
    ap.add_argument("--epochs",    type=int, default=2000)
    ap.add_argument("--batch_size",type=int, default=256)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay for the sampling weights.")
    ap.add_argument("--log_every", type=int, default=100)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
