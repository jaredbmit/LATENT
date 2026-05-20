"""Train a diffusion model (DiT) directly on normalised motion feature chunks.

This is the raw-space counterpart to train_diffusion.py which operates in the
ChunkVAE latent space. Both use the same MotionDiT architecture; the only
difference is the input tensor:
  latent diffusion : latents from VAE encoder  (N, latent_len=25, latent_dim=16)
  raw diffusion    : normalised motion chunks   (N, H=100,         D=64)

Conditioning modes (--cond_mode):
  none    — unconditional generation (default, matches rdiff_base behaviour)
  inpaint — mask the first n_cond frames as known; model denoises [n_cond:]
            conditioned on noised versions of the context (replacement inpainting)
  prepend — prepend clean context frames; model denoises [n_cond:] attending
            to the unnoised prefix

In both conditional modes --n_cond sets the number of conditioning frames N.
The model's latent_len is set to N + H so it processes the full [cond|chunk]
sequence.  latent_mean / latent_std in config.json are also (N+H, D).

Reads:
  storage/data/vae/features/*.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/data/latents/<run_name>/model.pt
  storage/data/latents/<run_name>/model_raw.pt
  storage/data/latents/<run_name>/config.json   (model_type: "motion_dit_raw")

Usage:
  uv run python scripts/diffusion/train_raw_diffusion.py
  uv run python scripts/diffusion/train_raw_diffusion.py --run_name rdiff_base --epochs 2000
  uv run python scripts/diffusion/train_raw_diffusion.py --run_name v2/rdiff_inpaint \
      --cond_mode inpaint --n_cond 10 --epochs 2000
  uv run python scripts/diffusion/train_raw_diffusion.py --run_name v2/rdiff_prepend \
      --cond_mode prepend --n_cond 10 --epochs 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from motion_latent.paths import FEAT_DIR, LATENTS_ROOT, STATS_PATH
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.schedule import cosine_schedule


class EMA:
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
                s.copy_(v)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cond_mode={args.cond_mode}  n_cond={args.n_cond}")

    N = args.n_cond   # conditioning frames
    H = args.H        # generative frames

    # --- Load motion chunks and conditioning frames ---
    dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H, n_cond=max(N, 1))
    D = dataset.D
    print(f"clips: {dataset.n_clips}  items: {len(dataset)}  H={H}  D={D}  N={N}")

    # Build full sequences (N + H, D): [cond | chunk]
    # For unconditional (N=0), seqs = chunks only.
    all_seqs = []
    for i in range(len(dataset)):
        chunk, cond = dataset[i]       # (H, D), (n_cond, D)
        if N > 0:
            seq = torch.cat([cond[-N:], chunk], dim=0)   # (N+H, D)
        else:
            seq = chunk                                   # (H, D)
        all_seqs.append(seq.numpy())
    all_seqs = np.stack(all_seqs)      # (n_samples, N+H, D)
    n_samples, seq_len, _ = all_seqs.shape
    print(f"sequences: {n_samples}  seq_len={seq_len}")

    # Per-position standardisation across the full [cond|chunk] window.
    state_mean = all_seqs.mean(axis=0)              # (seq_len, D)
    state_std  = np.maximum(all_seqs.std(axis=0), 1e-4)
    z_norm     = (all_seqs - state_mean) / state_std
    print(f"normalised: mean={z_norm.mean():.4f}  std={z_norm.std():.4f}")

    loader = DataLoader(TensorDataset(torch.from_numpy(z_norm).float()),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- Schedule & model ---
    # latent_len = N + H so the model processes the full [cond|chunk] sequence
    schedule = cosine_schedule(args.T)
    model    = MotionDiT(
        latent_len=seq_len, latent_dim=D,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ff_mult=args.ff_mult,
    ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ema       = EMA(model, args.ema_decay)
    ab        = schedule.alphas_bar.to(device)

    out_dir = LATENTS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for (z0,) in loader:
            z0  = z0.to(device)           # (B, seq_len, D)
            B   = z0.shape[0]
            t   = torch.randint(1, args.T + 1, (B,), device=device)
            ab_t = ab[t].view(B, 1, 1)

            if args.cond_mode == "none":
                # --- unconditional ---
                eps     = torch.randn_like(z0)
                z_t     = ab_t.sqrt() * z0 + (1 - ab_t).sqrt() * eps
                eps_hat = model(z_t, t)
                loss    = F.mse_loss(eps_hat, eps)

            elif args.cond_mode == "inpaint":
                # --- replacement inpainting ---
                # Noise the full sequence, then independently re-noise the cond
                # prefix so it's consistent with the replacement operation at
                # inference (ddim_inpaint_sample).
                eps      = torch.randn_like(z0)                          # (B, N+H, D)
                z_t      = ab_t.sqrt() * z0 + (1 - ab_t).sqrt() * eps

                eps_cond = torch.randn(B, N, D, device=device)
                z_t[:, :N] = ab_t.sqrt() * z0[:, :N] + (1 - ab_t).sqrt() * eps_cond

                eps_hat  = model(z_t, t)
                loss     = F.mse_loss(eps_hat[:, N:], eps[:, N:])        # loss on [N:] only

            elif args.cond_mode == "prepend":
                # --- clean prefix prepended ---
                # The conditioning tokens [0:N] are always passed clean;
                # only the generative tokens [N:] are noised.
                cond  = z0[:, :N]                                        # (B, N, D) clean
                chunk = z0[:, N:]                                        # (B, H, D)
                eps   = torch.randn_like(chunk)
                z_t_chunk = ab_t.sqrt() * chunk + (1 - ab_t).sqrt() * eps
                z_in      = torch.cat([cond, z_t_chunk], dim=1)          # (B, N+H, D)
                eps_hat   = model(z_in, t)
                loss      = F.mse_loss(eps_hat[:, N:], eps)              # loss on [N:] only

            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)
            loss_sum += loss.item()
        scheduler.step()

        if epoch % args.log_every == 0:
            print(f"epoch {epoch:5d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"loss={loss_sum/len(loader):.5f}")

    torch.save(ema.shadow,         out_dir / "model.pt")
    torch.save(model.state_dict(), out_dir / "model_raw.pt")

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({
        "model_type":  "motion_dit_raw",
        "latent_len":  seq_len,   # N + H — full sequence length the model processes
        "H":           H,         # generative frames
        "n_cond":      N,         # conditioning frames
        "latent_dim":  D,
        "latent_mean": state_mean.tolist(),   # (seq_len, D) — includes cond positions
        "latent_std":  state_std.tolist(),
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_name",   type=str,   default="v2/rdiff_base")
    ap.add_argument("--H",          type=int,   default=100,
                    help="Chunk length in frames (should match ChunkVAE training).")
    ap.add_argument("--n_cond",     type=int,   default=0,
                    help="Number of conditioning frames prepended (0 = unconditional).")
    ap.add_argument("--cond_mode",  type=str,   default="none",
                    choices=["none", "inpaint", "prepend"],
                    help="Conditioning mechanism: none | inpaint | prepend.")
    ap.add_argument("--T",          type=int,   default=1000)
    ap.add_argument("--d_model",    type=int,   default=256)
    ap.add_argument("--n_heads",    type=int,   default=4)
    ap.add_argument("--n_layers",   type=int,   default=8)
    ap.add_argument("--ff_mult",    type=int,   default=4)
    ap.add_argument("--epochs",     type=int,   default=2000)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--ema_decay",  type=float, default=0.999)
    ap.add_argument("--log_every",  type=int,   default=100)
    args = ap.parse_args()

    if args.cond_mode != "none" and args.n_cond == 0:
        ap.error(f"--cond_mode {args.cond_mode} requires --n_cond > 0")
    if args.cond_mode == "none" and args.n_cond > 0:
        ap.error("--n_cond > 0 requires --cond_mode inpaint or prepend")

    train(args)


if __name__ == "__main__":
    main()
