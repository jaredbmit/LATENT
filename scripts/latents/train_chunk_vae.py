"""Train the motion chunk VAE (1D temporal conv, unconditional).

Reads:
  storage/data/vae/features/*.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/runs/<run_name>/model.pt
  storage/runs/<run_name>/config.json

Usage:
  uv run python scripts/latents/train_chunk_vae.py
  uv run python scripts/latents/train_chunk_vae.py --run_name cvae_d32_l25 --latent_dim 32 --latent_len 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from motion_latent.paths import FEAT_DIR, RUNS_ROOT, STATS_PATH
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.chunk_vae.model import ChunkVAE


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset  = MotionChunkDataset(args.features, args.stats, H=args.H)
    D        = dataset.D
    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=2)
    print(f"clips: {dataset.n_clips}  items: {len(dataset)}  H={args.H}  D={D}")

    model = ChunkVAE(D=D, H=args.H, latent_len=args.latent_len,
                     latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                     kernel_size=args.kernel_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}  latent_len={args.latent_len}  latent_dim={args.latent_dim}  "
          f"kernel_size={args.kernel_size}  n_up={model.n_up}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    out_dir = RUNS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(1.0, epoch / args.beta_warmup) if args.beta_warmup > 0 else args.beta

        model.train()
        recon_sum = kl_sum = 0.0
        for chunk, _ in train_dl:
            chunk = chunk.to(device)                           # (B, H, D)
            recon, mu, lv = model(chunk)
            recon_loss = F.mse_loss(recon, chunk)
            kl_loss    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss       = recon_loss + beta * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_sum += recon_loss.item()
            kl_sum    += kl_loss.item()
        scheduler.step()

        if epoch % args.log_every == 0:
            n = len(train_dl)
            print(
                f"epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.2e}  beta={beta:.4f}  "
                f"recon={recon_sum/n:.4f}  kl={kl_sum/n:.4f}"
            )

    torch.save(model.state_dict(), out_dir / "model.pt")
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg["D"] = D
    cfg["model_type"] = "chunk_vae"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",    type=Path,  default=FEAT_DIR)
    ap.add_argument("--stats",       type=Path,  default=STATS_PATH)
    ap.add_argument("--run_name",    type=str,   default="v2/cvae_base")
    ap.add_argument("--H",           type=int,   default=100,
                    help="Chunk length in frames.")
    ap.add_argument("--latent_len",  type=int,   default=25,
                    help="Temporal length of the latent sequence.")
    ap.add_argument("--latent_dim",  type=int,   default=16,
                    help="Feature dimension of the latent sequence.")
    ap.add_argument("--hidden_dim",  type=int,   default=128,
                    help="Intermediate conv feature dimension.")
    ap.add_argument("--kernel_size", type=int,   default=9,
                    help="Conv kernel size (must be odd). Larger = more temporal context.")
    ap.add_argument("--epochs",      type=int,   default=500)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--beta",        type=float, default=1e-3,
                    help="KL weight.")
    ap.add_argument("--beta_warmup", type=int,   default=100,
                    help="Epochs to linearly ramp beta from 0. 0 disables warmup.")
    ap.add_argument("--log_every",   type=int,   default=25)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
