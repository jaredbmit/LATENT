"""Train the motion VAE.

Reads:
  storage/data/vae/chunks_H64_s1.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/data/latents/<run_name>/model.pt
  storage/data/latents/<run_name>/config.json

Usage:
  uv run python scripts/latents/train_vae.py
  uv run python scripts/latents/train_vae.py --latent_dim 32 --beta 0.5 --run_name vae_z32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.model import MotionVAE

CHUNKS_PATH = Path("storage/data/vae/chunks_H64_s1.npz")
STATS_PATH  = Path("storage/data/vae/norm_stats.npz")
OUT_ROOT    = Path("storage/data/latents")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- Data ---
    dataset  = MotionChunkDataset(args.chunks, args.stats)
    H, D     = dataset.chunks.shape[1], dataset.chunks.shape[2]
    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"chunks: {len(dataset)}  H={H}  D={D}")

    # --- Model ---
    model = MotionVAE(D=D, H=H, latent_dim=args.latent_dim, hidden=args.hidden, variant=args.variant).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # --- Output dir ---
    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(1.0, epoch / args.beta_warmup) if args.beta_warmup > 0 else args.beta

        model.train()
        train_recon = train_kl = train_iso = 0.0
        for chunk, s_t in train_dl:
            chunk, s_t = chunk.to(device), s_t.to(device)
            loss, metrics = model.loss(chunk, s_t, beta=beta, alpha=args.alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_recon += metrics["recon"]
            train_kl    += metrics["kl"]
            train_iso   += metrics["iso"]
        scheduler.step()

        if epoch % args.log_every == 0:
            n = len(train_dl)
            print(
                f"epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.2e}  beta={beta:.3f}  "
                f"recon={train_recon/n:.4f}  kl={train_kl/n:.4f}  iso={train_iso/n:.4f}"
            )

    # --- Save ---
    torch.save(model.state_dict(), out_dir / "model.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"H": H, "D": D})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks",     type=Path,  default=CHUNKS_PATH)
    ap.add_argument("--stats",      type=Path,  default=STATS_PATH)
    ap.add_argument("--run_name",   type=str,   default="vae_z16")
    ap.add_argument("--variant",    type=str,   default="hybrid",
                    choices=["conditional", "unconditional", "hybrid"])
    ap.add_argument("--latent_dim", type=int,   default=16)
    ap.add_argument("--hidden",     type=int,   default=128)
    ap.add_argument("--epochs",     type=int,   default=500)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--beta",        type=float, default=1.0)
    ap.add_argument("--alpha",       type=float, default=0.0,
                    help="Weight on pairwise isometry loss. 0 disables.")
    ap.add_argument("--beta_warmup", type=int,   default=100,
                    help="Epochs to linearly ramp beta from 0 to --beta. 0 disables warmup.")
    ap.add_argument("--log_every",  type=int,   default=25)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
